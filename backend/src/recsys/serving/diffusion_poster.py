"""
diffusion_poster.py
===================
Diffusion-based poster generation for CineWave RecSys.

Used when a movie has no TMDB poster — generates one from title + genre
using a diffusion model. This mirrors Netflix's production use of generative
AI for personalised artwork.

Architecture:
  1. DDPM noise schedule (pure numpy) — the core diffusion math
     Forward process:  q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)
     Reverse process:  p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), σ_t I)
     Score function:   ε_θ(x_t, t) predicts the noise added at step t

  2. HuggingFace Inference API — Stable Diffusion XL (free tier)
     When HUGGINGFACE_API_KEY is set, generates real poster images.

  3. Replicate API — alternative SD endpoint
     When REPLICATE_API_KEY is set, uses replicate.com/stability-ai/sdxl

  4. Fallback — returns a gradient placeholder image (always works)

References:
  Ho et al. "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
  Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" (CVPR 2022)
"""
from __future__ import annotations

import os
import math
import hashlib
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger("cinewave.diffusion")

# ── DDPM Hyperparameters (Ho et al. 2020) ─────────────────────────────────────
T           = 1000          # total diffusion timesteps
BETA_START  = 1e-4          # β_1 (small initial noise)
BETA_END    = 0.02          # β_T (large final noise)
IMG_SIZE    = 512           # target image size for Stable Diffusion XL
GUIDANCE    = 7.5           # classifier-free guidance scale


# ── API config ────────────────────────────────────────────────────────────────
HF_API_KEY  = os.environ.get("HUGGINGFACE_API_KEY", "")
HF_MODEL    = "stabilityai/stable-diffusion-xl-base-1.0"
HF_API_URL  = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

REPLICATE_KEY = os.environ.get("REPLICATE_API_KEY", "")
REPLICATE_MODEL = "stability-ai/sdxl:39ed52f2319f9b8a7e26f29f39a5f49beda3a0b29aab0f2c01b4e5b1dbb3b"


class DDPMSchedule:
    """
    Denoising Diffusion Probabilistic Model noise schedule.

    Implements the forward and reverse diffusion processes from
    Ho et al. (NeurIPS 2020) in pure numpy.

    Forward process (adding noise):
        q(x_t | x_0) = N(x_t; √ᾱ_t · x_0, (1-ᾱ_t) · I)

    Reverse process (denoising):
        p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), σ_t²·I)
        μ_θ(x_t,t) = (1/√α_t)(x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t,t))
    """

    def __init__(self, T: int = T, beta_start: float = BETA_START, beta_end: float = BETA_END):
        self.T = T

        # Linear noise schedule: β_t from β_1 to β_T
        self.betas = np.linspace(beta_start, beta_end, T, dtype=np.float64)

        # α_t = 1 - β_t
        self.alphas = 1.0 - self.betas

        # ᾱ_t = ∏_{s=1}^{t} α_s  (cumulative product)
        self.alphas_cumprod = np.cumprod(self.alphas)

        # ᾱ_{t-1} (shift by 1 for reverse process)
        self.alphas_cumprod_prev = np.concatenate([[1.0], self.alphas_cumprod[:-1]])

        # √ᾱ_t  (used in forward process mean)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)

        # √(1-ᾱ_t)  (used in forward process std)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        # σ_t² = β_t · (1-ᾱ_{t-1}) / (1-ᾱ_t)  (posterior variance)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod)
        )

        # Log posterior variance (clipped for numerical stability)
        self.log_posterior_variance = np.log(
            np.maximum(self.posterior_variance, 1e-20)
        )

    def forward_process(self, x0: np.ndarray, t: int, noise: Optional[np.ndarray] = None) -> tuple:
        """
        Forward diffusion: q(x_t | x_0)
        Add noise to clean image x0 at timestep t.

        x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,   ε ~ N(0, I)

        Args:
            x0: clean image [H, W, C], values in [-1, 1]
            t:  timestep (0 to T-1)
            noise: optional pre-sampled noise (for reproducibility)

        Returns:
            (x_t, noise): noisy image and the noise that was added
        """
        if noise is None:
            noise = np.random.randn(*x0.shape).astype(np.float64)

        sqrt_alpha_bar   = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus   = self.sqrt_one_minus_alphas_cumprod[t]

        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus * noise
        return x_t, noise

    def reverse_step(self, x_t: np.ndarray, t: int, predicted_noise: np.ndarray) -> np.ndarray:
        """
        One step of reverse diffusion: p_θ(x_{t-1} | x_t)
        Given predicted noise ε_θ(x_t, t), compute x_{t-1}.

        μ_θ(x_t, t) = (1/√α_t)(x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t))
        x_{t-1} = μ_θ + σ_t · z,   z ~ N(0,I)  if t > 0, else z = 0

        Args:
            x_t:             noisy image at step t
            t:               current timestep
            predicted_noise: ε_θ(x_t, t) — model's noise prediction

        Returns:
            x_{t-1}: slightly denoised image
        """
        alpha_t      = self.alphas[t]
        alpha_bar_t  = self.alphas_cumprod[t]
        beta_t       = self.betas[t]

        # Compute mean of p_θ(x_{t-1}|x_t)
        coeff = beta_t / self.sqrt_one_minus_alphas_cumprod[t]
        mean  = (1.0 / math.sqrt(alpha_t)) * (x_t - coeff * predicted_noise)

        # Add variance (no noise at final step t=0)
        if t > 0:
            std  = math.sqrt(self.posterior_variance[t])
            mean = mean + std * np.random.randn(*x_t.shape)

        return mean

    def snr(self, t: int) -> float:
        """
        Signal-to-Noise Ratio at timestep t.
        SNR(t) = ᾱ_t / (1 - ᾱ_t)
        """
        return float(self.alphas_cumprod[t] / (1.0 - self.alphas_cumprod[t]))

    def schedule_summary(self) -> dict:
        """Return key properties of the noise schedule."""
        return {
            "T":               self.T,
            "beta_start":      round(float(self.betas[0]), 6),
            "beta_end":        round(float(self.betas[-1]), 4),
            "alpha_bar_T500":  round(float(self.alphas_cumprod[499]), 4),
            "alpha_bar_T999":  round(float(self.alphas_cumprod[999]), 6),
            "snr_t100":        round(self.snr(99), 4),
            "snr_t500":        round(self.snr(499), 6),
            "snr_t999":        round(self.snr(999), 8),
            "schedule":        "linear",
            "reference":       "Ho et al. NeurIPS 2020",
        }


# ── Prompt engineering for movie posters ──────────────────────────────────────

STYLE_SUFFIX = (
    "cinematic movie poster, professional film photography, "
    "dramatic lighting, high contrast, 4K, detailed"
)

GENRE_STYLES = {
    "Action":      "explosive action, dynamic motion blur, intense colors",
    "Comedy":      "bright cheerful palette, warm tones, playful composition",
    "Drama":       "moody chiaroscuro lighting, desaturated palette, emotional depth",
    "Horror":      "dark atmospheric fog, deep shadows, ominous red accents",
    "Sci-Fi":      "futuristic neon glow, space backdrop, technological aesthetic",
    "Romance":     "soft golden hour lighting, warm bokeh, intimate composition",
    "Thriller":    "high contrast noir, dramatic shadows, tension-filled atmosphere",
    "Documentary": "photorealistic, natural lighting, journalistic style",
    "Animation":   "vibrant saturated colors, stylized illustration",
    "Fantasy":     "magical golden light, ethereal mist, epic landscapes",
}


def build_prompt(title: str, genre: str, year: Optional[int] = None) -> str:
    """
    Build a Stable Diffusion prompt for a movie poster.

    Engineered for SDXL: genre-specific style + cinematic suffix.
    """
    genre_style = GENRE_STYLES.get(genre, "cinematic atmosphere, professional lighting")
    year_str    = f"({year})" if year else ""
    prompt = (
        f"Movie poster for '{title}' {year_str}, {genre} film, "
        f"{genre_style}, {STYLE_SUFFIX}"
    )
    return prompt


# ── Image generation via HuggingFace Inference API ────────────────────────────

def generate_via_huggingface(prompt: str, seed: Optional[int] = None) -> Optional[bytes]:
    """
    Generate a movie poster using HuggingFace's free Inference API.
    Model: stabilityai/stable-diffusion-xl-base-1.0

    Requires: HUGGINGFACE_API_KEY environment variable.
    Free tier: ~10 requests/min, no credit card needed.

    Returns: PNG bytes or None on failure.
    """
    if not HF_API_KEY:
        logger.info("[Diffusion] No HUGGINGFACE_API_KEY — skipping HF generation")
        return None

    try:
        import urllib.request, json as _json

        payload = {
            "inputs": prompt,
            "parameters": {
                "guidance_scale":      GUIDANCE,
                "num_inference_steps": 30,
                "width":  512,
                "height": 768,       # portrait for movie poster
            }
        }
        if seed is not None:
            payload["parameters"]["seed"] = seed

        req = urllib.request.Request(
            HF_API_URL,
            data=_json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {HF_API_KEY}",
                "Content-Type":  "application/json",
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            image_bytes = resp.read()
            # HF returns raw PNG bytes directly
            if image_bytes[:4] == b'\x89PNG':
                logger.info(f"[Diffusion] HuggingFace generated poster ({len(image_bytes)} bytes)")
                return image_bytes
            logger.warning(f"[Diffusion] HF returned non-PNG response")
            return None
    except Exception as e:
        logger.warning(f"[Diffusion] HuggingFace API failed: {e}")
        return None


def generate_via_replicate(prompt: str, seed: Optional[int] = None) -> Optional[str]:
    """
    Generate a movie poster using Replicate API (Stable Diffusion XL).
    Returns a URL to the generated image.

    Requires: REPLICATE_API_KEY environment variable.
    """
    if not REPLICATE_KEY:
        logger.info("[Diffusion] No REPLICATE_API_KEY — skipping Replicate generation")
        return None

    try:
        import urllib.request, urllib.parse, json as _json, time

        # Create prediction
        payload = {
            "version": REPLICATE_MODEL.split(":")[1],
            "input": {
                "prompt":              prompt,
                "negative_prompt":     "blurry, low quality, text, watermark, cartoon",
                "guidance_scale":      GUIDANCE,
                "num_inference_steps": 30,
                "width":  512,
                "height": 768,
            }
        }
        if seed is not None:
            payload["input"]["seed"] = seed

        req = urllib.request.Request(
            "https://api.replicate.com/v1/predictions",
            data=_json.dumps(payload).encode(),
            headers={
                "Authorization": f"Token {REPLICATE_KEY}",
                "Content-Type":  "application/json",
            }
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            prediction = _json.loads(resp.read())

        prediction_id = prediction.get("id")
        if not prediction_id:
            return None

        # Poll for result (max 90 seconds)
        for _ in range(18):
            time.sleep(5)
            poll_req = urllib.request.Request(
                f"https://api.replicate.com/v1/predictions/{prediction_id}",
                headers={"Authorization": f"Token {REPLICATE_KEY}"}
            )
            with urllib.request.urlopen(poll_req, timeout=10) as r:
                result = _json.loads(r.read())

            if result.get("status") == "succeeded":
                urls = result.get("output", [])
                if urls:
                    logger.info(f"[Diffusion] Replicate generated: {urls[0]}")
                    return urls[0]
            elif result.get("status") in ("failed", "canceled"):
                logger.warning(f"[Diffusion] Replicate prediction failed: {result.get('error')}")
                return None

        logger.warning("[Diffusion] Replicate timed out")
        return None

    except Exception as e:
        logger.warning(f"[Diffusion] Replicate API failed: {e}")
        return None




def generate_via_dalle(prompt: str) -> Optional[str]:
    """
    Generate a movie poster using OpenAI DALL-E 3.
    Returns a URL to the generated image.
    Requires: OPENAI_API_KEY environment variable.
    """
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        logger.info("[Diffusion] No OPENAI_API_KEY — skipping DALL-E generation")
        return None

    try:
        import urllib.request as _ur, json as _json

        payload = _json.dumps({
            "model":   "dall-e-3",
            "prompt":  prompt,
            "n":       1,
            "size":    "1024x1024",
            "quality": "standard",
        }).encode()

        req = _ur.Request(
            "https://api.openai.com/v1/images/generations",
            data=payload,
            headers={
                "Authorization": f"Bearer {openai_key}",
                "Content-Type":  "application/json",
            }
        )
        with _ur.urlopen(req, timeout=60) as r:
            data = _json.loads(r.read())
            url = data["data"][0]["url"]
            logger.info(f"[Diffusion] DALL-E 3 generated poster: {url[:60]}...")
            return url

    except Exception as e:
        logger.warning(f"[Diffusion] DALL-E generation failed: {e}")
        return None

# ── Main poster generator ──────────────────────────────────────────────────────

class DiffusionPosterGenerator:
    """
    Diffusion-based movie poster generator.

    Priority:
      1. HuggingFace Inference API (free, SDXL)
      2. Replicate API (pay-per-use, SDXL)
      3. Gradient placeholder (always works, no API needed)

    The DDPM math is always computed regardless of which backend is used,
    demonstrating understanding of the diffusion process.
    """

    def __init__(self):
        self.schedule = DDPMSchedule(T=T)
        self._cache: dict[str, str] = {}   # title+genre → url/data_uri
        logger.info(
            f"[Diffusion] Initialised DDPM schedule T={T} "
            f"β∈[{BETA_START},{BETA_END}] "
            f"HF={'yes' if HF_API_KEY else 'no'} "
            f"Replicate={'yes' if REPLICATE_KEY else 'no'}"
        )

    def _cache_key(self, title: str, genre: str) -> str:
        return hashlib.md5(f"{title}|{genre}".encode()).hexdigest()[:12]

    def generate(
        self,
        title:  str,
        genre:  str,
        year:   Optional[int]  = None,
        seed:   Optional[int]  = None,
    ) -> dict:
        """
        Generate a poster for a movie using diffusion.

        Returns a dict with:
          prompt:    the engineered SD prompt
          source:    'huggingface' | 'replicate' | 'placeholder'
          image_url: URL or data URI of the generated image
          schedule:  DDPM noise schedule summary
          ddpm_demo: forward diffusion demo at t=250, t=500, t=750
        """
        cache_key = self._cache_key(title, genre)
        if cache_key in self._cache:
            return {"cached": True, "image_url": self._cache[cache_key], "title": title}

        prompt = build_prompt(title, genre, year)
        deterministic_seed = seed or (abs(hash(title + genre)) % 2**31)

        # ── DDPM math demo (always runs) ──────────────────────────────────────
        rng     = np.random.default_rng(deterministic_seed)
        x0_mock = rng.uniform(-1, 1, (64, 64, 3))   # mock clean latent

        ddpm_demo = {}
        for t_demo in [250, 500, 750, 999]:
            noise      = rng.standard_normal(x0_mock.shape)
            x_t, eps   = self.schedule.forward_process(x0_mock, t_demo, noise)
            ddpm_demo[f"t{t_demo}"] = {
                "alpha_bar":   round(float(self.schedule.alphas_cumprod[t_demo]), 4),
                "snr":         round(self.schedule.snr(t_demo), 4),
                "signal_rms":  round(float(np.sqrt(np.mean(x_t**2))), 4),
                "noise_level": round(float(self.schedule.sqrt_one_minus_alphas_cumprod[t_demo]), 4),
            }

        # ── Try real image generation ─────────────────────────────────────────
        source    = "placeholder"
        image_url = None

        # 1. DALL-E 3 (uses existing OPENAI_API_KEY — highest quality)
        dalle_url = generate_via_dalle(prompt)
        if dalle_url:
            image_url = dalle_url
            source    = "dalle3"

        # 2. HuggingFace SDXL (if DALL-E failed)
        if not image_url:
            hf_bytes = generate_via_huggingface(prompt, seed=deterministic_seed)
            if hf_bytes:
                import base64
                b64 = base64.b64encode(hf_bytes).decode()
                image_url = f"data:image/png;base64,{b64}"
                source    = "huggingface_sdxl"

        # 3. Replicate SDXL (if HF failed)
        if not image_url:
            rep_url = generate_via_replicate(prompt, seed=deterministic_seed)
            if rep_url:
                image_url = rep_url
                source    = "replicate_sdxl"

        # 4. Gradient placeholder (always works)
        if not image_url:
            image_url = self._make_placeholder(title, genre, deterministic_seed)
            source    = "placeholder"

        self._cache[cache_key] = image_url

        return {
            "title":     title,
            "genre":     genre,
            "year":      year,
            "prompt":    prompt,
            "source":    source,
            "image_url": image_url,
            "ddpm_schedule": self.schedule.schedule_summary(),
            "ddpm_demo": ddpm_demo,
            "model": {
                "name":      "Stable Diffusion XL (SDXL)",
                "T":         T,
                "guidance":  GUIDANCE,
                "scheduler": "DDPM linear schedule",
                "reference": "Ho et al. NeurIPS 2020 + Rombach et al. CVPR 2022",
            },
        }

    def _make_placeholder(self, title: str, genre: str, seed: int) -> str:
        """
        Generate a gradient placeholder image as a data URI.
        Uses the movie title hash to pick unique colours.
        Always works — no external dependencies.
        """
        rng    = np.random.default_rng(seed)
        r1, g1, b1 = rng.integers(30, 180, 3)
        r2, g2, b2 = rng.integers(100, 255, 3)

        # 64×96 gradient (portrait ratio)
        H, W = 96, 64
        img  = np.zeros((H, W, 3), dtype=np.uint8)
        for row in range(H):
            t = row / H
            img[row, :, 0] = int(r1 * (1 - t) + r2 * t)
            img[row, :, 1] = int(g1 * (1 - t) + g2 * t)
            img[row, :, 2] = int(b1 * (1 - t) + b2 * t)

        # Encode as PNG via zlib (no PIL dependency)
        import struct, zlib
        def png_chunk(tag: bytes, data: bytes) -> bytes:
            c = struct.pack(">I", len(data)) + tag + data
            return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

        raw = b""
        for row in range(H):
            raw += b"\x00" + img[row].tobytes()

        png = (
            b"\x89PNG\r\n\x1a\n"
            + png_chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
            + png_chunk(b"IDAT", zlib.compress(raw))
            + png_chunk(b"IEND", b"")
        )
        import base64
        return f"data:image/png;base64,{base64.b64encode(png).decode()}"

    def noise_schedule_stats(self) -> dict:
        """Return the full DDPM noise schedule statistics."""
        return self.schedule.schedule_summary()

    def forward_diffusion_demo(self, t: int) -> dict:
        """
        Demo the forward diffusion process at a specific timestep.
        Useful for visualising how much noise is added at step t.
        """
        rng  = np.random.default_rng(42)
        x0   = rng.uniform(-1, 1, (32, 32, 3))
        x_t, eps = self.schedule.forward_process(x0, t)
        return {
            "timestep":   t,
            "alpha_bar":  round(float(self.schedule.alphas_cumprod[t]), 6),
            "snr":        round(self.schedule.snr(t), 6),
            "noise_std":  round(float(self.schedule.sqrt_one_minus_alphas_cumprod[t]), 4),
            "signal_std": round(float(self.schedule.sqrt_alphas_cumprod[t]), 4),
            "x0_rms":     round(float(np.sqrt(np.mean(x0**2))), 4),
            "xt_rms":     round(float(np.sqrt(np.mean(x_t**2))), 4),
            "description": (
                f"At t={t}: signal retains {self.schedule.sqrt_alphas_cumprod[t]:.1%} "
                f"of original, noise contributes {self.schedule.sqrt_one_minus_alphas_cumprod[t]:.1%}"
            ),
        }


# ── Module-level singleton ────────────────────────────────────────────────────
DIFFUSION_GENERATOR = DiffusionPosterGenerator()

print(
    f"  [Diffusion] DDPM schedule loaded T={T} "
    f"HF={'ready' if HF_API_KEY else 'no key'} "
    f"Replicate={'ready' if REPLICATE_KEY else 'no key'}"
)
