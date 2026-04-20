'use client'
import { useState } from 'react'

const API = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'
const C = {
  red:'#e50914', green:'#46d369', blue:'#4f8ef7', purple:'#a78bfa',
  amber:'#f59e0b', bg:'#0a0a0a', card:'rgba(255,255,255,0.03)',
  border:'rgba(255,255,255,0.08)', muted:'#666'
}

const GENRES = ['Action','Comedy','Drama','Horror','Sci-Fi','Romance','Thriller','Documentary','Animation','Fantasy']

export default function DiffusionDemo() {
  const [title, setTitle]   = useState('Inception')
  const [genre, setGenre]   = useState('Sci-Fi')
  const [year, setYear]     = useState('2010')
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [schedule, setSchedule] = useState<any>(null)

  async function generate() {
    setLoading(true)
    setResult(null)
    try {
      const r = await fetch(`${API}/diffusion/generate?title=${encodeURIComponent(title)}&genre=${genre}&year=${year}`, { method:'POST' })
      setResult(await r.json())
    } catch(e) { console.error(e) }
    setLoading(false)
  }

  async function loadSchedule() {
    const r = await fetch(`${API}/diffusion/schedule`)
    setSchedule(await r.json())
  }

  const ddpm = result?.ddpm_demo || {}

  return (
    <div style={{ background:C.bg, minHeight:'100vh', color:'#fff', fontFamily:'system-ui,sans-serif', padding:'32px 40px' }}>

      {/* Header */}
      <div style={{ marginBottom:32 }}>
        <a href="/" style={{ color:C.muted, fontSize:12, textDecoration:'none' }}>← Back to CineWave</a>
        <div style={{ fontSize:11, color:C.muted, letterSpacing:3, textTransform:'uppercase' as const, marginTop:16, marginBottom:6 }}>CineWave · Generative AI</div>
        <h1 style={{ fontSize:32, fontWeight:900, letterSpacing:-1, marginBottom:8 }}>
          Diffusion Model — Poster Generation
        </h1>
        <p style={{ color:C.muted, fontSize:14, lineHeight:1.6, maxWidth:600 }}>
          DDPM noise schedule (Ho et al. NeurIPS 2020) + DALL-E 3 image generation.
          Forward process: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε. T=1000 timesteps, β∈[1e-4, 0.02].
        </p>
      </div>

      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:24 }}>

        {/* Generator */}
        <div>
          <div style={{ background:C.card, border:`1px solid ${C.border}`, borderRadius:10, padding:24, marginBottom:16 }}>
            <div style={{ fontSize:12, fontWeight:700, color:'#fff', textTransform:'uppercase' as const, letterSpacing:1.5, marginBottom:16 }}>
              🎬 Generate Movie Poster
            </div>

            <div style={{ marginBottom:12 }}>
              <label style={{ fontSize:11, color:C.muted, display:'block', marginBottom:4 }}>MOVIE TITLE</label>
              <input value={title} onChange={e=>setTitle(e.target.value)}
                style={{ width:'100%', background:'rgba(255,255,255,0.06)', border:`1px solid ${C.border}`,
                  borderRadius:6, padding:'8px 12px', color:'#fff', fontSize:14, outline:'none', boxSizing:'border-box' as const }}/>
            </div>

            <div style={{ marginBottom:12 }}>
              <label style={{ fontSize:11, color:C.muted, display:'block', marginBottom:4 }}>GENRE (LinUCB arm)</label>
              <select value={genre} onChange={e=>setGenre(e.target.value)}
                style={{ width:'100%', background:'#1a1a1a', border:`1px solid ${C.border}`,
                  borderRadius:6, padding:'8px 12px', color:'#fff', fontSize:14, outline:'none' }}>
                {GENRES.map(g => <option key={g} value={g}>{g}</option>)}
              </select>
            </div>

            <div style={{ marginBottom:20 }}>
              <label style={{ fontSize:11, color:C.muted, display:'block', marginBottom:4 }}>YEAR</label>
              <input value={year} onChange={e=>setYear(e.target.value)}
                style={{ width:'100%', background:'rgba(255,255,255,0.06)', border:`1px solid ${C.border}`,
                  borderRadius:6, padding:'8px 12px', color:'#fff', fontSize:14, outline:'none', boxSizing:'border-box' as const }}/>
            </div>

            <button onClick={generate} disabled={loading}
              style={{ width:'100%', background: loading ? 'rgba(229,9,20,0.3)' : C.red,
                border:'none', borderRadius:8, padding:'12px', color:'#fff',
                fontSize:14, fontWeight:700, cursor: loading ? 'not-allowed' : 'pointer' }}>
              {loading ? '⏳ Generating with DALL-E 3...' : '🎨 Generate Poster'}
            </button>
          </div>

          {/* Prompt */}
          {result?.prompt && (
            <div style={{ background:C.card, border:`1px solid ${C.border}`, borderRadius:10, padding:20, marginBottom:16 }}>
              <div style={{ fontSize:11, color:C.muted, marginBottom:8, textTransform:'uppercase' as const, letterSpacing:1 }}>Engineered Prompt</div>
              <div style={{ fontSize:12, color:'#9ca3af', lineHeight:1.6, fontFamily:'monospace' }}>{result.prompt}</div>
              <div style={{ marginTop:12, display:'flex', gap:8, flexWrap:'wrap' as const }}>
                <span style={{ background:`${C.green}18`, border:`1px solid ${C.green}44`, color:C.green, borderRadius:20, padding:'3px 10px', fontSize:11 }}>
                  Source: {result.source}
                </span>
                <span style={{ background:`${C.blue}18`, border:`1px solid ${C.blue}44`, color:C.blue, borderRadius:20, padding:'3px 10px', fontSize:11 }}>
                  guidance_scale: {result.model?.guidance}
                </span>
                <span style={{ background:`${C.purple}18`, border:`1px solid ${C.purple}44`, color:C.purple, borderRadius:20, padding:'3px 10px', fontSize:11 }}>
                  T={result.model?.T}
                </span>
              </div>
            </div>
          )}

          {/* DDPM Demo */}
          {result?.ddpm_demo && (
            <div style={{ background:C.card, border:`1px solid ${C.border}`, borderRadius:10, padding:20 }}>
              <div style={{ fontSize:12, fontWeight:700, color:'#fff', textTransform:'uppercase' as const, letterSpacing:1.5, marginBottom:16 }}>
                📊 DDPM Forward Process
              </div>
              <div style={{ fontSize:11, color:C.muted, marginBottom:12 }}>
                x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε — signal degrades, noise increases
              </div>
              {Object.entries(ddpm).map(([key, val]: any) => (
                <div key={key} style={{ marginBottom:14 }}>
                  <div style={{ display:'flex', justifyContent:'space-between', marginBottom:4 }}>
                    <span style={{ fontSize:12, color:'#9ca3af', fontFamily:'monospace' }}>{key}</span>
                    <span style={{ fontSize:11, color:C.muted }}>SNR={val.snr?.toFixed(4)} | signal={((1-val.noise_level)*100).toFixed(1)}%</span>
                  </div>
                  {/* Signal bar */}
                  <div style={{ height:6, background:'rgba(255,255,255,0.06)', borderRadius:3, marginBottom:2 }}>
                    <div style={{ height:'100%', borderRadius:3, background:C.green,
                      width:`${(1-val.noise_level)*100}%`, transition:'width 0.5s' }}/>
                  </div>
                  {/* Noise bar */}
                  <div style={{ height:6, background:'rgba(255,255,255,0.06)', borderRadius:3 }}>
                    <div style={{ height:'100%', borderRadius:3, background:C.red,
                      width:`${val.noise_level*100}%`, transition:'width 0.5s' }}/>
                  </div>
                  <div style={{ display:'flex', justifyContent:'space-between', marginTop:2 }}>
                    <span style={{ fontSize:9, color:C.green }}>signal (green)</span>
                    <span style={{ fontSize:9, color:C.red }}>noise (red)</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right: Generated Image + Schedule */}
        <div>
          {/* Generated Poster */}
          <div style={{ background:C.card, border:`1px solid ${C.border}`, borderRadius:10, padding:24, marginBottom:16 }}>
            <div style={{ fontSize:12, fontWeight:700, color:'#fff', textTransform:'uppercase' as const, letterSpacing:1.5, marginBottom:16 }}>
              🖼️ Generated Poster
            </div>
            {loading && (
              <div style={{ height:400, display:'flex', alignItems:'center', justifyContent:'center',
                background:'rgba(229,9,20,0.05)', borderRadius:8, border:`1px solid ${C.red}22` }}>
                <div style={{ textAlign:'center' as const }}>
                  <div style={{ fontSize:32, marginBottom:12 }}>🎨</div>
                  <div style={{ color:C.muted, fontSize:13 }}>DALL-E 3 generating...</div>
                  <div style={{ color:"#444", fontSize:11, marginTop:4 }}>Running DDPM schedule T=1000</div>
                </div>
              </div>
            )}
            {result?.image_url && !loading && (
              <div>
                <img src={result.image_url} alt={title}
                  style={{ width:'100%', borderRadius:8, display:'block',
                    maxHeight:500, objectFit:'cover' as const }}
                  onError={(e:any) => e.target.style.display='none'}/>
                <div style={{ marginTop:12, fontSize:12, color:C.muted }}>
                  {result.title} ({result.year}) · {result.genre}
                </div>
              </div>
            )}
            {!result && !loading && (
              <div style={{ height:300, display:'flex', alignItems:'center', justifyContent:'center',
                background:'rgba(255,255,255,0.02)', borderRadius:8, border:`1px dashed ${C.border}` }}>
                <div style={{ textAlign:'center' as const, color:C.muted }}>
                  <div style={{ fontSize:40, marginBottom:8 }}>🎬</div>
                  <div>Enter a movie title and click Generate</div>
                </div>
              </div>
            )}
          </div>

          {/* DDPM Schedule */}
          <div style={{ background:C.card, border:`1px solid ${C.border}`, borderRadius:10, padding:24 }}>
            <div style={{ fontSize:12, fontWeight:700, color:'#fff', textTransform:'uppercase' as const, letterSpacing:1.5, marginBottom:16 }}>
              📐 DDPM Noise Schedule
            </div>
            {!schedule ? (
              <div>
                <div style={{ fontSize:12, color:'#9ca3af', marginBottom:16, lineHeight:1.6 }}>
                  Linear β schedule from β₁=1e-4 to β_T=0.02.<br/>
                  ᾱ_t = ∏α_s, SNR(t) = ᾱ_t/(1-ᾱ_t)
                </div>
                <button onClick={loadSchedule}
                  style={{ background:`${C.blue}22`, border:`1px solid ${C.blue}44`,
                    color:C.blue, borderRadius:6, padding:'8px 16px', cursor:'pointer', fontSize:12 }}>
                  Load Schedule Stats from /diffusion/schedule
                </button>
              </div>
            ) : (
              <div>
                {[
                  { label:'Timesteps T', value: schedule.T, color:C.blue },
                  { label:'β start', value: schedule.beta_start, color:C.green },
                  { label:'β end', value: schedule.beta_end, color:C.amber },
                  { label:'ᾱ at t=500', value: schedule.alpha_bar_T500, color:C.purple },
                  { label:'ᾱ at t=999', value: schedule.alpha_bar_T999, color:C.red },
                  { label:'SNR at t=100', value: schedule.snr_t100, color:C.green },
                  { label:'SNR at t=500', value: schedule.snr_t500, color:C.amber },
                  { label:'SNR at t=999', value: schedule.snr_t999, color:C.red },
                ].map(s => (
                  <div key={s.label} style={{ display:'flex', justifyContent:'space-between',
                    padding:'8px 0', borderBottom:`1px solid ${C.border}` }}>
                    <span style={{ fontSize:12, color:'#9ca3af' }}>{s.label}</span>
                    <span style={{ fontSize:13, fontWeight:700, color:s.color, fontFamily:'monospace' }}>{s.value}</span>
                  </div>
                ))}
                <div style={{ marginTop:12, fontSize:11, color:C.muted }}>
                  Reference: Ho et al. "Denoising Diffusion Probabilistic Models" NeurIPS 2020
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
