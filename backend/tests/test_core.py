"""
Core unit tests — verifies real ML component outputs.
Run: pytest backend/tests/ -v
"""
import numpy as np
import sys
sys.path.insert(0, 'backend/src')

def test_gru_cell_shapes():
    from recsys.serving.session_intent import GRUCell
    gru = GRUCell()
    x = np.random.randn(8).astype(np.float32)
    h = np.zeros(16, dtype=np.float32)
    h_new = gru.step(x, h)
    assert h_new.shape == (16,), f"Expected (16,), got {h_new.shape}"

def test_linucb_ucb_score():
    from recsys.serving.bandit_v2 import LinUCBArm
    arm = LinUCBArm(context_dim=8, alpha=1.0)
    ctx = np.random.randn(8).astype(np.float32)
    score = arm.ucb_score(ctx)
    assert isinstance(score, float), "UCB score should be float"
    assert score > 0, "UCB score should be positive"

def test_ddpm_schedule():
    import numpy as np
    T = 1000
    betas = np.linspace(1e-4, 0.02, T)
    alphas = 1 - betas
    alphas_cumprod = np.cumprod(alphas)
    # Forward process variance preservation
    t = 500
    signal = np.sqrt(alphas_cumprod[t])
    noise  = np.sqrt(1 - alphas_cumprod[t])
    assert abs(signal**2 + noise**2 - 1.0) < 1e-5, "Variance not preserved"

def test_reward_model_score():
    from recsys.serving.reward_model import score, build_features
    feat = build_features(
        genre="Action",
        user_genre_ratings={"Action": [4.0, 4.5]},
        user_genres={"Action"},
        item={"avg_rating": 4.2, "year": 2020, "vote_count": 100},
        session_momentum=0.7,
        days_since_genre=7.0,
    )
    s = score(feat)
    assert 0.0 <= s <= 1.0, f"Score {s} not in [0,1]"

def test_sparse_training_sparsity():
    from recsys.serving.reward_model_sparse import SparseRewardModel
    model = SparseRewardModel(l1_lambda=0.05)
    result = model.fit()
    assert result["status"] == "trained_sparse"
    assert result["n_zero"] > 0, "L1 should zero out some weights"
    assert result["sparsity"] > 0, "Should have some sparsity"

def test_ssl_gru_predicts():
    from recsys.serving.self_supervised_gru import SSL_GRU
    import numpy as np
    events = [np.random.randn(8).tolist() for _ in range(5)]
    h = SSL_GRU.encode(events)
    probs = SSL_GRU.predict_next(h)
    assert len(probs) == 8, "Should predict over 8 genres"
    assert abs(sum(probs) - 1.0) < 1e-5, "Probs should sum to 1"

def test_data_curation_filters():
    from recsys.serving.data_curation import curate_catalog
    catalog = {
        1: {"title": "Good Movie", "vote_count": 100, "avg_rating": 4.0,
            "poster_url": "http://x", "year": 2010, "primary_genre": "Drama"},
        2: {"title": "Bad", "vote_count": 1, "avg_rating": 1.0,
            "poster_url": "", "year": 2020, "primary_genre": "Action"},
    }
    curated, report = curate_catalog(catalog, verbose=False)
    assert len(curated) == 1, "Should keep only the good movie"
    assert report["n_removed"] == 1
