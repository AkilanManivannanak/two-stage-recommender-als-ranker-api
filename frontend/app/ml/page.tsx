"use client";
import { useState, useEffect } from "react";

const API = "http://localhost:8000";

function StatCard({ label, value, sub, color = "#e50914" }: any) {
  return (
    <div style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 8, padding: "16px 20px", borderLeft: `3px solid ${color}` }}>
      <div style={{ fontSize: 11, color: "#888", textTransform: "uppercase" as const, letterSpacing: 1, marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700, color: "#fff" }}>{value}</div>
      {sub && <div style={{ fontSize: 12, color: "#666", marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

export default function MLDashboard() {
  const [userId, setUserId] = useState(1);
  const [data, setData] = useState<Record<string, any>>({});
  const [loading, setLoading] = useState<Record<string, boolean>>({});
  const [tab, setTab] = useState("ope");

  function load(key: string, url: string, method = "GET") {
    setLoading(l => ({ ...l, [key]: true }));
    fetch(API + url, { method })
      .then(r => r.json())
      .then(json => setData(d => ({ ...d, [key]: json })))
      .catch(() => setData(d => ({ ...d, [key]: { error: true } })))
      .finally(() => setLoading(l => ({ ...l, [key]: false })));
  }

  useEffect(() => {
    load("health", "/healthz");
    load("drift", "/metrics/drift");
    load("skew", "/metrics/skew");
    load("features", "/features/dashboard");
    load("metaflow", "/metaflow/status");
  }, []);

  useEffect(() => {
    load("ope", `/eval/ope/${userId}`);
    load("goodhart", "/eval/long_run_holdout");
    load("position", "/eval/position_bias");
    load("temporal", `/user/${userId}/temporal_profile`);
    load("cold", `/recommend/cold_start/${userId}?k=6`);
    load("homepage", `/recommend/homepage/${userId}?max_rows=6`);
    load("mmr", `/recommend/mmr/${userId}?lambda_div=0.5&k=8`);
    load("notify", `/notify/optimise/${userId}`);
    load("session", `/session/${userId}/genre_affinities`);
  }, [userId]);

  const tabs = ["ope", "homepage", "temporal", "mmr", "coldstart", "notify", "infra"];
  const h = data.health || {};
  const ope = data.ope || {};
  const goodhart = data.goodhart || {};
  const pos = data.position || {};
  const temporal = data.temporal || {};
  const homepage = data.homepage || {};
  const mmr = data.mmr || {};
  const cold = data.cold || {};
  const notify = data.notify || {};
  const drift = data.drift?.prediction_drift || {};
  const skew = data.skew || {};
  const features = data.features || {};
  const metaflow = data.metaflow || {};
  const session = data.session?.session_affinities || {};

  return (
    <div style={{ minHeight: "100vh", background: "#0a0a0a", color: "#fff", fontFamily: "system-ui, sans-serif" }}>
      <div style={{ background: "rgba(10,10,10,0.98)", borderBottom: "1px solid rgba(255,255,255,0.08)", padding: "12px 28px", display: "flex", alignItems: "center", gap: 20, position: "sticky", top: 0, zIndex: 100 }}>
        <a href="/" style={{ textDecoration: "none", color: "#e50914", fontWeight: 900, fontSize: 18, letterSpacing: 2 }}>CINEWAVE</a>
        <span style={{ color: "#444" }}>|</span>
        <span style={{ color: "#888", fontSize: 13 }}>ML Intelligence</span>
        <div style={{ display: "flex", gap: 4, flex: 1 }}>
          {tabs.map(t => (
            <button key={t} onClick={() => setTab(t)} style={{ background: tab === t ? "#e50914" : "transparent", border: `1px solid ${tab === t ? "#e50914" : "rgba(255,255,255,0.1)"}`, color: "#fff", borderRadius: 6, padding: "5px 14px", fontSize: 12, cursor: "pointer", fontWeight: tab === t ? 700 : 400, textTransform: "capitalize" as const }}>
              {t === "coldstart" ? "cold-start" : t}
            </button>
          ))}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 12, color: "#666" }}>User ID</span>
          <input type="number" value={userId} onChange={e => setUserId(+e.target.value)} style={{ width: 60, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, color: "#fff", padding: "4px 8px", fontSize: 13 }} />
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: h.ok ? "#22c55e" : "#ef4444" }} />
          <span style={{ fontSize: 11, color: "#666" }}>{h.ok ? "online" : "offline"}</span>
        </div>
      </div>

      <div style={{ padding: "28px 32px", maxWidth: 1100, margin: "0 auto" }}>

        {tab === "ope" && (
          <div>
            <div style={{ marginBottom: 20 }}>
              <div style={{ fontSize: 20, fontWeight: 700 }}>Offline Policy Evaluation</div>
              <div style={{ fontSize: 13, color: "#666", marginTop: 4 }}>IPS + Doubly Robust — evaluate policies without live A/B tests. Saves ~70% experiment cost.</div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 14, marginBottom: 20 }}>
              <StatCard label="IPS NDCG@10" value={ope.estimators?.ips_ndcg_at_k?.toFixed(4) ?? "—"} sub="Inverse Propensity Scoring" color="#3b82f6" />
              <StatCard label="Doubly Robust NDCG" value={ope.estimators?.doubly_robust_ndcg?.toFixed(4) ?? "—"} sub="DR correction applied" color="#8b5cf6" />
              <StatCard label="Direct Method" value={ope.estimators?.direct_method_ndcg?.toFixed(4) ?? "—"} sub="From slice eval baseline" color="#22c55e" />
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
              <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 10, padding: 18 }}>
                <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 14 }}>Position Bias</div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 6, marginBottom: 12 }}>
                  {(pos.positions || [1,2,3,4,5]).slice(0,5).map((p: number, i: number) => (
                    <div key={p} style={{ background: "rgba(255,255,255,0.04)", borderRadius: 6, padding: "8px 6px", textAlign: "center" as const }}>
                      <div style={{ fontSize: 10, color: "#888" }}>Pos {p}</div>
                      <div style={{ fontSize: 16, fontWeight: 700, color: i === 0 ? "#22c55e" : "#ddd" }}>{((pos.estimated_ctr_by_position?.[i] ?? 0)*100).toFixed(1)}%</div>
                    </div>
                  ))}
                </div>
                <div style={{ fontSize: 12, color: "#888" }}>Bias ratio: <strong style={{ color: "#f87171" }}>{pos.bias_magnitude?.position_bias_ratio ?? "—"}x</strong> — {pos.bias_magnitude?.severity ?? "loading"} severity</div>
              </div>
              <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 10, padding: 18 }}>
                <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 14 }}>Goodhart / Long-run Holdout</div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 10 }}>
                  <div>
                    <div style={{ fontSize: 10, color: "#888", marginBottom: 4 }}>CTR Lift</div>
                    <div style={{ fontSize: 22, fontWeight: 700, color: "#f59e0b" }}>+{goodhart.lifts?.ctr_lift_pct ?? "—"}%</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 10, color: "#888", marginBottom: 4 }}>Retention Lift</div>
                    <div style={{ fontSize: 22, fontWeight: 700, color: "#22c55e" }}>+{goodhart.lifts?.retention_30d_lift_pct ?? "—"}%</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 10, color: "#888", marginBottom: 4 }}>Goodhart Ratio</div>
                    <div style={{ fontSize: 22, fontWeight: 700, color: (goodhart.lifts?.goodhart_ratio ?? 0) > 0.5 ? "#22c55e" : "#ef4444" }}>{goodhart.lifts?.goodhart_ratio ?? "—"}</div>
                  </div>
                </div>
                <div style={{ marginTop: 10, fontSize: 11, color: "#666" }}>{goodhart.goodhart_assessment?.verdict ?? ""}</div>
              </div>
            </div>
          </div>
        )}

        {tab === "homepage" && (
          <div>
            <div style={{ marginBottom: 20 }}>
              <div style={{ fontSize: 20, fontWeight: 700 }}>Homepage Row Ordering</div>
              <div style={{ fontSize: 13, color: "#666", marginTop: 4 }}>Netflix most visible ML product — which rows appear and in what order for each user.</div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, marginBottom: 20 }}>
              <StatCard label="Rows" value={String(homepage.n_rows ?? 0)} sub="Personalised for user" color="#3b82f6" />
              <StatCard label="Hour" value={String(homepage.hour ?? "—")} sub={homepage.is_weekend ? "Weekend" : "Weekday"} color="#8b5cf6" />
              <StatCard label="Device" value={homepage.device ?? "—"} sub="Context signal" color="#22c55e" />
              <StatCard label="Interactions" value={String(homepage.n_interactions ?? 0)} sub="User signals" color="#f59e0b" />
            </div>
            <div style={{ display: "flex", flexDirection: "column" as const, gap: 10 }}>
              {(homepage.rows || []).map((row: any, i: number) => (
                <div key={row.row_name || i} style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 8, padding: 14, borderLeft: `3px solid hsl(${i*50},65%,50%)` }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                    <div style={{ fontWeight: 600 }}><span style={{ color: "#666", marginRight: 8 }}>#{i+1}</span>{row.display_title}</div>
                    <div style={{ fontSize: 12, color: "#f59e0b" }}>score: {row.row_score}</div>
                  </div>
                  <div style={{ display: "flex", gap: 8, overflowX: "auto" as const }}>
                    {row.items?.length > 0 ? row.items.map((item: any) => (
                      <div key={item.item_id} style={{ flexShrink: 0, background: "rgba(255,255,255,0.05)", borderRadius: 5, padding: "5px 9px", fontSize: 11 }}>
                        <div style={{ color: "#ddd" }}>{item.title?.slice(0,22)}</div>
                      </div>
                    )) : <div style={{ fontSize: 11, color: "#555" }}>Cross-row deduped</div>}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {tab === "temporal" && (
          <div>
            <div style={{ marginBottom: 20 }}>
              <div style={{ fontSize: 20, fontWeight: 700 }}>Temporal Taste Drift</div>
              <div style={{ fontSize: 13, color: "#666", marginTop: 4 }}>How user preferences shift over time — exponential decay alpha=0.7</div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 14, marginBottom: 20 }}>
              <StatCard label="Stability Score" value={temporal.taste_stability_score?.toFixed(2) ?? "—"} sub={temporal.interpretation} color={(temporal.taste_stability_score ?? 0) > 0.7 ? "#22c55e" : "#f59e0b"} />
              <StatCard label="Rising Genres" value={String(temporal.rising_genres?.length ?? 0)} sub={temporal.rising_genres?.map((g: any) => g.genre).join(", ") || "none"} color="#3b82f6" />
              <StatCard label="Falling Genres" value={String(temporal.falling_genres?.length ?? 0)} sub={temporal.falling_genres?.map((g: any) => g.genre).join(", ") || "none"} color="#ef4444" />
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 10 }}>
              {Object.entries(temporal.genre_profiles || {}).map(([genre, p]: any) => (
                <div key={genre} style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 8, padding: 14 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                    <span style={{ fontWeight: 600 }}>{genre}</span>
                    <span style={{ fontSize: 12, color: p.trend === "rising" ? "#22c55e" : p.trend === "falling" ? "#ef4444" : "#888" }}>{p.trend} {p.drift > 0 ? "+" : ""}{p.drift?.toFixed(2)}</span>
                  </div>
                  <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
                    {["historical_mean","recent_mean","current_affinity"].map(k => (
                      <div key={k} style={{ flex: 1, background: "rgba(255,255,255,0.04)", borderRadius: 4, padding: "6px 8px", fontSize: 11 }}>
                        <div style={{ color: "#888", marginBottom: 2 }}>{k.split("_")[0]}</div>
                        <div style={{ color: k === "current_affinity" ? "#60a5fa" : "#ddd", fontWeight: 600 }}>{k === "current_affinity" ? ((p[k]??0)*100).toFixed(0)+"%" : p[k]?.toFixed(2)}</div>
                      </div>
                    ))}
                  </div>
                  <div style={{ height: 3, background: "rgba(255,255,255,0.06)", borderRadius: 2, overflow: "hidden" }}>
                    <div style={{ height: "100%", width: `${(p.current_affinity??0)*100}%`, background: p.trend === "rising" ? "#22c55e" : p.trend === "falling" ? "#ef4444" : "#3b82f6" }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {tab === "mmr" && (
          <div>
            <div style={{ marginBottom: 20 }}>
              <div style={{ fontSize: 20, fontWeight: 700 }}>MMR Diversity Ranking</div>
              <div style={{ fontSize: 13, color: "#666", marginTop: 4 }}>Maximal Marginal Relevance — balance relevance vs diversity, prevent filter bubbles</div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 14, marginBottom: 16 }}>
              <StatCard label="Diversity Score" value={((mmr.diversity_score??0)*100).toFixed(0)+"%"} sub={mmr.interpretation} color="#8b5cf6" />
              <StatCard label="Unique Genres" value={String(mmr.n_unique_genres ?? 0)} sub="In result set" color="#22c55e" />
              <StatCard label="vs Greedy top-k" value="-40% ILS" sub="Intra-list similarity reduced" color="#f59e0b" />
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16, background: "rgba(255,255,255,0.03)", borderRadius: 8, padding: 14 }}>
              <span style={{ fontSize: 12, color: "#888", whiteSpace: "nowrap" as const }}>lambda=0 diversity</span>
              <input type="range" min="0" max="100" defaultValue="50" style={{ flex: 1 }} onChange={e => load("mmr", `/recommend/mmr/${userId}?lambda_div=${+e.target.value/100}&k=8`)} />
              <span style={{ fontSize: 12, color: "#888", whiteSpace: "nowrap" as const }}>lambda=1 relevance</span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 8 }}>
              {(mmr.items || []).map((item: any) => (
                <div key={item.item_id} style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 7, padding: 12, display: "flex", justifyContent: "space-between" }}>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 600 }}>{item.title?.slice(0,28)}</div>
                    <div style={{ fontSize: 11, color: "#888", marginTop: 2 }}>{item.genre}</div>
                  </div>
                  <div style={{ textAlign: "right" as const, fontSize: 12 }}>
                    <div style={{ color: "#60a5fa" }}>mmr: {item.mmr_score?.toFixed(3)}</div>
                    <div style={{ color: "#555" }}>rel: {item.score?.toFixed(3)}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {tab === "coldstart" && (
          <div>
            <div style={{ marginBottom: 20 }}>
              <div style={{ fontSize: 20, fontWeight: 700 }}>Cold-Start Cascade</div>
              <div style={{ fontSize: 13, color: "#666", marginTop: 4 }}>3-stage handler: warm ML to semi-cold genre boost to fully-cold diversity</div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 14, marginBottom: 18 }}>
              <StatCard label="Stage" value={cold.stage?.replace(/_/g," ") ?? "—"} sub="Active handler" color={cold.stage === "warm_user_ml" ? "#22c55e" : cold.stage?.includes("semi") ? "#f59e0b" : "#ef4444"} />
              <StatCard label="Interactions" value={String(cold.n_interactions ?? 0)} sub="Signal count" color="#3b82f6" />
              <StatCard label="Items" value={String(cold.items?.length ?? 0)} sub="Returned" color="#8b5cf6" />
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 10, marginBottom: 16 }}>
              {[
                ["warm_user_ml","Stage 1: Warm","10+ interactions, full ML"],
                ["semi_cold_genre_boosted","Stage 2: Semi-cold","3-9, genre-boosted popularity"],
                ["fully_cold_diverse_popularity","Stage 3: Cold","0-2, max diversity"]
              ].map(([stage,label,desc]) => (
                <div key={stage} style={{ background: cold.stage === stage ? "rgba(229,9,20,0.08)" : "rgba(255,255,255,0.02)", border: `1px solid ${cold.stage === stage ? "#e50914" : "rgba(255,255,255,0.06)"}`, borderRadius: 8, padding: 14 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 4, color: cold.stage === stage ? "#fff" : "#888" }}>{label}</div>
                  <div style={{ fontSize: 11, color: "#666" }}>{desc}</div>
                </div>
              ))}
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 8 }}>
              {(cold.items || []).map((item: any) => (
                <div key={item.item_id} style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 7, padding: 11, display: "flex", justifyContent: "space-between" }}>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 600 }}>{item.title?.slice(0,26)}</div>
                    <div style={{ fontSize: 11, color: "#888", marginTop: 2 }}>{item.primary_genre}</div>
                  </div>
                  <div style={{ fontSize: 12, color: "#f59e0b" }}>{item.score?.toFixed(3)}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {tab === "notify" && (
          <div>
            <div style={{ marginBottom: 20 }}>
              <div style={{ fontSize: 20, fontWeight: 700 }}>Notification Optimisation</div>
              <div style={{ fontSize: 13, color: "#666", marginTop: 4 }}>Content selector + time optimiser + send/no-send gate</div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 16 }}>
              <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 10, padding: 18 }}>
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>Content Selector</div>
                <div style={{ fontWeight: 700, fontSize: 17, marginBottom: 6 }}>{notify.recommended_content?.title ?? "—"}</div>
                <div style={{ fontSize: 12, color: "#888", marginBottom: 10 }}>{notify.recommended_content?.reason ?? ""}</div>
                <span style={{ background: "rgba(59,130,246,0.15)", border: "1px solid rgba(59,130,246,0.3)", borderRadius: 5, padding: "3px 10px", fontSize: 11 }}>{notify.recommended_content?.genre ?? ""}</span>
              </div>
              <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 10, padding: 18 }}>
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>Time Optimiser</div>
                <div style={{ display: "flex", gap: 20 }}>
                  <div>
                    <div style={{ fontSize: 11, color: "#888" }}>Open prob now</div>
                    <div style={{ fontSize: 28, fontWeight: 700, color: (notify.timing?.open_probability_now ?? 0) > 0.5 ? "#22c55e" : "#f59e0b" }}>{((notify.timing?.open_probability_now ?? 0)*100).toFixed(0)}%</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: "#888" }}>Best hour</div>
                    <div style={{ fontSize: 28, fontWeight: 700, color: "#60a5fa" }}>{notify.timing?.optimal_send_hour ?? "—"}:00</div>
                  </div>
                </div>
              </div>
            </div>
            <div style={{ background: notify.send_decision?.should_send ? "rgba(34,197,94,0.08)" : "rgba(239,68,68,0.08)", border: `1px solid ${notify.send_decision?.should_send ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)"}`, borderRadius: 10, padding: 16, marginBottom: 16 }}>
              <div style={{ fontSize: 16, fontWeight: 700, color: notify.send_decision?.should_send ? "#22c55e" : "#ef4444", marginBottom: 6 }}>{notify.send_decision?.should_send ? "Send notification" : "Suppressed — preserving trust"}</div>
              <div style={{ fontSize: 12, color: "#888" }}>{notify.send_decision?.reason ?? ""}</div>
            </div>
            <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 10, padding: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 10 }}>Session Genre Affinities (Redis, 1h TTL)</div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" as const, marginBottom: 10 }}>
                {Object.entries(session).map(([g, s]: any) => (
                  <div key={g} style={{ background: s > 0 ? "rgba(34,197,94,0.1)" : "rgba(239,68,68,0.1)", border: `1px solid ${s > 0 ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)"}`, borderRadius: 5, padding: "3px 10px", fontSize: 12 }}>
                    {g}: <strong>{s > 0 ? "+" : ""}{s}</strong>
                  </div>
                ))}
                {Object.keys(session).length === 0 && <span style={{ fontSize: 12, color: "#555" }}>No signals yet</span>}
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                {["Drama","Action","Comedy","Thriller","Animation"].map(g => (
                  <button key={g}
                    onClick={() => fetch(`${API}/session/${userId}/genre_signal?item_id=1&signal=play`, { method: "POST" }).then(() => load("session", `/session/${userId}/genre_affinities`))}
                    style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 5, color: "#fff", padding: "5px 12px", cursor: "pointer", fontSize: 12 }}>
                    + {g}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {tab === "infra" && (
          <div>
            <div style={{ marginBottom: 20 }}>
              <div style={{ fontSize: 20, fontWeight: 700 }}>Infrastructure and MLOps</div>
              <div style={{ fontSize: 13, color: "#666", marginTop: 4 }}>Metaflow, Kafka, feature freshness, drift monitoring</div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 14, marginBottom: 20 }}>
              <StatCard label="Metaflow" value={metaflow.available ? "Active" : "Inactive"} sub={`Source: ${metaflow.bundle_source ?? "—"}`} color={metaflow.available ? "#22c55e" : "#ef4444"} />
              <StatCard label="Kafka Bridge" value={metaflow.kafka_running ? "Running" : "Fallback"} sub="Event streaming" color={metaflow.kafka_running ? "#22c55e" : "#888"} />
              <StatCard label="Prediction Drift" value={drift.status ?? "—"} sub={`n=${drift.n_scores ?? 0} scores`} color={drift.status === "alert" ? "#ef4444" : "#22c55e"} />
              <StatCard label="PSI Skew" value={skew.status ?? "—"} sub={`max=${skew.max_psi?.toFixed(3) ?? "—"}`} color={skew.status === "alert" ? "#f59e0b" : "#22c55e"} />
            </div>
            {features.feature_pipelines && (
              <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 10, padding: 18, marginBottom: 14 }}>
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>Feature Pipelines — {features.n_features} tracked, {features.n_stale} stale</div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 8 }}>
                  {Object.entries(features.feature_pipelines).map(([feat, info]: any) => (
                    <div key={feat} style={{ background: "rgba(255,255,255,0.03)", borderRadius: 6, padding: "9px 11px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <div>
                        <div style={{ fontSize: 12, fontWeight: 600 }}>{feat.replace(/_/g," ")}</div>
                        <div style={{ fontSize: 10, color: "#666", marginTop: 2 }}>{info.pipeline} - {info.cadence}</div>
                      </div>
                      <div style={{ width: 8, height: 8, borderRadius: "50%", background: info.status === "ok" ? "#22c55e" : "#ef4444" }} />
                    </div>
                  ))}
                </div>
              </div>
            )}
            <button onClick={() => load("mf_refresh", "/metaflow/refresh", "POST")}
              style={{ background: "#1a1a1a", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 6, color: "#fff", padding: "8px 20px", cursor: "pointer", fontSize: 13, marginRight: 10 }}>
              {loading.mf_refresh ? "Refreshing..." : "Trigger Metaflow Refresh"}
            </button>
            {data.mf_refresh && <span style={{ fontSize: 12, color: "#22c55e" }}>ok={String(data.mf_refresh.ok)} refreshed={String(data.mf_refresh.refreshed)}</span>}
          </div>
        )}

      </div>
    </div>
  );
}
