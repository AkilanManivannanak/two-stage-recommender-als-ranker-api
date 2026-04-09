"use client";
import { useState, useEffect } from "react";

const API = "http://localhost:8000";

function StatCard({ label, value, sub, color = "#e50914" }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div style={{
      background: "rgba(255,255,255,0.04)",
      border: "1px solid rgba(255,255,255,0.08)",
      borderRadius: 8,
      padding: "16px 20px",
      borderLeft: `3px solid ${color}`,
    }}>
      <div style={{ fontSize: 11, color: "#888", textTransform: "uppercase", letterSpacing: 1, marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 24, fontWeight: 700, color: "#fff" }}>{value}</div>
      {sub && <div style={{ fontSize: 12, color: "#666", marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function SectionHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div style={{ marginBottom: 20, borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: 12 }}>
      <div style={{ fontSize: 18, fontWeight: 700, color: "#fff" }}>{title}</div>
      <div style={{ fontSize: 12, color: "#666", marginTop: 4 }}>{subtitle}</div>
    </div>
  );
}

export default function MLDashboard() {
  const [userId, setUserId] = useState(1);
  const [data, setData] = useState<Record<string, any>>({});
  const [loading, setLoading] = useState<Record<string, boolean>>({});
  const [activeSection, setActiveSection] = useState("ope");

  async function fetch_data(key: string, url: string, method = "GET") {
    setLoading(l => ({ ...l, [key]: true }));
    try {
      const r = await fetch(API + url, { method });
      const j = await r.json();
      setData(d => ({ ...d, [key]: j }));
    } catch (e) {
      setData(d => ({ ...d, [key]: { error: String(e) } }));
    } finally {
      setLoading(l => ({ ...l, [key]: false }));
    }
  }

  useEffect(() => {
    fetch_data("health", "/healthz");
    fetch_data("drift", "/metrics/drift");
    fetch_data("skew", "/metrics/skew");
    fetch_data("features", "/features/dashboard");
    fetch_data("metaflow", "/metaflow/status");
  }, []);

  useEffect(() => {
    fetch_data("ope_user", `/eval/ope/${userId}`);
    fetch_data("temporal", `/user/${userId}/temporal_profile`);
    fetch_data("cold_start", `/recommend/cold_start/${userId}?k=6`);
    fetch_data("homepage", `/recommend/homepage/${userId}?max_rows=5`);
    fetch_data("mmr", `/recommend/mmr/${userId}?lambda_div=0.5&k=8`);
    fetch_data("notify", `/notify/optimise/${userId}`);
    fetch_data("session_aff", `/session/${userId}/genre_affinities`);
  }, [userId]);

  const sections = [
    { id: "ope", label: "OPE" },
    { id: "homepage", label: "Homepage" },
    { id: "temporal", label: "Taste Drift" },
    { id: "mmr", label: "MMR Diversity" },
    { id: "coldstart", label: "Cold-Start" },
    { id: "notify", label: "Notifications" },
    { id: "infra", label: "Infrastructure" },
  ];

  const h = data.health || {};
  const ope = data.ope_user || {};
  const temporal = data.temporal || {};
  const homepage = data.homepage || {};
  const mmr = data.mmr || {};
  const cold = data.cold_start || {};
  const notify = data.notify || {};
  const drift = data.drift || {};
  const skew = data.skew || {};
  const features = data.features || {};
  const metaflow = data.metaflow || {};
  const session_aff = data.session_aff || {};

  return (
    <div style={{ minHeight: "100vh", background: "#0a0a0a", color: "#fff", fontFamily: "'SF Pro Display', -apple-system, sans-serif" }}>
      {/* Header */}
      <div style={{ background: "rgba(10,10,10,0.95)", borderBottom: "1px solid rgba(255,255,255,0.08)", padding: "14px 32px", display: "flex", alignItems: "center", gap: 24, position: "sticky", top: 0, zIndex: 100 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 28, height: 28, background: "#e50914", borderRadius: 4, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, fontWeight: 900 }}>C</div>
          <span style={{ fontWeight: 700, fontSize: 16 }}>CineWave ML</span>
        </div>
        <div style={{ display: "flex", gap: 4, flex: 1, overflowX: "auto" }}>
          {sections.map(s => (
            <button key={s.id} onClick={() => setActiveSection(s.id)} style={{
              background: activeSection === s.id ? "#e50914" : "transparent",
              border: "1px solid " + (activeSection === s.id ? "#e50914" : "rgba(255,255,255,0.12)"),
              color: "#fff", borderRadius: 6, padding: "5px 14px", fontSize: 12, cursor: "pointer", whiteSpace: "nowrap", fontWeight: activeSection === s.id ? 700 : 400,
            }}>{s.label}</button>
          ))}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 12, color: "#666" }}>User</span>
          <input type="number" value={userId} onChange={e => setUserId(Number(e.target.value))}
            style={{ width: 60, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, color: "#fff", padding: "4px 8px", fontSize: 13 }} />
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: h.ok ? "#22c55e" : "#ef4444" }} />
          <span style={{ fontSize: 12, color: "#888" }}>API {h.ok ? "online" : "offline"}</span>
        </div>
      </div>

      <div style={{ padding: "32px", maxWidth: 1200, margin: "0 auto" }}>

        {/* OPE Section */}
        {activeSection === "ope" && (
          <div>
            <SectionHeader title="Offline Policy Evaluation" subtitle="IPS + Doubly Robust estimators — evaluate new policies without running live A/B tests" />
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16, marginBottom: 24 }}>
              <StatCard label="IPS NDCG@10" value={ope.estimators?.ips_ndcg_at_k?.toFixed(4) || "—"} sub="Inverse Propensity Scoring" color="#3b82f6" />
              <StatCard label="Doubly Robust NDCG" value={ope.estimators?.doubly_robust_ndcg?.toFixed(4) || "—"} sub="DR correction applied" color="#8b5cf6" />
              <StatCard label="Direct Method" value={ope.estimators?.direct_method_ndcg?.toFixed(4) || "—"} sub="From slice eval" color="#22c55e" />
            </div>
            <div style={{ background: "rgba(255,255,255,0.03)", borderRadius: 10, padding: 20, marginBottom: 24, border: "1px solid rgba(255,255,255,0.06)" }}>
              <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12, color: "#ddd" }}>Position Bias Correction</div>
              {(data.position_bias || null) ? null : (
                <button onClick={() => fetch_data("position_bias", "/eval/position_bias")}
                  style={{ background: "#e50914", border: "none", borderRadius: 6, color: "#fff", padding: "8px 20px", cursor: "pointer", fontSize: 13, fontWeight: 600 }}>
                  {loading.position_bias ? "Loading…" : "Run Position Bias Analysis"}
                </button>
              )}
              {data.position_bias && (
                <div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, marginBottom: 16 }}>
                    {data.position_bias.positions?.slice(0,4).map((pos: number, i: number) => (
                      <div key={pos} style={{ background: "rgba(255,255,255,0.04)", borderRadius: 6, padding: "10px 12px" }}>
                        <div style={{ fontSize: 10, color: "#888", marginBottom: 4 }}>Position {pos}</div>
                        <div style={{ fontSize: 18, fontWeight: 700, color: i === 0 ? "#22c55e" : "#ddd" }}>
                          {(data.position_bias.estimated_ctr_by_position?.[i] * 100)?.toFixed(1)}%
                        </div>
                        <div style={{ fontSize: 10, color: "#666" }}>CTR</div>
                      </div>
                    ))}
                  </div>
                  <div style={{ display: "flex", gap: 12 }}>
                    <div style={{ background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 6, padding: "8px 14px", fontSize: 12 }}>
                      Bias ratio: <strong style={{ color: "#f87171" }}>{data.position_bias.bias_magnitude?.position_bias_ratio}x</strong>
                    </div>
                    <div style={{ background: "rgba(59,130,246,0.1)", border: "1px solid rgba(59,130,246,0.3)", borderRadius: 6, padding: "8px 14px", fontSize: 12 }}>
                      Severity: <strong style={{ color: "#60a5fa" }}>{data.position_bias.bias_magnitude?.severity}</strong>
                    </div>
                  </div>
                </div>
              )}
            </div>
            <div style={{ background: "rgba(255,255,255,0.03)", borderRadius: 10, padding: 20, border: "1px solid rgba(255,255,255,0.06)" }}>
              <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12, color: "#ddd" }}>Goodhart / Long-run Holdout</div>
              {!data.goodhart ? (
                <button onClick={() => fetch_data("goodhart", "/eval/long_run_holdout")}
                  style={{ background: "#1a1a1a", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 6, color: "#fff", padding: "8px 20px", cursor: "pointer", fontSize: 13 }}>
                  {loading.goodhart ? "Loading…" : "Load Holdout Analysis"}
                </button>
              ) : (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 12 }}>
                  <StatCard label="CTR Lift" value={`+${data.goodhart.lifts?.ctr_lift_pct}%`} sub="ML vs holdback" color="#f59e0b" />
                  <StatCard label="30-Day Retention Lift" value={`+${data.goodhart.lifts?.retention_30d_lift_pct}%`} sub="Real satisfaction" color="#22c55e" />
                  <StatCard label="Goodhart Ratio" value={data.goodhart.lifts?.goodhart_ratio} sub={data.goodhart.goodhart_assessment?.verdict?.slice(0,40)} color={data.goodhart.lifts?.goodhart_ratio > 0.5 ? "#22c55e" : "#ef4444"} />
                </div>
              )}
            </div>
          </div>
        )}

        {/* Homepage Row Ordering */}
        {activeSection === "homepage" && (
          <div>
            <SectionHeader title="Homepage Row Ordering" subtitle="Netflix's most visible ML product — which rows appear, in what order, for each user" />
            {loading.homepage ? <div style={{ color: "#888" }}>Loading…</div> : (
              <div>
                <div style={{ display: "flex", gap: 12, marginBottom: 20 }}>
                  <StatCard label="Rows generated" value={String(homepage.n_rows || 0)} sub="Personalised for user" color="#3b82f6" />
                  <StatCard label="Hour of day" value={String(homepage.hour || "—")} sub={homepage.is_weekend ? "Weekend mode" : "Weekday mode"} color="#8b5cf6" />
                  <StatCard label="Device" value={homepage.device || "—"} sub="Context signal" color="#22c55e" />
                  <StatCard label="Interactions" value={String(homepage.n_interactions || 0)} sub="Warm user signals" color="#f59e0b" />
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  {homepage.rows?.map((row: any, i: number) => (
                    <div key={row.row_name} style={{
                      background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)",
                      borderRadius: 10, padding: 16, borderLeft: `3px solid hsl(${i*40},70%,50%)`
                    }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                        <div>
                          <span style={{ fontSize: 11, color: "#888", marginRight: 8 }}>#{i + 1}</span>
                          <span style={{ fontWeight: 700, fontSize: 15 }}>{row.display_title}</span>
                          <span style={{ marginLeft: 8, fontSize: 11, color: "#666" }}>{row.retrieval_intent}</span>
                        </div>
                        <div style={{ fontSize: 13, color: "#f59e0b", fontWeight: 700 }}>
                          score: {row.row_score}
                        </div>
                      </div>
                      <div style={{ display: "flex", gap: 8, overflowX: "auto" }}>
                        {row.items?.map((item: any) => (
                          <div key={item.item_id} style={{ flexShrink: 0, background: "rgba(255,255,255,0.05)", borderRadius: 6, padding: "6px 10px", fontSize: 12 }}>
                            <div style={{ color: "#ddd", marginBottom: 2 }}>{item.title?.slice(0, 20)}</div>
                            <div style={{ color: "#666", fontSize: 11 }}>{item.score?.toFixed(3)}</div>
                          </div>
                        ))}
                        {row.items?.length === 0 && <div style={{ color: "#555", fontSize: 12 }}>No items (cross-row deduped)</div>}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Temporal Taste Drift */}
        {activeSection === "temporal" && (
          <div>
            <SectionHeader title="Temporal Taste Drift" subtitle="How user preferences shift over time — exponential decay weighting α=0.7" />
            {loading.temporal ? <div style={{ color: "#888" }}>Loading…</div> : (
              <div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16, marginBottom: 24 }}>
                  <StatCard label="Taste Stability" value={temporal.taste_stability_score?.toFixed(2) || "—"} sub={temporal.interpretation} color={temporal.taste_stability_score > 0.7 ? "#22c55e" : "#f59e0b"} />
                  <StatCard label="Rising Genres" value={String(temporal.rising_genres?.length || 0)} sub={temporal.rising_genres?.map((g: any) => g.genre).join(", ") || "none"} color="#3b82f6" />
                  <StatCard label="Falling Genres" value={String(temporal.falling_genres?.length || 0)} sub={temporal.falling_genres?.map((g: any) => g.genre).join(", ") || "none"} color="#ef4444" />
                </div>
                {temporal.genre_profiles && (
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 12 }}>
                    {Object.entries(temporal.genre_profiles).map(([genre, profile]: [string, any]) => (
                      <div key={genre} style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 8, padding: 14 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                          <span style={{ fontWeight: 600, fontSize: 14 }}>{genre}</span>
                          <span style={{ fontSize: 12, color: profile.trend === "rising" ? "#22c55e" : profile.trend === "falling" ? "#ef4444" : "#888" }}>
                            {profile.trend} {profile.drift > 0 ? "+" : ""}{profile.drift?.toFixed(2)}
                          </span>
                        </div>
                        <div style={{ display: "flex", gap: 8 }}>
                          <div style={{ flex: 1, background: "rgba(255,255,255,0.04)", borderRadius: 4, padding: "6px 8px", fontSize: 11 }}>
                            <div style={{ color: "#888" }}>Historical</div>
                            <div style={{ color: "#ddd", fontWeight: 600 }}>{profile.historical_mean?.toFixed(2)}</div>
                          </div>
                          <div style={{ flex: 1, background: "rgba(255,255,255,0.04)", borderRadius: 4, padding: "6px 8px", fontSize: 11 }}>
                            <div style={{ color: "#888" }}>Recent</div>
                            <div style={{ color: "#ddd", fontWeight: 600 }}>{profile.recent_mean?.toFixed(2)}</div>
                          </div>
                          <div style={{ flex: 1, background: "rgba(255,255,255,0.04)", borderRadius: 4, padding: "6px 8px", fontSize: 11 }}>
                            <div style={{ color: "#888" }}>Current</div>
                            <div style={{ color: "#60a5fa", fontWeight: 700 }}>{(profile.current_affinity * 100)?.toFixed(0)}%</div>
                          </div>
                        </div>
                        <div style={{ marginTop: 6, height: 3, background: "rgba(255,255,255,0.06)", borderRadius: 2, overflow: "hidden" }}>
                          <div style={{ height: "100%", width: `${profile.current_affinity * 100}%`, background: profile.trend === "rising" ? "#22c55e" : profile.trend === "falling" ? "#ef4444" : "#3b82f6", transition: "width 0.5s" }} />
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* MMR Diversity */}
        {activeSection === "mmr" && (
          <div>
            <SectionHeader title="MMR Diversity Ranking" subtitle="Maximal Marginal Relevance — balance relevance vs diversity, prevent filter bubbles" />
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16, marginBottom: 24 }}>
              <StatCard label="Diversity Score" value={(mmr.diversity_score * 100)?.toFixed(0) + "%" || "—"} sub={mmr.interpretation} color="#8b5cf6" />
              <StatCard label="Unique Genres" value={String(mmr.n_unique_genres || 0)} sub="In result set" color="#22c55e" />
              <StatCard label="Lambda λ" value="0.50" sub="Balanced mode" color="#f59e0b" />
            </div>
            <div style={{ marginBottom: 16, display: "flex", gap: 8, alignItems: "center" }}>
              <span style={{ fontSize: 12, color: "#888" }}>λ = 0 (max diversity)</span>
              <input type="range" min="0" max="100" defaultValue="50"
                style={{ flex: 1 }}
                onChange={e => fetch_data("mmr", `/recommend/mmr/${userId}?lambda_div=${Number(e.target.value)/100}&k=8`)} />
              <span style={{ fontSize: 12, color: "#888" }}>λ = 1 (pure relevance)</span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 10 }}>
              {mmr.items?.map((item: any, i: number) => (
                <div key={item.item_id} style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 8, padding: 14, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>{item.title?.slice(0,30)}</div>
                    <div style={{ fontSize: 11, color: "#888" }}>{item.genre}</div>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <div style={{ fontSize: 12, color: "#60a5fa" }}>mmr: {item.mmr_score?.toFixed(3)}</div>
                    <div style={{ fontSize: 11, color: "#555" }}>rel: {item.score?.toFixed(3)}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Cold-Start */}
        {activeSection === "coldstart" && (
          <div>
            <SectionHeader title="Cold-Start Cascade" subtitle="3-stage handler: warm user ML → semi-cold genre boost → fully-cold diversity" />
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16, marginBottom: 24 }}>
              <StatCard label="Stage" value={cold.stage || "—"} sub="Current handler level" color={cold.stage === "warm_user_ml" ? "#22c55e" : cold.stage?.includes("semi") ? "#f59e0b" : "#ef4444"} />
              <StatCard label="Interactions" value={String(cold.n_interactions || 0)} sub="Signal count" color="#3b82f6" />
              <StatCard label="Items returned" value={String(cold.items?.length || 0)} sub="Recommendations" color="#8b5cf6" />
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 10, marginBottom: 20 }}>
              {["warm_user_ml", "semi_cold_genre_boosted", "fully_cold_diverse_popularity"].map(stage => (
                <div key={stage} style={{
                  background: cold.stage === stage ? "rgba(229,9,20,0.1)" : "rgba(255,255,255,0.02)",
                  border: `1px solid ${cold.stage === stage ? "#e50914" : "rgba(255,255,255,0.06)"}`,
                  borderRadius: 8, padding: 14,
                }}>
                  <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6, color: cold.stage === stage ? "#fff" : "#888" }}>
                    {stage === "warm_user_ml" ? "Stage 1: Warm" : stage.includes("semi") ? "Stage 2: Semi-cold" : "Stage 3: Cold"}
                  </div>
                  <div style={{ fontSize: 11, color: "#666" }}>
                    {stage === "warm_user_ml" ? "≥10 interactions → Full ML pipeline" : stage.includes("semi") ? "3–9 interactions → Genre-boosted popularity" : "0–2 interactions → Max diversity"}
                  </div>
                </div>
              ))}
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 10 }}>
              {cold.items?.map((item: any) => (
                <div key={item.item_id} style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 8, padding: 12, display: "flex", justifyContent: "space-between" }}>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 600 }}>{item.title?.slice(0,28)}</div>
                    <div style={{ fontSize: 11, color: "#888", marginTop: 2 }}>{item.primary_genre}</div>
                  </div>
                  <div style={{ fontSize: 12, color: "#f59e0b" }}>{item.score?.toFixed(3)}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Notifications */}
        {activeSection === "notify" && (
          <div>
            <SectionHeader title="Notification Optimisation" subtitle="ML-ranked push notifications: content selector + time optimiser + send gate" />
            {loading.notify ? <div style={{ color: "#888" }}>Loading…</div> : (
              <div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 24 }}>
                  <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 10, padding: 20 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 16, color: "#ddd" }}>Content Selector</div>
                    {notify.recommended_content && (
                      <div>
                        <div style={{ fontWeight: 700, fontSize: 18, marginBottom: 8 }}>{notify.recommended_content.title}</div>
                        <div style={{ fontSize: 12, color: "#888", marginBottom: 12 }}>{notify.recommended_content.reason}</div>
                        <div style={{ background: "rgba(59,130,246,0.1)", border: "1px solid rgba(59,130,246,0.3)", borderRadius: 6, padding: "6px 12px", fontSize: 12, display: "inline-block" }}>
                          {notify.recommended_content.genre}
                        </div>
                      </div>
                    )}
                  </div>
                  <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 10, padding: 20 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 16, color: "#ddd" }}>Time Optimiser</div>
                    {notify.timing && (
                      <div>
                        <div style={{ display: "flex", gap: 12, marginBottom: 12 }}>
                          <div>
                            <div style={{ fontSize: 11, color: "#888" }}>Current open prob</div>
                            <div style={{ fontSize: 24, fontWeight: 700, color: notify.timing.open_probability_now > 0.5 ? "#22c55e" : "#f59e0b" }}>
                              {(notify.timing.open_probability_now * 100).toFixed(0)}%
                            </div>
                          </div>
                          <div>
                            <div style={{ fontSize: 11, color: "#888" }}>Optimal hour</div>
                            <div style={{ fontSize: 24, fontWeight: 700, color: "#60a5fa" }}>{notify.timing.optimal_send_hour}:00</div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
                <div style={{
                  background: notify.send_decision?.should_send ? "rgba(34,197,94,0.08)" : "rgba(239,68,68,0.08)",
                  border: `1px solid ${notify.send_decision?.should_send ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)"}`,
                  borderRadius: 10, padding: 20
                }}>
                  <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 8, color: notify.send_decision?.should_send ? "#22c55e" : "#ef4444" }}>
                    {notify.send_decision?.should_send ? "✓ Send notification" : "✗ Suppressed"}
                  </div>
                  <div style={{ fontSize: 13, color: "#888" }}>{notify.send_decision?.reason}</div>
                  <div style={{ marginTop: 12, fontSize: 12, color: "#666" }}>{notify.impact}</div>
                </div>
              </div>
            )}

            <div style={{ marginTop: 24 }}>
              <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12, color: "#ddd" }}>Session Genre Affinities (Real-time)</div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 12 }}>
                {Object.entries(session_aff.session_affinities || {}).map(([genre, score]: [string, any]) => (
                  <div key={genre} style={{
                    background: score > 0 ? "rgba(34,197,94,0.1)" : "rgba(239,68,68,0.1)",
                    border: `1px solid ${score > 0 ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)"}`,
                    borderRadius: 6, padding: "4px 10px", fontSize: 12
                  }}>
                    {genre}: <strong>{score > 0 ? "+" : ""}{score}</strong>
                  </div>
                ))}
                {Object.keys(session_aff.session_affinities || {}).length === 0 && (
                  <div style={{ fontSize: 12, color: "#666" }}>No session signals yet. Click items to build genre affinity.</div>
                )}
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                {["Drama", "Action", "Comedy", "Thriller"].map(genre => (
                  <button key={genre} onClick={() => {
                    fetch(`${API}/session/${userId}/genre_signal?item_id=1&signal=click`, { method: "POST" })
                      .then(() => fetch_data("session_aff", `/session/${userId}/genre_affinities`));
                  }} style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, color: "#fff", padding: "6px 14px", cursor: "pointer", fontSize: 12 }}>
                    + {genre} click
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Infrastructure */}
        {activeSection === "infra" && (
          <div>
            <SectionHeader title="Infrastructure & MLOps" subtitle="Metaflow, Kafka, feature freshness, drift monitoring" />
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 16, marginBottom: 24 }}>
              <StatCard label="Metaflow" value={metaflow.available ? "Active" : "Inactive"} sub={`Source: ${metaflow.bundle_source || "—"}`} color={metaflow.available ? "#22c55e" : "#ef4444"} />
              <StatCard label="Kafka Bridge" value={metaflow.kafka_running ? "Running" : "Fallback"} sub="Event streaming" color={metaflow.kafka_running ? "#22c55e" : "#888"} />
              <StatCard label="Drift Status" value={drift.prediction_drift?.status || "—"} sub={`n=${drift.prediction_drift?.n_scores || 0} scores`} color={drift.prediction_drift?.status === "alert" ? "#ef4444" : "#22c55e"} />
              <StatCard label="PSI Skew" value={skew.status || "—"} sub={`max_psi=${skew.max_psi?.toFixed(4) || "—"}`} color={skew.status === "alert" ? "#f59e0b" : "#22c55e"} />
            </div>
            {features.feature_pipelines && (
              <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 10, padding: 20, marginBottom: 20 }}>
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 14, color: "#ddd" }}>Feature Pipelines ({features.n_features} tracked)</div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 8 }}>
                  {Object.entries(features.feature_pipelines || {}).map(([feat, info]: [string, any]) => (
                    <div key={feat} style={{ background: "rgba(255,255,255,0.03)", borderRadius: 6, padding: "10px 12px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <div>
                        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 2 }}>{feat.replace(/_/g, " ")}</div>
                        <div style={{ fontSize: 10, color: "#666" }}>{info.pipeline} · {info.cadence}</div>
                      </div>
                      <div style={{ width: 8, height: 8, borderRadius: "50%", background: info.status === "ok" ? "#22c55e" : "#ef4444" }} />
                    </div>
                  ))}
                </div>
              </div>
            )}
            <div style={{ display: "flex", gap: 12 }}>
              <button onClick={() => fetch_data("metaflow_refresh", "/metaflow/refresh", "POST")}
                style={{ background: "#1a1a1a", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 6, color: "#fff", padding: "8px 20px", cursor: "pointer", fontSize: 13 }}>
                {loading.metaflow_refresh ? "Refreshing…" : "Trigger Metaflow Refresh"}
              </button>
              {data.metaflow_refresh && (
                <div style={{ padding: "8px 12px", background: "rgba(34,197,94,0.1)", border: "1px solid rgba(34,197,94,0.3)", borderRadius: 6, fontSize: 12 }}>
                  ok={String(data.metaflow_refresh.ok)} · refreshed={String(data.metaflow_refresh.refreshed)}
                </div>
              )}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
