-- CineWave RecSys Analytical SQL Queries
-- SELECT + JOIN + GROUP BY examples

-- Query 1: NDCG@10 per policy version
SELECT r.policy_version,
       COUNT(DISTINCT r.user_id) AS unique_users,
       ROUND(AVG(
           CASE WHEN e.event_type = 'play_start' THEN 1.0
                WHEN e.event_type = 'watch_90pct' THEN 2.0
                ELSE 0.0 END / LOG(2, r.rank + 1)
       )::NUMERIC, 4) AS ndcg_at_10,
       ROUND(AVG(r.als_score)::NUMERIC, 4) AS avg_als_score
FROM recommendations r
LEFT JOIN events e ON e.user_id = r.user_id AND e.item_id = r.item_id
WHERE r.rank <= 10
  AND r.served_at >= NOW() - INTERVAL '30 days'
GROUP BY r.policy_version
ORDER BY ndcg_at_10 DESC;

-- Query 2: CTR and session depth by user activity decile
SELECT u.activity_decile,
       u.profile_name,
       COUNT(DISTINCT u.user_id) AS users,
       COUNT(CASE WHEN e.event_type = 'play_start' THEN 1 END) AS plays,
       COUNT(CASE WHEN e.event_type = 'watch_90pct' THEN 1 END) AS completions,
       COUNT(CASE WHEN e.event_type = 'skip' THEN 1 END) AS skips,
       ROUND(COUNT(CASE WHEN e.event_type = 'play_start' THEN 1 END) * 100.0
             / NULLIF(COUNT(DISTINCT r.rec_id), 0), 2) AS ctr_pct
FROM users u
JOIN recommendations r ON r.user_id = u.user_id
LEFT JOIN events e ON e.user_id = u.user_id AND e.item_id = r.item_id
WHERE r.served_at >= NOW() - INTERVAL '30 days'
GROUP BY u.activity_decile, u.profile_name
HAVING COUNT(DISTINCT u.user_id) >= 10
ORDER BY u.activity_decile;

-- Query 3: Reward per session (for offline RL / off-policy RL policy update)
SELECT e.user_id,
       e.session_id,
       u.profile_name,
       COUNT(*) AS events_in_session,
       SUM(e.reward) AS total_reward,
       ROUND(AVG(e.reward)::NUMERIC, 3) AS avg_reward
FROM events e
JOIN users u ON u.user_id = e.user_id
WHERE e.session_id IS NOT NULL
  AND e.reward IS NOT NULL
  AND e.occurred_at >= NOW() - INTERVAL '24 hours'
GROUP BY e.user_id, e.session_id, u.profile_name
HAVING COUNT(*) >= 2
ORDER BY total_reward DESC
LIMIT 1000;
