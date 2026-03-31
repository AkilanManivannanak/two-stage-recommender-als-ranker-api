'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import { api, CatalogItem } from '@/lib/api';
import { useAppState, useAppDispatch } from '@/lib/store';

export default function TitlePage() {
  const params = useParams();
  const itemId = Number(params?.itemId);
  const { activeUser } = useAppState();
  const dispatch = useAppDispatch();
  const userId = activeUser?.user_id ?? 1;

  const [item, setItem] = useState<CatalogItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [explanation, setExplanation] = useState<string | null>(null);

  useEffect(() => {
    if (!Number.isFinite(itemId)) return;
    setLoading(true);
    Promise.allSettled([
      api.item(itemId),
      api.explain({ user_id: userId, item_ids: [itemId] }),
    ]).then(([itemRes, explainRes]) => {
      if (itemRes.status === 'fulfilled') setItem(itemRes.value as CatalogItem);
      if (explainRes.status === 'fulfilled') {
        const e = (explainRes.value as any).explanations?.[0];
        if (e) setExplanation(e.reason);
      }
    }).finally(() => setLoading(false));
  }, [itemId, userId]);

  const sendFeedback = (event: 'play' | 'like' | 'dislike' | 'add_to_list') => {
    api.feedback({ user_id: userId, item_id: itemId, event }).catch(() => {});
    dispatch({ type: 'ADD_SESSION_ITEM', payload: itemId });
  };

  if (loading) return <div className="flex items-center justify-center min-h-screen text-white/60">Loading…</div>;
  if (!item) return <div className="flex items-center justify-center min-h-screen text-white/60">Item not found</div>;

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      <div className="grid grid-cols-1 md:grid-cols-[240px_1fr] gap-8">
        <div className="w-full aspect-[2/3]">
          {item.poster_url
            ? <img src={item.poster_url} alt={item.title} className="w-full h-full object-cover rounded-xl border border-white/10" />
            : <div className="w-full h-full rounded-xl bg-white/5 border border-white/10 flex items-center justify-center text-white/40">No poster</div>
          }
        </div>
        <div>
          <h1 className="text-3xl font-bold text-white">{item.title}</h1>
          <div className="flex gap-2 mt-2 text-sm text-cine-muted">
            {item.year && <span>{item.year}</span>}
            {item.primary_genre && <span className="text-cine-accent">{item.primary_genre}</span>}
            {item.maturity_rating && <span className="border border-cine-border px-1">{item.maturity_rating}</span>}
          </div>
          <p className="mt-4 text-cine-text-dim leading-relaxed">{item.description || 'No description available.'}</p>
          {explanation && (
            <div className="mt-4 bg-cine-card border border-cine-border rounded-lg p-3">
              <p className="text-xs text-cine-accent font-semibold mb-1">Why recommended</p>
              <p className="text-xs text-cine-text-dim">{explanation}</p>
            </div>
          )}
          <div className="flex flex-wrap gap-3 mt-6">
            <button onClick={() => sendFeedback('play')} className="px-5 py-2.5 rounded bg-white text-black font-bold hover:bg-white/90 transition">▶ Play</button>
            <button onClick={() => sendFeedback('like')} className="px-4 py-2.5 rounded bg-cine-card border border-cine-border text-white hover:border-white/40 transition">👍</button>
            <button onClick={() => sendFeedback('dislike')} className="px-4 py-2.5 rounded bg-cine-card border border-cine-border text-white hover:border-red-500/60 transition">👎</button>
            <button onClick={() => sendFeedback('add_to_list')} className="px-4 py-2.5 rounded bg-cine-card border border-cine-border text-white hover:border-white/40 transition">+ Watchlist</button>
          </div>
        </div>
      </div>
    </div>
  );
}
