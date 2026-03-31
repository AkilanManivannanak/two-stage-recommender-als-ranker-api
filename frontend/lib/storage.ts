// Safe localStorage helpers (per-user session persistence)
export const STORAGE_KEYS = {
  activeUserId: "cinewave.active_user_id",
  devOverlay: "cinewave.dev_overlay",
  shadowCompare: "cinewave.shadow_compare",
  sessionPrefix: "cinewave.session.user.", // + user_id
};

function isBrowser(): boolean {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

export function readJson<T>(key: string, fallback: T): T {
  if (!isBrowser()) return fallback;
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

export function writeJson<T>(key: string, value: T): void {
  if (!isBrowser()) return;
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // ignore
  }
}

export function readString(key: string, fallback: string | null = null): string | null {
  if (!isBrowser()) return fallback;
  try {
    const v = window.localStorage.getItem(key);
    return v ?? fallback;
  } catch {
    return fallback;
  }
}

export function writeString(key: string, value: string): void {
  if (!isBrowser()) return;
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // ignore
  }
}

export function removeKey(key: string): void {
  if (!isBrowser()) return;
  try {
    window.localStorage.removeItem(key);
  } catch {
    // ignore
  }
}

export function getActiveUserId(): number | null {
  const raw = readString(STORAGE_KEYS.activeUserId, null);
  if (!raw) return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

export function setActiveUserId(userId: number): void {
  writeString(STORAGE_KEYS.activeUserId, String(userId));
}

export function clearActiveUserId(): void {
  removeKey(STORAGE_KEYS.activeUserId);
}

export function getSessionItemIds(userId: number): number[] {
  const key = STORAGE_KEYS.sessionPrefix + String(userId);
  const arr = readJson<number[]>(key, []);
  return Array.isArray(arr) ? arr.map((x) => Number(x)).filter((x) => Number.isFinite(x)) : [];
}

export function setSessionItemIds(userId: number, itemIds: number[]): void {
  const key = STORAGE_KEYS.sessionPrefix + String(userId);
  const clean = itemIds.map((x) => Number(x)).filter((x) => Number.isFinite(x));
  writeJson(key, clean);
}

export function clearSessionItemIds(userId: number): void {
  const key = STORAGE_KEYS.sessionPrefix + String(userId);
  removeKey(key);
}

// -----------------
// Last /recommend request metadata per user + row
// (used to enrich feedback context and title pages)
// -----------------

export type RowKey = "top" | "because" | "popular" | "diverse" | "cold";

export type RowRequestMeta = {
  requestId: string;
  bundleId: string | null;
  ts: number;
};

function rowMetaKey(userId: number) {
  return `cinewave.rowmeta.${userId}`;
}

export function getRowMeta(userId: number): Partial<Record<RowKey, RowRequestMeta>> {
  return readJson(rowMetaKey(userId), {});
}

export function setRowMeta(userId: number, row: RowKey, meta: RowRequestMeta) {
  const cur = getRowMeta(userId);
  cur[row] = meta;
  writeJson(rowMetaKey(userId), cur);
}
