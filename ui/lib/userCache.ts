// Lightweight client-side cache for /api/users/lookup. The user table doesn't
// change often (a name/avatar update is rare), so caching by username for a
// few minutes saves a round-trip every time someone re-opens the Projects
// page or scrolls between tabs.

const STORAGE_KEY = "pixelkit.userCache.v1";
// 24h TTL: user names + avatars change rarely. invalidateUser() drops the
// caller's own row when they update their profile, so they see fresh state
// immediately without making everyone else re-query.
const TTL_MS = 24 * 60 * 60 * 1000;

export type OwnerInfo = { name: string | null; image: string | null };
type Entry = OwnerInfo & { ts: number };
type Store = Record<string, Entry>;


function load(): Store {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Store) : {};
  } catch {
    return {};
  }
}


function save(store: Store) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
  } catch {
    // Quota or private mode, silently degrade to no cache.
  }
}


/** Returns user info for the given usernames, hitting the network only for
 *  the ones missing from cache or older than the TTL. */
export async function lookupUsers(usernames: string[]): Promise<Record<string, OwnerInfo>> {
  const wanted = Array.from(new Set(usernames.map((u) => u.toLowerCase()).filter(Boolean)));
  if (wanted.length === 0) return {};

  const store = load();
  const now = Date.now();
  const result: Record<string, OwnerInfo> = {};
  const stale: string[] = [];

  for (const u of wanted) {
    const e = store[u];
    if (e && now - e.ts < TTL_MS) {
      result[u] = { name: e.name, image: e.image };
    } else {
      stale.push(u);
    }
  }

  if (stale.length === 0) return result;

  try {
    const r = await fetch(
      `/api/users/lookup?usernames=${encodeURIComponent(stale.join(","))}`,
      { cache: "no-store" },
    );
    if (r.ok) {
      const fresh: Record<string, OwnerInfo> = await r.json();
      const next: Store = { ...store };
      for (const u of stale) {
        // Even a username with no DB row gets cached as a negative hit so
        // we don't keep re-asking for handles that don't exist.
        const info = fresh[u] ?? { name: null, image: null };
        result[u] = info;
        next[u] = { ...info, ts: now };
      }
      save(next);
    }
  } catch {
    // Network failure, return whatever we already had from cache.
  }

  return result;
}


/** Wipe a single user from the cache so the next lookup re-fetches.
 *  Call this after the current user changes their own avatar/name. */
export function invalidateUser(username: string) {
  if (!username) return;
  const store = load();
  delete store[username.toLowerCase()];
  save(store);
}
