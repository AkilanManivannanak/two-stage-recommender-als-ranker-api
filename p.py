"""
p.py — CineWave TMDB Catalog Patcher v3
Fetches 1200+ real movies from TMDB with real posters.
Run: docker cp p.py recsys_api:/app/p.py && docker exec recsys_api python3 /app/p.py
"""
import json, sys, time, os

try:
    import urllib.request
except ImportError:
    sys.exit(1)

TMDB_KEY    = "191853b81cda0419b8fb4e79f32bddb8"
TMDB_BASE   = "https://api.themoviedb.org/3"
POSTER_BASE = "https://image.tmdb.org/t/p/w500"

SPECIFIC_FIXES = {
    "Alice in Borderland":         "https://image.tmdb.org/t/p/w500/20mOwPg9gNXUAj78hdXWoOZBNhiB.jpg",
    "Gladiator II":                "https://image.tmdb.org/t/p/w500/2cxhvwyE0RtuhMkstmKTkzKzrpq.jpg",
    "The Last Dance":              "https://image.tmdb.org/t/p/w500/7GBMipjNFf7qKOvl6tZMRF68mvg.jpg",
    "Deadpool & Wolverine":        "https://image.tmdb.org/t/p/w500/8cdWjvZQUExUUTzyp4IoijrKMWe.jpg",
    "Deadpool and Wolverine":      "https://image.tmdb.org/t/p/w500/8cdWjvZQUExUUTzyp4IoijrKMWe.jpg",
    "Money Heist: Korea":          "https://image.tmdb.org/t/p/w500/sWMoMDrGJv2EFPAeNbCvHJbKNAS.jpg",
    "Red Notice":                  "https://image.tmdb.org/t/p/w500/wdE6ewaKZHr62bLqCn7A2DiGShm.jpg",
    "The Gray Man":                "https://image.tmdb.org/t/p/w500/myABCsnMTGcVHwfYw138Ma8s9Ko.jpg",
    "The Night Agent":             "https://image.tmdb.org/t/p/w500/oWQFGlFPyVwfHMBhCHMaUgmmb3J.jpg",
    "Get Out":                     "https://image.tmdb.org/t/p/w500/tFXcEccSQMVl9dsvXCMomM8Uh1r.jpg",
    "The Haunting of Bly Manor":   "https://image.tmdb.org/t/p/w500/gEvKgLMWEpzGdWfFBBrgThMPjLy.jpg",
    "Nosferatu":                   "https://image.tmdb.org/t/p/w500/5qGIxdEO841C29qqkbMvsLAifwR.jpg",
    "Ripley":                      "https://image.tmdb.org/t/p/w500/rO5mBxfbLyqT9SeEaAO4CMVFouq.jpg",
    "The Watcher":                 "https://image.tmdb.org/t/p/w500/rDiDCmFuq7A9ILFELsJFm1b6PJL.jpg",
    "Baby Reindeer":               "https://image.tmdb.org/t/p/w500/jMyBcMj4VmyEdwCsaJYmqVX1fcg.jpg",
    "Formula 1: Drive to Survive": "https://image.tmdb.org/t/p/w500/jTRpSaLoEKqNHicN2oNNFMIADe4.jpg",
    "Our Planet":                  "https://image.tmdb.org/t/p/w500/GWokSNkLNxYC7aVRBYjJLuBGmUG.jpg",
    "Avatar: The Last Airbender":  "https://image.tmdb.org/t/p/w500/coh8mWBGpHJgHfGEZsITEmSXDFP.jpg",
}

GENRE_MAP = {
    28:"Action",12:"Adventure",16:"Animation",35:"Comedy",80:"Crime",
    99:"Documentary",18:"Drama",10751:"Family",14:"Fantasy",36:"History",
    27:"Horror",10402:"Music",9648:"Mystery",10749:"Romance",878:"Sci-Fi",
    53:"Thriller",10752:"War",37:"Western",
}

def tmdb_get(path, params=None):
    url = f"{TMDB_BASE}{path}?api_key={TMDB_KEY}&language=en-US"
    if params:
        for k, v in params.items():
            url += f"&{k}={v}"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return None

def fetch_tmdb_movies(target=1200):
    movies = {}
    print(f"  Fetching {target}+ movies from TMDB...")
    for endpoint in ["/movie/popular", "/movie/top_rated", "/movie/now_playing"]:
        for page in range(1, 25):
            data = tmdb_get(endpoint, {"page": page})
            if not data:
                continue
            for m in data.get("results", []):
                mid = str(m.get("id",""))
                if not mid or mid in movies:
                    continue
                if not m.get("poster_path") or not m.get("title"):
                    continue
                gids = m.get("genre_ids", [])
                genre = GENRE_MAP.get(gids[0], "Drama") if gids else "Drama"
                yr = (m.get("release_date") or "")[:4]
                movies[mid] = {
                    "item_id": None,
                    "title": m["title"],
                    "poster_url": POSTER_BASE + m["poster_path"],
                    "primary_genre": genre,
                    "year": int(yr) if yr.isdigit() else None,
                    "avg_rating": round(m.get("vote_average", 7.0), 1),
                    "popularity": round(m.get("popularity", 50.0), 1),
                    "description": m.get("overview", ""),
                    "maturity_rating": "PG-13",
                }
            if len(movies) >= target:
                break
            time.sleep(0.04)
        if len(movies) >= target:
            break
    print(f"  Fetched {len(movies)} unique movies from TMDB")
    return list(movies.values())

def norm(s):
    return str(s or "").lower().strip()

def run():
    print("="*55)
    print("CineWave TMDB Catalog Patcher v3")
    print("="*55)

    print("\n[1] Fetching from TMDB...")
    tmdb_movies = fetch_tmdb_movies(1200)

    try:
        sys.path.insert(0, "/app/src")
        from recsys.serving.app import CATALOG
    except Exception as e:
        print(f"Cannot import CATALOG: {e}")
        return

    # Build TMDB lookup
    tmdb_by_title = {norm(m["title"]): m for m in tmdb_movies}
    existing_posters = {v.get("poster_url","") for v in CATALOG.values()}

    print(f"\n[2] Patching {len(CATALOG)} existing catalog entries...")
    poster_fixed = 0
    for item_id, item in CATALOG.items():
        title = item.get("title","")
        key = norm(title)

        # Apply specific fixes
        for fix_t, fix_url in SPECIFIC_FIXES.items():
            if norm(fix_t) == key:
                item["poster_url"] = fix_url
                poster_fixed += 1
                break

        # Match TMDB
        if key in tmdb_by_title:
            tmdb = tmdb_by_title[key]
            cur = item.get("poster_url","")
            if not cur or "NUDE" in cur or not cur.startswith("https://image.tmdb.org"):
                item["poster_url"] = tmdb["poster_url"]
                poster_fixed += 1
            if not item.get("description") and tmdb.get("description"):
                item["description"] = tmdb["description"]
            if not item.get("year") and tmdb.get("year"):
                item["year"] = tmdb["year"]

    print(f"  Posters fixed: {poster_fixed}")

    print(f"\n[3] Injecting new movies to reach 1200+...")
    existing_titles = {norm(v.get("title","")) for v in CATALOG.values()}
    max_id = max(CATALOG.keys()) if CATALOG else 10000
    added = 0

    for m in tmdb_movies:
        key = norm(m["title"])
        if key in existing_titles:
            continue
        if m["poster_url"] in existing_posters:
            continue
        max_id += 1
        m["item_id"] = max_id
        CATALOG[max_id] = m
        existing_titles.add(key)
        existing_posters.add(m["poster_url"])
        added += 1

    print(f"  Added {added} new movies")
    print(f"  Catalog size: {len(CATALOG)}")
    print("\n" + "="*55)
    print(f"Done! {len(CATALOG)} movies with real TMDB posters")
    print("="*55)

run()
