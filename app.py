"""GameSense — single-file Streamlit app for game recommendations.

Inference is pure NumPy against the pre-trained artifacts shipped in this repo.
Nothing in this file trains, modifies or replaces the model.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import base64
import html
import json
import random
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = APP_DIR / "gamesense_artifacts.npz"
GAMES_PATH = APP_DIR / "games.json"
REPORT_PATH = APP_DIR / "eval_report.md"

FUZZY_CUTOFF = 0.75
STEAM_SEARCH_URL = "https://store.steampowered.com/api/storesearch/"
USER_AGENT = "GameSense/1.0 (+streamlit app)"


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #


@dataclass
class Catalog:
    titles: list[str]
    n_players: np.ndarray
    normalized: list[str]
    norm_to_indices: dict[str, list[int]]
    by_popularity: list[str]


@dataclass
class Match:
    query: str
    title: str | None = None
    index: int | None = None
    players: int = 0
    stage: str = "not_found"  # exact | whole-word | fuzzy | not_found | empty
    score: float = 0.0
    duplicate: bool = False


@dataclass
class Artifacts:
    H: np.ndarray
    Q: np.ndarray
    b: np.ndarray
    alpha: float
    catalog: Catalog


# --------------------------------------------------------------------------- #
# Pure logic (unit-tested in tests/test_app.py)
# --------------------------------------------------------------------------- #


def normalize_title(text: str) -> str:
    """Lowercase, collapse every run of non-alphanumerics to one space, strip."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def build_catalog(titles: list[str], n_players: np.ndarray) -> Catalog:
    titles = [str(t) for t in titles]
    players = np.asarray(n_players, dtype=np.int64)
    if len(titles) != players.size:
        raise ValueError("titles and n_players must have the same length")
    normalized = [normalize_title(t) for t in titles]
    norm_to_indices: dict[str, list[int]] = {}
    for i, nrm in enumerate(normalized):
        norm_to_indices.setdefault(nrm, []).append(i)
    order = np.argsort(-players, kind="mergesort")
    return Catalog(
        titles=titles,
        n_players=players,
        normalized=normalized,
        norm_to_indices=norm_to_indices,
        by_popularity=[titles[i] for i in order],
    )


def match_title(query: str, catalog: Catalog) -> Match:
    """Three-stage matcher: exact -> whole-word containment -> fuzzy (>= 0.75)."""
    raw = str(query).strip()
    norm = normalize_title(raw)
    if not norm:
        return Match(query=raw, stage="empty")

    exact = catalog.norm_to_indices.get(norm)
    if exact:
        idx = max(exact, key=lambda i: int(catalog.n_players[i]))
        return Match(raw, catalog.titles[idx], idx, int(catalog.n_players[idx]), "exact")

    # Whole-word containment: every query word appears as a word in the title.
    # Ties go to the most-played game.
    q_words = norm.split()
    contained: int | None = None
    for i, nrm in enumerate(catalog.normalized):
        words = nrm.split()
        if q_words and all(w in words for w in q_words):
            if contained is None or catalog.n_players[i] > catalog.n_players[contained]:
                contained = i
    if contained is not None:
        return Match(
            raw,
            catalog.titles[contained],
            contained,
            int(catalog.n_players[contained]),
            "whole-word",
        )

    best_idx: int | None = None
    best_ratio = 0.0
    for i, nrm in enumerate(catalog.normalized):
        ratio = SequenceMatcher(None, norm, nrm).ratio()
        if ratio > best_ratio:
            best_ratio, best_idx = ratio, i
    if best_idx is not None and best_ratio >= FUZZY_CUTOFF:
        return Match(
            raw,
            catalog.titles[best_idx],
            best_idx,
            int(catalog.n_players[best_idx]),
            "fuzzy",
            score=best_ratio,
        )

    return Match(query=raw, stage="not_found")


def match_lines(lines: list[str], catalog: Catalog) -> tuple[list[Match], list[Match]]:
    """Match free-text lines, flagging repeats of an already-matched game."""
    matched: list[Match] = []
    unmatched: list[Match] = []
    seen: set[int] = set()
    for line in lines:
        if not str(line).strip():
            continue
        m = match_title(line, catalog)
        if m.index is None:
            unmatched.append(m)
        else:
            m.duplicate = m.index in seen
            seen.add(m.index)
            matched.append(m)
    return matched, unmatched


def rank_recommendations(
    H: np.ndarray,
    Q: np.ndarray,
    b: np.ndarray,
    alpha: float,
    liked_indices: list[int],
    k: int = 10,
) -> list[dict]:
    """Score every game for the liked profile and return the top k.

    user_vec = sum(H[liked]) / n ** alpha ; scores = user_vec @ Q.T + b
    Already-liked games are masked out. Ties break on original item order.
    """
    idx = np.unique(np.asarray(liked_indices, dtype=np.int64))
    if idx.size == 0:
        raise ValueError("At least one liked game index is required.")
    n = idx.size
    user_vec = H[idx].sum(axis=0) / (n ** float(alpha))
    scores = user_vec.astype(np.float64) @ Q.T + b
    scores[idx] = -np.inf

    available = int(np.isfinite(scores).sum())
    if available == 0:
        return []
    k = max(1, min(int(k), available))
    top = np.argsort(-scores, kind="mergesort")[:k]
    because = idx[np.argmax(H[idx] @ Q[top].T, axis=0)]
    return [
        {
            "rank": rank,
            "game_index": int(j),
            "because_index": int(ci),
            "score": float(scores[j]),
        }
        for rank, (j, ci) in enumerate(zip(top.tolist(), because.tolist()), start=1)
    ]


def load_artifacts(base_dir: Path) -> Artifacts:
    with np.load(Path(base_dir) / ARTIFACT_PATH.name) as art:
        H = art["H"].astype(np.float32, copy=False)
        Q = art["Q"].astype(np.float32, copy=False)
        b = art["b"].astype(np.float32, copy=False)
        alpha = float(art["alpha"])
    meta = json.loads((Path(base_dir) / GAMES_PATH.name).read_text(encoding="utf-8"))
    catalog = build_catalog(meta["titles"], meta["n_players"])
    if not (H.shape[0] == Q.shape[0] == b.shape[0] == len(catalog.titles)):
        raise ValueError("Artifacts and games.json disagree on the number of games")
    return Artifacts(H=H, Q=Q, b=b, alpha=alpha, catalog=catalog)


@st.cache_resource(show_spinner=False)
def load_artifacts_cached() -> Artifacts:
    return load_artifacts(APP_DIR)


# --------------------------------------------------------------------------- #
# Optional cover art (degrades to a placeholder offline)
# --------------------------------------------------------------------------- #


def _download_as_data_uri(url: str) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            ctype = resp.headers.get_content_type()
            if not ctype.startswith("image/"):
                return None
            blob = resp.read(500_001)
        if not blob or len(blob) > 500_000:
            return None
        return f"data:{ctype};base64,{base64.b64encode(blob).decode('ascii')}"
    except Exception:
        return None


def fetch_cover(title: str) -> str | None:
    """Best-effort Steam-store cover for an exact title match. None when offline."""
    try:
        query = urllib.parse.urlencode({"term": title, "l": "english", "cc": "US"})
        req = urllib.request.Request(STEAM_SEARCH_URL + "?" + query, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            payload = json.loads(resp.read(700_000).decode("utf-8", "ignore"))
    except Exception:
        return None

    wanted = normalize_title(title)
    for item in payload.get("items") or []:
        if normalize_title(item.get("name", "")) != wanted:
            continue
        image_url = item.get("header_image") or item.get("tiny_image")
        if image_url:
            return _download_as_data_uri(image_url)
    return None


@st.cache_data(show_spinner=False, ttl=86_400, max_entries=300)
def cached_cover(title: str) -> str | None:
    return fetch_cover(title)


def load_covers(titles: list[str], enabled: bool) -> dict[str, str | None]:
    if not enabled or not titles:
        return {}
    covers: dict[str, str | None] = {}
    pool = ThreadPoolExecutor(max_workers=10)
    try:
        futures = {pool.submit(cached_cover, t): t for t in titles}
        done, _pending = wait(futures, timeout=4.0)
        for future in done:
            try:
                covers[futures[future]] = future.result()
            except Exception:
                covers[futures[future]] = None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return covers


# --------------------------------------------------------------------------- #
# Theming (light / dark)
# --------------------------------------------------------------------------- #


def theme_variables(light: bool) -> str:
    if light:
        return """
:root {
  --gs-bg: #f8fafc;
  --gs-surface: #ffffff;
  --gs-surface-2: #f1f5f9;
  --gs-border: #cbd5e1;
  --gs-text: #0f172a;
  --gs-muted: #64748b;
  --gs-accent: #6d28d9;
  --gs-accent-rgb: 109, 40, 217;
  --gs-accent-2: #0891b2;
  --gs-accent-2-rgb: 8, 145, 178;
  --gs-good: #059669;
  --gs-warn: #d97706;
  --gs-card-bg: #ffffff;
}
"""
    return """
:root {
  --gs-bg: #070b12;
  --gs-surface: #111827;
  --gs-surface-2: #161f2e;
  --gs-border: #243044;
  --gs-text: #e9eefb;
  --gs-muted: #9fb0c9;
  --gs-accent: #8b5cf6;
  --gs-accent-rgb: 139, 92, 246;
  --gs-accent-2: #22d3ee;
  --gs-accent-2-rgb: 34, 211, 238;
  --gs-good: #34d399;
  --gs-warn: #fbbf24;
  --gs-card-bg: linear-gradient(180deg, #141d2c 0%, #0f1520 100%);
}
"""


SHARED_CSS = """
.stApp {
  background:
    radial-gradient(1100px 560px at 12% -12%, rgba(var(--gs-accent-rgb), 0.20), transparent 62%),
    radial-gradient(900px 520px at 100% 0%, rgba(var(--gs-accent-2-rgb), 0.13), transparent 58%),
    var(--gs-bg);
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container {
  max-width: 1180px;
  padding-top: 2.2rem;
  padding-bottom: 3rem;
}
html, body, .stApp, [data-testid="stSidebar"] * {
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
  color: var(--gs-text);
}
[data-testid="stSidebar"] {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    min-width: 0 !important;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
/* Suppress Streamlit's native sidebar collapse/expand controls */
[data-testid="collapsedControl"],
button[data-testid="collapsedControl"],
[data-testid="collapsedControl"] *,
[data-testid="stSidebarCollapsedControl"],
button[data-testid="stSidebarCollapsedControl"],
.stApp [data-testid="stSidebar"] button:first-child,
.stApp [data-testid="stSidebar"] section button:first-child,
.stApp [data-testid="stSidebar"] > div > button:first-child,
button[aria-label="Collapse sidebar"],
button[aria-label="collapse sidebar"],
button[aria-label="Expand sidebar"],
button[aria-label="expand sidebar"] {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
    width: 0 !important;
    height: 0 !important;
    opacity: 0 !important;
    position: absolute !important;
    z-index: -1 !important;
}

.gs-brand {
  display: inline-flex; align-items: center; gap: 0.55rem;
  font-weight: 800; letter-spacing: 0.02em;
  color: var(--gs-text);
}
.gs-logo {
  display: grid; place-items: center; width: 30px; height: 30px;
  border-radius: 9px; font-size: 0.85rem; color: #fff;
  background: linear-gradient(135deg, var(--gs-accent), var(--gs-accent-2));
  box-shadow: 0 6px 18px rgba(var(--gs-accent-rgb), 0.35);
}
.gs-hero h1 {
  margin: 0.4rem 0 0.35rem;
  font-size: clamp(1.9rem, 4.2vw, 3rem);
  line-height: 1.1;
  letter-spacing: -0.02em;
  background: linear-gradient(92deg, var(--gs-text) 20%, var(--gs-accent) 65%, var(--gs-accent-2) 100%);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
}
.gs-hero p { margin: 0; max-width: 62ch; color: var(--gs-muted); font-size: 1rem; line-height: 1.55; }
.gs-stats { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.9rem 0 0.2rem; }
.gs-stat {
  display: inline-flex; align-items: baseline; gap: 0.4rem;
  padding: 0.45rem 0.9rem; border-radius: 10px;
  background: var(--gs-surface-2);
  border: 1px solid var(--gs-border);
  color: var(--gs-muted); font-size: 0.8rem;
}
.gs-stat b { color: var(--gs-text); font-size: 0.95rem; }
[data-testid="stMetric"] {
  background: var(--gs-surface-2);
  border: 1px solid var(--gs-border);
  padding: 0.6rem 0.9rem;
  border-radius: 10px;
}
[data-testid="stMetric"] [data-testid="stMetricValue"],
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
  color: var(--gs-text) !important;
}
[data-testid="stMetric"] [data-testid="stMetricLabel"] {
  color: var(--gs-muted) !important;
}
[data-testid="stMetricValue"] { font-size: 1.1rem !important; }
.gs-kicker {
  margin: 0.15rem 0 0.5rem; font-size: 0.72rem; font-weight: 700;
  letter-spacing: 0.12em; text-transform: uppercase; color: var(--gs-muted);
}
.gs-sec {
  display: flex; align-items: flex-end; justify-content: space-between;
  gap: 1rem; flex-wrap: wrap; margin: 0.35rem 0 0.15rem;
}
.gs-sec h2 { margin: 0; font-size: 1.35rem; color: var(--gs-text); letter-spacing: -0.01em; }
.gs-sec p { margin: 0; color: var(--gs-muted); font-size: 0.85rem; }

.gs-pills { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.4rem 0 0.2rem; }
.gs-pill {
  display: inline-flex; align-items: center; gap: 0.35rem;
  padding: 0.24rem 0.6rem; border-radius: 999px;
  font-size: 0.79rem; border: 1px solid; line-height: 1.3;
}
.gs-pill.ok {
  color: var(--gs-good);
  background: rgba(5, 150, 105, 0.10);
  border-color: rgba(5, 150, 105, 0.42);
}
.gs-pill.bad {
  color: var(--gs-warn);
  background: rgba(217, 119, 6, 0.10);
  border-color: rgba(217, 119, 6, 0.42);
}
.gs-pill.dim {
  color: var(--gs-muted);
  background: rgba(100, 116, 139, 0.10);
  border-color: rgba(100, 116, 139, 0.34);
}

.stButton > button {
  border-radius: 10px; border: 1px solid var(--gs-border);
  background: var(--gs-surface-2); color: var(--gs-text); font-weight: 600;
  transition: border-color 120ms ease, background 120ms ease;
}
.stButton > button:hover { border-color: var(--gs-accent); background: var(--gs-surface-2); color: var(--gs-accent-2); }
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--gs-accent), var(--gs-accent-2)); border-color: transparent;
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.1); }
.stButton > button:focus-visible,
[data-baseweb="select"]:focus-within,
[data-testid="stTextArea"] textarea:focus-visible {
  outline: 2px solid var(--gs-accent-2); outline-offset: 2px;
}
[data-baseweb="tag"] {
  background: var(--gs-surface-2) !important;
  border: 1px solid var(--gs-accent) !important;
}
[data-baseweb="tag"] span { color: var(--gs-text) !important; }
[data-testid="stMultiSelect"] [data-baseweb="select"] > div,
[data-testid="stTextArea"] textarea {
  background: var(--gs-surface) !important;
  border-color: var(--gs-border) !important;
  color: var(--gs-text) !important;
}
[data-testid="stExpander"] details {
  border: 1px solid var(--gs-border); border-radius: 14px;
  background: var(--gs-surface-2);
}
a { color: var(--gs-accent-2); }
a:focus-visible { outline: 2px solid var(--gs-accent-2); outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) {
  .stButton > button { transition: none; }
}
"""


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #


def _esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def render_hero(catalog: Catalog) -> None:
    st.markdown(
        f"""
<div class="gs-brand"><span class="gs-logo" aria-hidden="true">&#9654;</span>GameSense</div>
""",
        unsafe_allow_html=True,
    )
    st.title("Find your next favorite game.")
    st.markdown(
        "Tell us what you already love. A matrix-factorization model trained on real Steam "
        "play histories ranks games that fit your taste.",
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Games in catalogue", f"{len(catalog.titles):,}")
    c2.metric("Inference", "NumPy only")
    c3.metric("Dataset", "Steam play data")


def render_card(game: str, because: str, players: int, rank: int, cover_uri: str | None) -> None:
    with st.container(border=True):
        st.markdown(
            f'<div style="display:flex;justify-content:flex-end;margin:-6px -8px 8px -8px;">'
            f'<span style="padding:2px 8px;border-radius:999px;font-size:0.7rem;font-weight:700;'
            f'color:var(--gs-accent-2);background:var(--gs-surface-2);border:1px solid var(--gs-border);">#{rank}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        if cover_uri:
            st.image(cover_uri, width="stretch")
        else:
            st.markdown(
                f'<div style="height:110px;display:grid;place-items:center;'
                f'border-radius:8px;border:1px dashed var(--gs-border);background:var(--gs-surface-2);'
                f'color:var(--gs-muted);font-size:2.2rem;font-weight:800;letter-spacing:0.06em">'
                f"{_esc(game[:1].upper())}</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div style="font-weight:700;font-size:0.95rem;color:var(--gs-text);line-height:1.3;margin:0.5rem 0 0.25rem;">{game}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="margin:0 0 0.4rem 0;color:var(--gs-accent);font-size:0.8rem;'
            f'line-height:1.4">Because you liked <strong style="color:var(--gs-text);">{_esc(because)}</strong></p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.5rem;"><span style="font-size:0.75rem;color:var(--gs-muted)">👥 {players:,} players</span></div>',
            unsafe_allow_html=True,
        )


def render_pills(items: list[tuple[str, str]], label: str) -> None:
    pills = "".join(f'<span class="gs-pill {kind}">{_esc(text)}</span>' for kind, text in items)
    st.markdown(
        f'<div class="gs-pills" role="group" aria-label="{_esc(label)}">{pills}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #


def main() -> None:
    st.set_page_config(page_title="GameSense", page_icon="🎮", layout="wide", initial_sidebar_state="collapsed")

    if "theme_light" not in st.session_state:
        st.session_state["theme_light"] = False
    if "liked" not in st.session_state:
        st.session_state["liked"] = []
    if "typed" not in st.session_state:
        st.session_state["typed"] = ""
    if "covers" not in st.session_state:
        st.session_state["covers"] = True

    def _on_theme_toggle() -> None:
        st.session_state["theme_light"] = st.session_state["theme_toggle"]

    theme_css_block = theme_variables(st.session_state["theme_light"])
    st.markdown("<style>" + theme_css_block + SHARED_CSS + "</style>", unsafe_allow_html=True)

    try:
        artifacts = load_artifacts_cached()
    except Exception as exc:
        st.error(
            "GameSense could not load its model files. Keep `gamesense_artifacts.npz` and "
            f"`games.json` next to `app.py`. Details: {exc}"
        )
        return

    catalog = artifacts.catalog

    def add_game(title: str) -> None:
        liked = list(st.session_state.get("liked") or [])
        if title not in liked:
            liked.append(title)
        st.session_state["liked"] = liked

    def surprise() -> None:
        liked = list(st.session_state.get("liked") or [])
        pool = [t for t in catalog.by_popularity[:40] if t not in liked]
        if pool:
            liked.extend(random.sample(pool, k=min(3, len(pool))))
        st.session_state["liked"] = liked

    def clear_all() -> None:
        st.session_state["liked"] = []
        st.session_state["typed"] = ""

    render_hero(catalog)

    # Controls panel (replaces Streamlit's native sidebar to avoid its collapse arrow)
    with st.container(border=True):
        tc1, tc2, tc3, tc4 = st.columns([1.6, 1.4, 1.6, 1.2])
        with tc1:
            st.toggle(
                "Light mode",
                value=st.session_state["theme_light"],
                key="theme_toggle",
                on_change=_on_theme_toggle,
            )
        with tc2:
            k = st.slider("How many recommendations", min_value=3, max_value=30, value=10, step=1)
            st.caption("Ranked by the model, best first.")
        with tc3:
            st.toggle(
                "Steam cover art",
                key="covers",
                help="Best effort, fetched from the public Steam store and cached. Turn off to stay fully offline.",
            )
        with tc4:
            st.button("Clear all", on_click=clear_all, width="stretch")
    st.caption(
        "Inference: user vector = mean of liked-game embeddings, scored against every "
        "catalogue game. Games you already listed are removed."
    )

    st.markdown('<p class="gs-kicker">Step 1 — pick your games</p>', unsafe_allow_html=True)
    with st.container(border=True):
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            st.multiselect(
                "Games you love",
                options=catalog.by_popularity,
                key="liked",
                placeholder="Search 993 Steam titles…",
            )
            st.caption("Selections appear as removable chips. Most-played titles are listed first.")
        with right:
            st.text_area(
                "Or type titles, one per line",
                key="typed",
                height=132,
                placeholder="Half-Life 2\nPortal 2\nTeam Fortress 2",
            )
            st.caption("Loose matching: exact name → whole-word containment → fuzzy spelling.")

    typed_lines = [line for line in str(st.session_state.get("typed", "")).splitlines() if line.strip()]
    matched_lines, unmatched_lines = match_lines(typed_lines, catalog)

    if typed_lines:
        ok_items = [
            ("ok", f"&#10003; {m.query} &rarr; {m.title}" if normalize_title(m.query) != normalize_title(m.title or "") else f"&#10003; {m.title}")
            for m in matched_lines
            if not m.duplicate
        ]
        dup_items = [("dim", f"{m.query}: already included") for m in matched_lines if m.duplicate]
        bad_items = [("bad", f"&#10007; Not found: {m.query}") for m in unmatched_lines]
        if ok_items:
            render_pills(ok_items, "Matched typed titles")
        if bad_items:
            render_pills(bad_items, "Unmatched typed titles")
        if dup_items:
            render_pills(dup_items, "Duplicate typed titles")

    st.markdown('<p class="gs-kicker" style="margin-top:1rem;">Quick start</p>', unsafe_allow_html=True)
    quick_cols = st.columns([1, 1, 1, 1, 1, 1.35])
    for col, title in zip(quick_cols, catalog.by_popularity[:5]):
        with col:
            st.button(title, key=f"qs-{title}", on_click=add_game, args=(title,), width="stretch")
    with quick_cols[5]:
        st.button("Surprise me", key="qs-surprise", on_click=surprise, width="stretch")

    liked_titles = list(st.session_state.get("liked") or [])
    title_to_index = {t: i for i, t in enumerate(catalog.titles)}
    selected_indices: list[int] = [title_to_index[t] for t in liked_titles if t in title_to_index]
    for m in matched_lines:
        if m.index is not None and m.index not in selected_indices:
            selected_indices.append(m.index)

    overlap = [
        m.title for m in matched_lines if m.index is not None and m.title in liked_titles
    ]

    st.markdown('<div class="gs-sec"><h2>Your recommendations</h2></div>', unsafe_allow_html=True)

    if not selected_indices:
        st.info("Nothing selected yet — add a game above or tap a quick-start pick. "
                "Two or three games with a shared mood usually work best.")
    else:
        if overlap:
            st.caption("Already in your chips, so counted once: " + ", ".join(sorted(set(overlap))))
        if len(selected_indices) == 1:
            st.warning(
                "Only one game selected — picks come from that single title. Add one or two more "
                "for a sharper, more personal mix."
            )
        if unmatched_lines:
            st.warning(
                "Some typed titles were not found and were skipped: "
                + ", ".join(m.query for m in unmatched_lines)
            )

        recs = rank_recommendations(
            artifacts.H,
            artifacts.Q,
            artifacts.b,
            artifacts.alpha,
            selected_indices,
            k=k,
        )
        rows = [
            {
                "rank": r["rank"],
                "game": catalog.titles[r["game_index"]],
                "because": catalog.titles[r["because_index"]],
                "players": int(catalog.n_players[r["game_index"]]),
                "score": r["score"],
            }
            for r in recs
        ]

        st.caption(
            f"{len(rows)} picks from {len(selected_indices)} liked "
            f"{'game' if len(selected_indices) == 1 else 'games'}."
        )
        covers = load_covers([row["game"] for row in rows], st.session_state["covers"])

        # Render cards in rows of 3
        for start in range(0, len(rows), 3):
            chunk = rows[start:start + 3]
            cols = st.columns(len(chunk), gap="medium")
            for col, row in zip(cols, chunk):
                with col:
                    render_card(row["game"], row["because"], row["players"], row["rank"], covers.get(row["game"]))

        csv = pd.DataFrame(
            [
                {
                    "rank": row["rank"],
                    "game": row["game"],
                    "because_you_liked": row["because"],
                    "players": row["players"],
                    "score": round(row["score"], 6),
                }
                for row in rows
            ]
        ).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv,
            file_name="gamesense_recommendations.csv",
            mime="text/csv",
            width="content",
        )

    st.markdown("")
    report = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else ""
    with st.expander("About the model — what these picks mean, and what they cannot do"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Hit@10", "0.81", "held-out users with a real game in the top 10")
        c2.metric("Recall@10", "0.28", "share of real games recovered")
        c3.metric("NDCG@10", "0.33", "rewards correct early hits")
        st.markdown(
            """
**How it works.** Each game is represented by a 128-dimensional embedding learned from real Steam play histories.
GameSense averages the embeddings of the games you select to build a taste profile, then scores every other game
against it. The "Because you liked …" line names the selection that pulls strongest toward each recommendation.

**Offline results.** Tested on held-out users: using half of each person's library as input, at least one real game
from the other half appeared in the top 10 for about 4 in 5 users (Hit@10 = 0.81), with Recall@10 = 0.28 and
NDCG@10 = 0.33.

**Dataset.** Built on a Steam play-history dataset from around 2016 containing 993 games. Recommendations reflect
play co-occurrence within that catalogue; titles released later are not represented.
"""
        )
        if report:
            st.markdown("**Raw evaluation report**")
            st.markdown(report)

    st.markdown("")
    st.caption(
        "GameSense · Model trained offline; this app only serves predictions · "
        f"{len(catalog.titles):,} games from a Steam play dataset."
    )


if __name__ == "__main__":
    main()
