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
  --gs-bg: #f8f9f3;
  --gs-surface: #ffffff;
  --gs-surface-2: #edf3e8;
  --gs-surface-3: #e2ecd9;
  --gs-border: #c2cfbd;
  --gs-border-hover: #10b981;
  --gs-text: #092d20;
  --gs-muted: #516a57;
  --gs-accent: #0f6645;
  --gs-accent-rgb: 15, 102, 69;
  --gs-accent-2: #10b981;
  --gs-accent-2-rgb: 16, 185, 129;
  --gs-accent-3: #0284c7;
  --gs-accent-3-rgb: 2, 132, 199;
  --gs-accent-4: #f59e0b;
  --gs-good: #059669;
  --gs-warn: #d97706;
  --gs-card-bg: #ffffff;
  --gs-card-hover: #fcfdfa;
  --gs-glow: rgba(16, 185, 129, 0.22);
}
"""
    return """
:root {
  --gs-bg: #061510;
  --gs-surface: #0a2118;
  --gs-surface-2: #0e2d21;
  --gs-surface-3: #13392a;
  --gs-border: #1a4233;
  --gs-border-hover: #00f5a0;
  --gs-text: #f2f7f4;
  --gs-muted: #9eb58e;
  --gs-accent: #10b981;
  --gs-accent-rgb: 16, 185, 129;
  --gs-accent-2: #00f5a0;
  --gs-accent-2-rgb: 0, 245, 160;
  --gs-accent-3: #38bdf8;
  --gs-accent-3-rgb: 56, 189, 248;
  --gs-accent-4: #fbbf24;
  --gs-accent-5: #a855f7;
  --gs-good: #34d399;
  --gs-warn: #fbbf24;
  --gs-card-bg: linear-gradient(150deg, #0e2b20 0%, #081a13 100%);
  --gs-card-hover: linear-gradient(150deg, #13382b 0%, #0a2219 100%);
  --gs-glow: rgba(0, 245, 160, 0.28);
}
"""


SHARED_CSS = """
@keyframes gsAurora {
  0% {
    background-position: 0% 0%, 100% 0%, 50% 100%;
  }
  50% {
    background-position: 8% 4%, 92% 6%, 48% 95%;
  }
  100% {
    background-position: 0% 0%, 100% 0%, 50% 100%;
  }
}

@keyframes gsGradientShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

@keyframes gsPulseGlow {
  0%, 100% {
    box-shadow: 0 0 15px rgba(var(--gs-accent-2-rgb), 0.35), 0 0 30px rgba(var(--gs-accent-rgb), 0.2);
    transform: translateY(0);
  }
  50% {
    box-shadow: 0 0 25px rgba(var(--gs-accent-2-rgb), 0.6), 0 0 45px rgba(var(--gs-accent-rgb), 0.35);
    transform: translateY(-2px);
  }
}

@keyframes gsRadarPing {
  0% {
    transform: scale(0.9);
    opacity: 0.9;
  }
  75% {
    transform: scale(2.3);
    opacity: 0;
  }
  100% {
    transform: scale(2.5);
    opacity: 0;
  }
}

@keyframes gsCardRise {
  from {
    opacity: 0;
    transform: translateY(14px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes gsBorderFlow {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

.stApp {
  background:
    radial-gradient(1100px 580px at 12% -10%, rgba(var(--gs-accent-rgb), 0.28), transparent 64%),
    radial-gradient(900px 520px at 98% 5%, rgba(var(--gs-accent-2-rgb), 0.20), transparent 58%),
    radial-gradient(850px 480px at 50% 100%, rgba(56, 189, 248, 0.12), transparent 60%),
    var(--gs-bg);
  background-attachment: fixed;
  animation: gsAurora 22s ease-in-out infinite alternate;
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container {
  max-width: 1200px;
  padding-top: 1.6rem;
  padding-bottom: 3.5rem;
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

/* Top bar with links */
.gs-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
  padding: 0.75rem 1rem;
  margin-bottom: 1.25rem;
  border-radius: 14px;
  background: rgba(14, 45, 33, 0.55);
  border: 1px solid var(--gs-border);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}
.gs-brand-wrap {
  display: inline-flex;
  align-items: center;
  gap: 0.7rem;
}
.gs-brand {
  display: inline-flex;
  align-items: center;
  gap: 0.6rem;
  font-weight: 800;
  font-size: 1.15rem;
  letter-spacing: 0.02em;
  color: var(--gs-text);
}
.gs-logo {
  display: grid;
  place-items: center;
  width: 32px;
  height: 32px;
  border-radius: 10px;
  font-size: 0.9rem;
  color: #061510;
  font-weight: 900;
  background: linear-gradient(135deg, var(--gs-accent-2) 0%, var(--gs-accent) 60%, var(--gs-accent-3) 100%);
  animation: gsPulseGlow 3.5s ease-in-out infinite;
}
.gs-badge-version {
  font-size: 0.7rem;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 999px;
  color: var(--gs-accent-2);
  background: rgba(0, 245, 160, 0.12);
  border: 1px solid rgba(0, 245, 160, 0.35);
  letter-spacing: 0.05em;
}
.gs-status-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.73rem;
  font-weight: 600;
  color: var(--gs-muted);
  padding: 2px 9px;
  border-radius: 999px;
  background: rgba(16, 185, 129, 0.08);
  border: 1px solid rgba(16, 185, 129, 0.25);
}
.gs-status-dot-wrap {
  position: relative;
  width: 8px;
  height: 8px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}
.gs-status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--gs-accent-2);
}
.gs-status-dot-pulse {
  position: absolute;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--gs-accent-2);
  animation: gsRadarPing 2s cubic-bezier(0, 0, 0.2, 1) infinite;
}

.gs-nav-links {
  display: inline-flex;
  align-items: center;
  gap: 0.65rem;
  flex-wrap: wrap;
}
.gs-nav-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.42rem 0.85rem;
  border-radius: 10px;
  font-size: 0.8rem;
  font-weight: 600;
  text-decoration: none;
  transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1);
  border: 1px solid var(--gs-border);
  background: var(--gs-surface-2);
  color: var(--gs-text) !important;
}
.gs-nav-btn:hover {
  border-color: var(--gs-accent-2);
  color: #ffffff !important;
  transform: translateY(-2px);
  box-shadow: 0 4px 14px rgba(0, 245, 160, 0.3);
}
.gs-nav-btn.gs-nav-github:hover {
  background: linear-gradient(135deg, rgba(16, 185, 129, 0.25), rgba(56, 189, 248, 0.25));
  border-color: var(--gs-accent-2);
}
.gs-nav-btn.gs-nav-portfolio {
  border-color: rgba(0, 245, 160, 0.4);
  background: linear-gradient(135deg, rgba(0, 245, 160, 0.12), rgba(16, 185, 129, 0.08));
}
.gs-nav-btn.gs-nav-portfolio:hover {
  background: linear-gradient(135deg, rgba(0, 245, 160, 0.3), rgba(16, 185, 129, 0.25));
  border-color: var(--gs-accent-2);
  box-shadow: 0 4px 16px rgba(0, 245, 160, 0.35);
}
.gs-nav-icon {
  width: 15px;
  height: 15px;
  flex-shrink: 0;
}
.gs-btn-arrow {
  font-size: 0.85rem;
  opacity: 0.7;
  transition: transform 0.2s ease;
}
.gs-nav-btn:hover .gs-btn-arrow {
  transform: translate(2px, -2px);
  opacity: 1;
}

/* Hero Section */
.gs-hero-wrap {
  margin: 0.5rem 0 1.25rem;
}
.gs-hero-title {
  margin: 0.3rem 0 0.5rem;
  font-size: clamp(2rem, 4.5vw, 3.25rem);
  line-height: 1.15;
  letter-spacing: -0.025em;
  font-weight: 800;
  color: var(--gs-text);
}
.gs-gradient-word {
  background: linear-gradient(90deg, #f2f7f4 0%, var(--gs-accent-2) 28%, var(--gs-accent-3) 65%, #a855f7 90%, var(--gs-accent-2) 100%);
  background-size: 200% auto;
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: gsGradientShift 7s linear infinite;
}
.gs-hero-subtitle {
  margin: 0 0 1rem;
  max-width: 68ch;
  color: var(--gs-muted);
  font-size: 1.05rem;
  line-height: 1.6;
}
.gs-highlight-tag {
  color: var(--gs-accent-2);
  font-weight: 600;
  text-decoration: underline;
  text-decoration-color: rgba(0, 245, 160, 0.4);
}

/* Stats / Metrics */
[data-testid="stMetric"] {
  background: var(--gs-surface-2) !important;
  border: 1px solid var(--gs-border) !important;
  padding: 0.7rem 1rem !important;
  border-radius: 12px !important;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15) !important;
  transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
[data-testid="stMetric"]:hover {
  transform: translateY(-3px);
  border-color: var(--gs-accent-2) !important;
  box-shadow: 0 8px 24px rgba(0, 245, 160, 0.2) !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: var(--gs-accent-2) !important;
  font-size: 1.35rem !important;
  font-weight: 800 !important;
}
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
  color: var(--gs-text) !important;
}
[data-testid="stMetric"] [data-testid="stMetricLabel"] {
  color: var(--gs-muted) !important;
  font-weight: 600 !important;
  font-size: 0.85rem !important;
}

.gs-kicker {
  margin: 0.2rem 0 0.5rem;
  font-size: 0.74rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--gs-accent-2);
  display: flex;
  align-items: center;
  gap: 0.4rem;
}
.gs-kicker::before {
  content: "";
  display: inline-block;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--gs-accent-2);
  box-shadow: 0 0 8px var(--gs-accent-2);
}
.gs-sec {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
  margin: 0.8rem 0 0.3rem;
}
.gs-sec h2 {
  margin: 0;
  font-size: 1.45rem;
  color: var(--gs-text);
  letter-spacing: -0.015em;
  font-weight: 800;
}
.gs-sec p {
  margin: 0;
  color: var(--gs-muted);
  font-size: 0.88rem;
}

/* Pills */
.gs-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin: 0.5rem 0 0.3rem;
}
.gs-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.3rem 0.75rem;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 600;
  border: 1px solid;
  line-height: 1.3;
  transition: all 0.2s ease;
}
.gs-pill:hover {
  transform: translateY(-2px);
}
.gs-pill.ok {
  color: #34d399;
  background: rgba(16, 185, 129, 0.12);
  border-color: rgba(52, 211, 153, 0.45);
  box-shadow: 0 2px 8px rgba(16, 185, 129, 0.15);
}
.gs-pill.bad {
  color: #fbbf24;
  background: rgba(245, 158, 11, 0.12);
  border-color: rgba(251, 191, 36, 0.45);
}
.gs-pill.dim {
  color: var(--gs-muted);
  background: rgba(158, 181, 142, 0.10);
  border-color: rgba(158, 181, 142, 0.32);
}

/* Buttons */
.stButton > button {
  border-radius: 11px;
  border: 1px solid var(--gs-border);
  background: var(--gs-surface-2);
  color: var(--gs-text);
  font-weight: 600;
  padding: 0.5rem 1rem;
  transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.18);
  position: relative;
  overflow: hidden;
}
.stButton > button:hover {
  border-color: var(--gs-accent-2);
  background: var(--gs-surface-3);
  color: var(--gs-accent-2);
  transform: translateY(-2px);
  box-shadow: 0 6px 18px rgba(0, 245, 160, 0.22);
}
.stButton > button:active {
  transform: scale(0.98);
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #059669 0%, #10b981 50%, #00f5a0 100%) !important;
  color: #061510 !important;
  font-weight: 800 !important;
  border: none !important;
  box-shadow: 0 4px 18px rgba(0, 245, 160, 0.35) !important;
}
.stButton > button[kind="primary"]:hover {
  filter: brightness(1.12);
  box-shadow: 0 8px 28px rgba(0, 245, 160, 0.5) !important;
  transform: translateY(-2px) scale(1.015);
}
.stButton > button:focus-visible,
[data-baseweb="select"]:focus-within,
[data-testid="stTextArea"] textarea:focus-visible {
  outline: 2px solid var(--gs-accent-2);
  outline-offset: 2px;
}
[data-baseweb="tag"] {
  background: var(--gs-surface-2) !important;
  border: 1px solid var(--gs-accent-2) !important;
  border-radius: 8px !important;
}
[data-baseweb="tag"] span {
  color: var(--gs-text) !important;
  font-weight: 600 !important;
}
[data-testid="stMultiSelect"] [data-baseweb="select"] > div,
[data-testid="stTextArea"] textarea {
  background: var(--gs-surface) !important;
  border-color: var(--gs-border) !important;
  color: var(--gs-text) !important;
  border-radius: 11px !important;
}
[data-testid="stMultiSelect"] [data-baseweb="select"] > div:hover,
[data-testid="stTextArea"] textarea:hover {
  border-color: var(--gs-border-hover) !important;
}
[data-testid="stExpander"] details {
  border: 1px solid var(--gs-border);
  border-radius: 14px;
  background: var(--gs-surface-2);
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
}
[data-testid="stExpander"] summary:hover {
  color: var(--gs-accent-2) !important;
}

/* Glassmorphism Containers */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(10, 33, 24, 0.5) !important;
  backdrop-filter: blur(12px) !important;
  -webkit-backdrop-filter: blur(12px) !important;
  border: 1px solid var(--gs-border) !important;
  border-radius: 16px !important;
  box-shadow: 0 6px 24px rgba(0, 0, 0, 0.2) !important;
  transition: border-color 0.25s ease, box-shadow 0.25s ease;
}

/* Modern Recommendation Card */
.gs-card-container {
  background: var(--gs-card-bg);
  border: 1px solid var(--gs-border);
  border-radius: 14px;
  padding: 0.95rem;
  margin-bottom: 0.85rem;
  transition: all 0.32s cubic-bezier(0.16, 1, 0.3, 1);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
  position: relative;
  overflow: hidden;
  animation: gsCardRise 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
}
.gs-card-container:hover {
  transform: translateY(-7px) scale(1.015);
  border-color: var(--gs-accent-2);
  background: var(--gs-card-hover);
  box-shadow: 0 20px 38px -10px rgba(0, 245, 160, 0.32), 0 0 20px rgba(16, 185, 129, 0.18);
}
.gs-card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.65rem;
}
.gs-rank-pill {
  padding: 2px 9px;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.04em;
  color: #061510;
  background: linear-gradient(135deg, var(--gs-accent-2) 0%, var(--gs-accent) 100%);
  box-shadow: 0 2px 8px rgba(0, 245, 160, 0.4);
}
.gs-player-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--gs-muted);
  background: rgba(14, 45, 33, 0.6);
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid var(--gs-border);
}
.gs-live-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--gs-accent-2);
  box-shadow: 0 0 6px var(--gs-accent-2);
}
.gs-cover-box {
  width: 100%;
  height: 125px;
  border-radius: 10px;
  overflow: hidden;
  margin-bottom: 0.65rem;
  background: var(--gs-surface-2);
  border: 1px solid rgba(194, 216, 181, 0.1);
  position: relative;
}
.gs-cover-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.45s cubic-bezier(0.16, 1, 0.3, 1);
}
.gs-card-container:hover .gs-cover-img {
  transform: scale(1.08);
}
.gs-cover-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  border: 1px dashed var(--gs-border);
}
.gs-placeholder-letter {
  font-size: 2.6rem;
  font-weight: 900;
  color: var(--gs-muted);
  opacity: 0.55;
}
.gs-placeholder-icon {
  position: absolute;
  bottom: 8px;
  right: 10px;
  font-size: 1.1rem;
  opacity: 0.7;
}
.gs-card-title {
  margin: 0 0 0.35rem;
  font-size: 0.98rem;
  font-weight: 700;
  line-height: 1.35;
  color: var(--gs-text);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  min-height: 2.65rem;
}
.gs-card-because {
  margin: 0.35rem 0 0;
  padding: 0.4rem 0.55rem;
  border-radius: 8px;
  background: rgba(14, 45, 33, 0.5);
  border: 1px solid rgba(194, 216, 181, 0.15);
  font-size: 0.76rem;
  line-height: 1.4;
}
.gs-because-tag {
  display: block;
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--gs-accent-2);
  font-weight: 700;
  margin-bottom: 2px;
}
.gs-because-name {
  color: var(--gs-text);
  font-weight: 600;
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* Footer Signature Banner */
.gs-footer-card {
  margin-top: 2.5rem;
  padding: 1.5rem;
  border-radius: 18px;
  background: radial-gradient(at 100% 0%, rgba(16, 185, 129, 0.15), transparent 70%),
              linear-gradient(135deg, rgba(14, 45, 33, 0.85) 0%, rgba(6, 21, 16, 0.95) 100%);
  border: 1px solid rgba(0, 245, 160, 0.35);
  box-shadow: 0 12px 36px rgba(0, 0, 0, 0.4), 0 0 25px rgba(0, 245, 160, 0.15);
  display: flex;
  flex-direction: column;
  gap: 1.1rem;
}
.gs-footer-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
}
.gs-footer-profile {
  display: flex;
  align-items: center;
  gap: 0.9rem;
}
.gs-footer-avatar {
  width: 46px;
  height: 46px;
  border-radius: 12px;
  background: linear-gradient(135deg, #059669 0%, #00f5a0 50%, #38bdf8 100%);
  color: #061510;
  font-weight: 900;
  font-size: 1.1rem;
  display: grid;
  place-items: center;
  box-shadow: 0 0 18px rgba(0, 245, 160, 0.45);
  animation: gsPulseGlow 4s ease-in-out infinite;
}
.gs-footer-name {
  margin: 0;
  font-size: 1.15rem;
  font-weight: 800;
  color: var(--gs-text);
}
.gs-footer-sub {
  margin: 0;
  font-size: 0.82rem;
  color: var(--gs-muted);
}
.gs-footer-actions {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-wrap: wrap;
}
.gs-footer-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.55rem 1.05rem;
  border-radius: 10px;
  font-size: 0.85rem;
  font-weight: 700;
  text-decoration: none;
  transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
  border: 1px solid;
}
.gs-btn-portfolio-footer {
  color: #061510 !important;
  background: linear-gradient(135deg, #00f5a0 0%, #10b981 100%);
  border-color: #00f5a0;
  box-shadow: 0 4px 16px rgba(0, 245, 160, 0.35);
}
.gs-btn-portfolio-footer:hover {
  filter: brightness(1.12);
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0, 245, 160, 0.5);
}
.gs-btn-github-footer {
  color: var(--gs-text) !important;
  background: rgba(14, 45, 33, 0.7);
  border-color: var(--gs-border);
}
.gs-btn-github-footer:hover {
  border-color: var(--gs-accent-2);
  background: rgba(0, 245, 160, 0.15);
  transform: translateY(-2px);
  box-shadow: 0 6px 18px rgba(0, 245, 160, 0.25);
}
.gs-footer-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  border-top: 1px solid rgba(194, 216, 181, 0.15);
  padding-top: 0.85rem;
}
.gs-tech-pill {
  font-size: 0.74rem;
  font-weight: 600;
  color: var(--gs-muted);
  background: rgba(14, 45, 33, 0.5);
  border: 1px solid rgba(194, 216, 181, 0.18);
  border-radius: 6px;
  padding: 2px 8px;
}

a { color: var(--gs-accent-2); text-decoration: none; }
a:hover { text-decoration: underline; }
a:focus-visible { outline: 2px solid var(--gs-accent-2); outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) {
  .stApp { animation: none; }
  .gs-gradient-word { animation: none; }
  .gs-logo, .gs-footer-avatar { animation: none; }
  .gs-status-dot-pulse { animation: none; }
  .gs-card-container { animation: none; }
  .stButton > button { transition: none; }
}
"""


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #


def _esc(text: str) -> str:
    return html.escape(str(text), quote=True)


GITHUB_REPO_URL = "https://github.com/AhsanCS64/GameSenseV2"
PORTFOLIO_URL = "https://ahsanhabibkushal.vercel.app/"


def render_hero(catalog: Catalog) -> None:
    st.markdown(
        f"""
<div class="gs-topbar">
  <div class="gs-brand-wrap">
    <div class="gs-brand">
      <span class="gs-logo" aria-hidden="true">&#9654;</span>
      <span>GameSense</span>
    </div>
    <span class="gs-badge-version">v2.0</span>
    <span class="gs-status-badge">
      <span class="gs-status-dot-wrap">
        <span class="gs-status-dot-pulse"></span>
        <span class="gs-status-dot"></span>
      </span>
      <span>Neural Ready</span>
    </span>
  </div>
  <div class="gs-nav-links">
    <a href="{GITHUB_REPO_URL}" target="_blank" rel="noopener noreferrer" class="gs-nav-btn gs-nav-github" title="View Source on GitHub">
      <svg class="gs-nav-icon" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
      </svg>
      <span>GitHub Repo</span>
      <span class="gs-btn-arrow">&nearr;</span>
    </a>
    <a href="{PORTFOLIO_URL}" target="_blank" rel="noopener noreferrer" class="gs-nav-btn gs-nav-portfolio" title="Visit Ahsan Habib's Portfolio">
      <svg class="gs-nav-icon" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm7.93 9h-3.18a15.65 15.65 0 0 0-1.38-4.56A8.02 8.02 0 0 1 19.93 11zm-5.95-6.79A13.8 13.8 0 0 1 15.6 11h-3.6V4.08a7.84 7.84 0 0 1 1.98.13zM10 4.08V11H6.4a13.8 13.8 0 0 1 1.62-6.79 7.84 7.84 0 0 1 1.98-.13zM4.07 13h3.18a15.65 15.65 0 0 0 1.38 4.56A8.02 8.02 0 0 1 4.07 13zm1.86-2a8.02 8.02 0 0 1 2.68-4.56A15.65 15.65 0 0 0 7.23 11zM10 19.92V13h3.6v6.92a7.84 7.84 0 0 1-1.98.13 7.84 7.84 0 0 1-1.62-.13zm4-13.13A13.8 13.8 0 0 1 15.6 13h-1.6v-6.21zm1.37 10.77A15.65 15.65 0 0 0 16.75 13h3.18a8.02 8.02 0 0 1-4.56 4.56z"/>
      </svg>
      <span>H. M. Ahsan Habib Kushal</span>
      <span class="gs-btn-arrow">&nearr;</span>
    </a>
  </div>
</div>
<div class="gs-hero-wrap">
  <h1 class="gs-hero-title">Find your next <span class="gs-gradient-word">favorite game</span>.</h1>
  <p class="gs-hero-subtitle">
    Tell us what you already love. A matrix-factorization model trained on real
    Steam play histories computes your taste embedding to rank games curated for you.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Games in catalogue", f"{len(catalog.titles):,}")
    c2.metric("Inference", "NumPy Only")
    c3.metric("Dataset", "Steam Play Data")


def render_card(game: str, because: str, players: int, rank: int, cover_uri: str | None) -> None:
    cover_html = (
        f'<div class="gs-cover-box"><img src="{cover_uri}" alt="{_esc(game)}" class="gs-cover-img" loading="lazy" /></div>'
        if cover_uri
        else (
            f'<div class="gs-cover-box gs-cover-placeholder">'
            f'<div class="gs-placeholder-letter">{_esc(game[:1].upper())}</div>'
            f'<span class="gs-placeholder-icon">🎮</span>'
            f'</div>'
        )
    )
    st.markdown(
        f"""
<div class="gs-card-container">
  <div class="gs-card-header">
    <span class="gs-rank-pill">#{rank}</span>
    <span class="gs-player-badge">
      <span class="gs-live-dot"></span>
      {players:,} players
    </span>
  </div>
  {cover_html}
  <div class="gs-card-body">
    <h3 class="gs-card-title">{_esc(game)}</h3>
    <div class="gs-card-because">
      <span class="gs-because-tag">Because you liked</span>
      <strong class="gs-because-name">{_esc(because)}</strong>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_footer(catalog: Catalog) -> None:
    st.markdown(
        f"""
<div class="gs-footer-card">
  <div class="gs-footer-top">
    <div class="gs-footer-profile">
      <div class="gs-footer-avatar">AH</div>
      <div>
        <h4 class="gs-footer-name">H. M. Ahsan Habib Kushal</h4>
        <p class="gs-footer-sub">Software Engineering & Machine Learning • Creator of GameSense</p>
      </div>
    </div>
    <div class="gs-footer-actions">
      <a href="{PORTFOLIO_URL}" target="_blank" rel="noopener noreferrer" class="gs-footer-btn gs-btn-portfolio-footer">
        <svg class="gs-nav-icon" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm7.93 9h-3.18a15.65 15.65 0 0 0-1.38-4.56A8.02 8.02 0 0 1 19.93 11zm-5.95-6.79A13.8 13.8 0 0 1 15.6 11h-3.6V4.08a7.84 7.84 0 0 1 1.98.13zM10 4.08V11H6.4a13.8 13.8 0 0 1 1.62-6.79 7.84 7.84 0 0 1 1.98-.13zM4.07 13h3.18a15.65 15.65 0 0 0 1.38 4.56A8.02 8.02 0 0 1 4.07 13zm1.86-2a8.02 8.02 0 0 1 2.68-4.56A15.65 15.65 0 0 0 7.23 11zM10 19.92V13h3.6v6.92a7.84 7.84 0 0 1-1.98.13 7.84 7.84 0 0 1-1.62-.13zm4-13.13A13.8 13.8 0 0 1 15.6 13h-1.6v-6.21zm1.37 10.77A15.65 15.65 0 0 0 16.75 13h3.18a8.02 8.02 0 0 1-4.56 4.56z"/>
        </svg>
        <span>Portfolio Website</span>
        <span class="gs-btn-arrow">&nearr;</span>
      </a>
      <a href="{GITHUB_REPO_URL}" target="_blank" rel="noopener noreferrer" class="gs-footer-btn gs-btn-github-footer">
        <svg class="gs-nav-icon" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
        </svg>
        <span>GitHub Repository</span>
        <span class="gs-btn-arrow">&nearr;</span>
      </a>
    </div>
  </div>
  <div class="gs-footer-badges">
    <span class="gs-tech-pill">🎮 Steam Matrix Factorization</span>
    <span class="gs-tech-pill">⚡ Pure NumPy Inference</span>
    <span class="gs-tech-pill">🌿 Green & Colorful Aesthetic</span>
    <span class="gs-tech-pill">📊 {len(catalog.titles):,} Games Catalog</span>
    <span class="gs-tech-pill">🚀 Streamlit App</span>
  </div>
</div>
""",
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
    render_footer(catalog)

    st.markdown("")
    st.caption(
        "GameSense · Model trained offline; this app only serves predictions · "
        f"{len(catalog.titles):,} games from a Steam play dataset."
    )


if __name__ == "__main__":
    main()
