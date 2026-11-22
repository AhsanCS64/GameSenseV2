# GameSense

Personalized Steam game recommendations from your already-trained model — pure NumPy inference, no retraining, no placeholder data.

## What it does

You pick games you like (multiselect searchable by popularity, or free-text one-per-line with exact / whole-word / fuzzy matching) and get ranked recommendations. Every result shows **"Because you liked <game>"** — the liked title that pulls strongest toward that recommendation — plus the player count from the dataset.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Single-file Streamlit app: model loading, title matcher, NumPy inference, UI |
| `tests/test_app.py` | pytest suite: normalization, all three matching stages, liked-games-never-recommended, determinism |
| `gamesense_artifacts.npz` | Trained arrays `H`, `Q`, `b`, `alpha` (read-only) |
| `games.json` | `titles` + `n_players` (same row order as the artifacts) |
| `config.json` | Training settings (informational) |
| `eval_report.md` | Offline test metrics (summarized in "About the model") |
| `.streamlit/config.toml` | Dark gaming theme |
| `requirements.txt` | Pinned, tested dependencies |

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Test

```bash
python -m pytest -q
```

## Features

- Searchable multiselect of all 993 titles, most-played first; selections are removable chips
- Free-text box (one title per line): exact → whole-word containment (most-played wins) → difflib fuzzy ≥ 0.75; shows matched and not-found titles
- "How many recommendations" slider (3–30, default 10)
- Recommendations auto-refresh on every change; results as cards with rank, title, "Because you liked …", player count
- Quick-start row: top 5 most-played games + "Surprise me" (adds 3 popular games)
- Optional Steam cover art (public store search, cached, degrades to letter placeholders fully offline)
- Empty state, single-game warning, duplicate/unmatched notices, CSV download
- "About the model" drawer with plain-language metrics and the 2016 Steam-data caveat
- Light/dark theme toggle with mode-aware color coding (sidebar)
- Dark responsive theme, keyboard navigation and focus states

## Deployment (Streamlit Community Cloud)

1. Push this folder to a GitHub repo (everything is well under 100 MB).
2. Go to streamlit.io/cloud, add the app from your repo.
3. Set the entry point to `app.py`, branch `main`.
4. **Advanced settings:**
   - Main file path: `app.py`
   - Requirements file path: `requirements.txt`
   - Python version: `3.11` (default) — requirements are pinned accordingly.
   - Working directory: leave blank (repo root).
   - Environment variables / secrets: leave blank — none needed.
   - If you prefer Python 3.12+, update `requirements.txt` to `numpy==2.5.2` and `pandas==3.0.5`.
5. Deploy. The app loads `gamesense_artifacts.npz` and `games.json` from paths relative to `app.py`, so it works straight from the repo.
