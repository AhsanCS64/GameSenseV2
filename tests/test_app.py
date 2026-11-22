"""Unit tests for GameSense pure-logic functions (matching + inference)."""

from pathlib import Path

import numpy as np
import pytest

import app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def catalog():
    meta = json_load((ROOT / "games.json"))
    return app.build_catalog(meta["titles"], meta["n_players"])


def json_load(path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def artifacts():
    return app.load_artifacts_cached()


def test_normalize():
    assert app.normalize_title("Team  Fortress 2!!") == "team fortress 2"
    assert app.normalize_title("Garry's Mod") == "garry s mod"
    assert app.normalize_title("  HALF-LIFE 2  ") == "half life 2"


def test_match_exact(catalog):
    m = app.match_title("Team Fortress 2", catalog)
    assert m.stage == "exact"
    assert m.title == "Team Fortress 2"
    assert m.index is not None


def test_match_case_insensitive(catalog):
    m = app.match_title("dota 2", catalog)
    assert m.stage == "exact"
    assert m.title == "Dota 2"


def test_match_whole_word_containment_picks_most_played(catalog):
    # "Elder Scrolls" matches Morrowind/Oblivion/Skyrim/Online; Skyrim is most played.
    m = app.match_title("Elder Scrolls", catalog)
    assert m.stage == "whole-word"
    assert m.title == "The Elder Scrolls V Skyrim"


def test_match_whole_word_requires_all_words(catalog):
    # "Portal Stories" words both present only in "Portal Stories Mel".
    m = app.match_title("Portal Stories", catalog)
    assert m.stage == "whole-word"
    assert m.title == "Portal Stories Mel"


def test_match_fuzzy_typo(catalog):
    m = app.match_title("Team Fortres 2", catalog)
    assert m.stage == "fuzzy"
    assert m.title == "Team Fortress 2"


def test_match_fuzzy_below_cutoff_not_found(catalog):
    m = app.match_title("q w e r t y u i o p a s d f g", catalog)
    assert m.stage == "not_found"
    assert m.title is None


def test_match_lines_duplicates_and_unmatched(catalog):
    lines = ["dota 2", "Dota 2", "notagameasdf", "Half Life 2", ""]
    matched, unmatched = app.match_lines(lines, catalog)
    matched_titles = [m.title for m in matched]
    assert len(matched_titles) == 3  # dota 2 x2 + Half Life 2
    assert sum(1 for m in matched if m.duplicate) == 1
    assert any(m.title == "Half-Life 2" for m in matched)
    assert any(m.query == "notagameasdf" for m in unmatched)


def test_listed_games_never_recommended(artifacts):
    idx = [artifacts.catalog.titles.index("Counter-Strike Global Offensive"),
           artifacts.catalog.titles.index("Team Fortress 2"),
           artifacts.catalog.titles.index("Dota 2")]
    recs = app.rank_recommendations(artifacts.H, artifacts.Q, artifacts.b,
                                    artifacts.alpha, idx, k=25)
    recommended = {r["game_index"] for r in recs}
    assert recommended.isdisjoint(set(idx))


def test_recommendations_deterministic(artifacts):
    idx = [artifacts.catalog.titles.index("The Elder Scrolls V Skyrim"),
           artifacts.catalog.titles.index("Fallout 4"),
           artifacts.catalog.titles.index("Grand Theft Auto V")]
    a = app.rank_recommendations(artifacts.H, artifacts.Q, artifacts.b,
                                 artifacts.alpha, idx, k=10)
    b_ = app.rank_recommendations(artifacts.H, artifacts.Q, artifacts.b,
                                  artifacts.alpha, idx, k=10)
    assert a == b_


def test_results_sorted_by_score_descending(artifacts):
    idx = [artifacts.catalog.titles.index("Portal 2")]
    recs = app.rank_recommendations(artifacts.H, artifacts.Q, artifacts.b,
                                    artifacts.alpha, idx, k=20)
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_k_clamped_to_available_games(artifacts):
    n = len(artifacts.catalog.titles)
    recs = app.rank_recommendations(artifacts.H, artifacts.Q, artifacts.b,
                                    artifacts.alpha, list(range(n)), k=30)
    assert recs == []
    # Liking all-but-one leaves exactly one eligible game.
    recs = app.rank_recommendations(artifacts.H, artifacts.Q, artifacts.b,
                                    artifacts.alpha, list(range(n - 1)), k=30)
    assert len(recs) <= 1


def test_because_is_one_of_liked_games(artifacts):
    idx = [artifacts.catalog.titles.index("Portal"),
           artifacts.catalog.titles.index("Half-Life 2")]
    recs = app.rank_recommendations(artifacts.H, artifacts.Q, artifacts.b,
                                    artifacts.alpha, idx, k=10)
    liked = set(idx)
    for r in recs:
        assert r["because_index"] in liked
