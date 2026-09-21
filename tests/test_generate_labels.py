"""Tests for rl/generate_labels.py -- the imitation-warm-start dataset
builder. Deliberately dependency-free (no torch, no endplay): everything
tested here is pure Python + NumPy, using `dds_leads=False` so it never
needs a real DDS install. See rl/generate_labels.py's own docstring for
the honest scope this is checking (endgame-solver labels only, in this
environment)."""
from rl.generate_labels import generate_examples
from rl.features import FEATURE_SIZE, N_CARDS


def test_generates_only_endgame_labels_when_dds_leads_disabled():
    examples, skipped = generate_examples(num_deals=6, seed=0, endgame_limit=5, dds_leads=False)
    assert len(examples) > 0
    assert all(source == "endgame" for _, _, _, source in examples)
    assert skipped >= 0


def test_examples_have_the_right_shapes_and_legal_actions():
    examples, _ = generate_examples(num_deals=4, seed=1, endgame_limit=5, dds_leads=False)
    assert examples, "expected at least one labeled decision from 4 deals"
    for x, mask, action, source in examples:
        assert x.shape == (FEATURE_SIZE,)
        assert mask.shape == (N_CARDS,)
        assert 0 <= action < N_CARDS
        # The labeled card must actually be legal in that position --
        # anything else would mean the endgame solver's answer and the
        # feature encoder's mask disagree about the position.
        assert mask[action] == 1.0
        assert source == "endgame"


def test_higher_endgame_limit_never_produces_fewer_labels():
    """Raising --endgame-limit only widens which positions the exact
    solver is *attempted* on -- it should never cause fewer decisions to
    get labeled (a regression here would mean the limit is being applied
    backwards)."""
    ex_low, _ = generate_examples(num_deals=5, seed=2, endgame_limit=3, dds_leads=False)
    ex_high, _ = generate_examples(num_deals=5, seed=2, endgame_limit=6, dds_leads=False)
    assert len(ex_high) >= len(ex_low)


def test_zero_deals_returns_empty_without_error():
    examples, skipped = generate_examples(num_deals=0, seed=0, endgame_limit=5, dds_leads=False)
    assert examples == []
    assert skipped == 0
