"""Verify the prompt-condition registry is byte-identical to the legacy templates.

If this test fails, on-disk persona embeddings/ratings produced before the
registry refactor would no longer round-trip — a silent change of input
would invalidate prior results. Keep the legacy strings frozen.
"""

import pytest

from pipeline.prompt_conditions import (
    CONDITIONS,
    NO_PERSONA_CONDITIONS,
    PERSONA_CONDITIONS,
    get_dataset_suffix,
    resolve_condition,
)


# ---- Legacy strings copied verbatim from
#      the original extract_embeddings.py (before the registry) ----
# DO NOT EDIT — these are the ground truth. The .replace() at the bottom of
# each block restores a historical trailing space before the newline that
# editors silently strip.
LEGACY_DEFAULT_PERSONA_TEMPLATE = """Imagine you are a person with the following specific details:

{persona}

You are observing the provided image. Based on your details, rate how interesting you find this image by selecting only one of the following options:
"Not Interesting," "Slightly Interesting," "Moderately Interesting," "Very Interesting," or "Extremely Interesting".

Provide a brief explanation in one short sentence without going into excessive detail. Present the output as a JSON object in the following format:
{{
    "interestingness": "...",
    "explanation": "..."
}}""".replace("following options:\n", "following options: \n")

LEGACY_BASELINE_PROMPT = """You are observing the provided image. Rate how interesting you find this image by selecting only one of the following options:
"Not Interesting," "Slightly Interesting," "Moderately Interesting," "Very Interesting," or "Extremely Interesting".

Provide a brief explanation in one short sentence without going into excessive detail. Present the output as a JSON object in the following format:
{{
    "interestingness": "...",
    "explanation": "..."
}}""".replace("following options:\n", "following options: \n")


SAMPLE_PERSONA = (
    "Origin: Germany\n"
    "Age: 30\n"
    "Sex: female\n"
    "Professional Background: software engineer\n"
    "Mental Workload: moderate\n"
    "Emotion: calm"
)


def test_original_persona_byte_identical_to_legacy():
    expected = LEGACY_DEFAULT_PERSONA_TEMPLATE.format(persona=SAMPLE_PERSONA)
    actual = CONDITIONS["original_persona"].build(SAMPLE_PERSONA)
    assert actual == expected, (
        f"original_persona drift:\n--- expected ---\n{expected!r}\n"
        f"--- actual ---\n{actual!r}"
    )


def test_original_baseline_byte_identical_to_legacy():
    expected = LEGACY_BASELINE_PROMPT.format()  # neutralizes {{...}} → {...}
    actual = CONDITIONS["original_baseline"].build()
    assert actual == expected, (
        f"original_baseline drift:\n--- expected ---\n{expected!r}\n"
        f"--- actual ---\n{actual!r}"
    )


@pytest.mark.parametrize("cid", ["D1", "D2"])
def test_default_alts_share_baseline_output_block(cid):
    cond = CONDITIONS[cid]
    out = cond.build()
    # The added preamble lives at the start; the rest must match the baseline
    # output block character-for-character.
    assert out.endswith(LEGACY_BASELINE_PROMPT.format()), (
        f"{cid} does not end with the legacy baseline output block"
    )
    assert not cond.requires_persona


@pytest.mark.parametrize("cid", ["P1", "P2", "P3", "P4"])
def test_persona_alts_require_persona(cid):
    cond = CONDITIONS[cid]
    assert cond.requires_persona
    out = cond.build(SAMPLE_PERSONA)
    # Persona text must appear somewhere in the rendered prompt.
    assert SAMPLE_PERSONA in out, f"{cid} prompt does not contain persona body"
    # The persona-version output block ("Based on your details, rate") must appear.
    assert "Based on your details, rate" in out


def test_p1_does_not_use_imagine_preamble():
    # P1 spec: "A person with the following specific details is observing the
    # image:" — explicitly NOT the "Imagine you are" ritual.
    out = CONDITIONS["P1"].build(SAMPLE_PERSONA)
    assert out.startswith(
        "A person with the following specific details is observing the image:"
    )
    assert "Imagine you are a person" not in out


def test_p2_places_persona_after_task():
    out = CONDITIONS["P2"].build(SAMPLE_PERSONA)
    out_idx = out.index("You are observing the provided image")
    persona_idx = out.index(SAMPLE_PERSONA)
    assert out_idx < persona_idx, "P2 must show the task before the persona"
    assert "Note: the viewer has the following background:" in out


def test_p3_includes_override_line():
    out = CONDITIONS["P3"].build(SAMPLE_PERSONA)
    assert (
        "Answer only as this persona would, not as a general AI assistant." in out
    )


def test_p4_uses_statistical_framing():
    out = CONDITIONS["P4"].build(SAMPLE_PERSONA)
    assert out.startswith(
        "Research shows that people with the following profile tend to perceive"
    )


# Preamble policy — locks in design intent from prompt_conditions_overview.md
# "What it isolates" notes:
#   P1: "ritual is load-bearing" → drop "Imagine you are" preamble
#   P2: "position-only test"     → keep preamble (grammatical awkwardness accepted)
#   P3: "original + override"    → keep preamble
#   P4: "no roleplay"            → drop "Imagine you are" preamble
@pytest.mark.parametrize("cid", ["P1", "P4"])
def test_no_roleplay_conditions_drop_imagine_preamble(cid):
    out = CONDITIONS[cid].build(SAMPLE_PERSONA)
    assert "Imagine you are a person" not in out, (
        f"{cid} must not contain the roleplay preamble"
    )


@pytest.mark.parametrize("cid", ["original_persona", "P2", "P3"])
def test_roleplay_conditions_keep_imagine_preamble(cid):
    out = CONDITIONS[cid].build(SAMPLE_PERSONA)
    assert "Imagine you are a person with the following specific details:" in out, (
        f"{cid} must keep the original roleplay preamble"
    )


@pytest.mark.parametrize("cid", ["original_baseline", "D1", "D2"])
def test_no_persona_conditions_reject_persona(cid):
    with pytest.raises(ValueError):
        CONDITIONS[cid].build(SAMPLE_PERSONA)


@pytest.mark.parametrize("cid", ["original_persona", "P1", "P2", "P3", "P4"])
def test_persona_conditions_require_persona(cid):
    with pytest.raises(ValueError):
        CONDITIONS[cid].build(None)


def test_condition_membership_partition():
    # Every condition is in exactly one of the two sets.
    assert NO_PERSONA_CONDITIONS | PERSONA_CONDITIONS == set(CONDITIONS)
    assert NO_PERSONA_CONDITIONS & PERSONA_CONDITIONS == set()


def test_get_dataset_suffix():
    assert get_dataset_suffix("personas_prototypical", "original_persona") == (
        "personas_prototypical"
    )
    assert get_dataset_suffix("personas_prototypical", "P1") == (
        "personas_prototypical_P1"
    )
    assert get_dataset_suffix("personas_big5_prototypical", "P3") == (
        "personas_big5_prototypical_P3"
    )
    assert get_dataset_suffix(None, "D1") == "defaults_D1"
    assert get_dataset_suffix(None, "D2") == "defaults_D2"
    assert get_dataset_suffix(None, "original_baseline") == "defaults_baseline"
    # No-persona conditions ignore persona_set when given.
    assert get_dataset_suffix("anything", "D1") == "defaults_D1"
    with pytest.raises(KeyError):
        get_dataset_suffix("personas_prototypical", "ZZZ")
    with pytest.raises(ValueError):
        get_dataset_suffix(None, "P1")


def test_resolve_condition_legacy_semantics():
    assert resolve_condition(None, use_baseline_prompt=True, has_persona=False) == (
        "original_baseline"
    )
    assert resolve_condition(None, use_baseline_prompt=False, has_persona=True) == (
        "original_persona"
    )
    assert resolve_condition("P3", use_baseline_prompt=False, has_persona=True) == "P3"
    with pytest.raises(ValueError):
        resolve_condition(None, use_baseline_prompt=False, has_persona=False)
