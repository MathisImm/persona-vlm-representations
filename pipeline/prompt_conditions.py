"""Prompt-condition registry for the persona experiment.

Defines every framing variant (baseline, D1/D2 alternative defaults, original
persona, P1-P4 persona alternatives) that the VLM extraction scripts can
emit. The registry is the single source of truth: extract_embeddings.py and
extract_persona_text.py used to carry duplicate hardcoded templates; both
now import from here.

`original_baseline` and `original_persona` reproduce the previous hardcoded
prompts byte-for-byte so existing on-disk embeddings/ratings remain valid
against the new code (verified by tests/test_prompt_conditions.py).
"""

from dataclasses import dataclass
from typing import Callable, Optional


# Output instruction used by every condition that presents a persona
# (original_persona, P1-P4). Mirrors the trailing portion of the previous
# DEFAULT_PERSONA_TEMPLATE in extract_embeddings.py exactly. The leading line
# preserves a historical trailing space before its newline.
_OUTPUT_BLOCK_PERSONA = (
    "You are observing the provided image. Based on your details, rate how"
    " interesting you find this image by selecting only one of the following"
    " options: \n"  # trailing space is intentional and historical
    '"Not Interesting," "Slightly Interesting," "Moderately Interesting,"'
    ' "Very Interesting," or "Extremely Interesting".\n'
    "\n"
    "Provide a brief explanation in one short sentence without going into"
    " excessive detail. Present the output as a JSON object in the following"
    " format:\n"
    "{\n"
    '    "interestingness": "...",\n'
    '    "explanation": "..."\n'
    "}"
)

# Output instruction used by every condition without a persona
# (original_baseline, D1, D2). Mirrors the previous BASELINE_PROMPT exactly.
_OUTPUT_BLOCK_DEFAULT = (
    "You are observing the provided image. Rate how interesting you find this"
    " image by selecting only one of the following options: \n"  # trailing space
    '"Not Interesting," "Slightly Interesting," "Moderately Interesting,"'
    ' "Very Interesting," or "Extremely Interesting".\n'
    "\n"
    "Provide a brief explanation in one short sentence without going into"
    " excessive detail. Present the output as a JSON object in the following"
    " format:\n"
    "{\n"
    '    "interestingness": "...",\n'
    '    "explanation": "..."\n'
    "}"
)

# The "Imagine you are a person..." preamble that introduces a persona body.
# `{persona}` is filled at build time.
_PERSONA_BLOCK = """Imagine you are a person with the following specific details:

{persona}"""


@dataclass(frozen=True)
class PromptCondition:
    id: str
    requires_persona: bool
    _build: Callable[[Optional[str]], str]

    def build(self, persona: Optional[str] = None) -> str:
        if self.requires_persona and not persona:
            raise ValueError(f"Condition {self.id!r} requires a persona description")
        if not self.requires_persona and persona:
            raise ValueError(f"Condition {self.id!r} does not take a persona; got one")
        return self._build(persona)


def _build_original_baseline(_persona):
    return _OUTPUT_BLOCK_DEFAULT


def _build_d1(_persona):
    return "You are an AI.\n\n" + _OUTPUT_BLOCK_DEFAULT


def _build_d2(_persona):
    return "You are a human being.\n\n" + _OUTPUT_BLOCK_DEFAULT


def _build_original_persona(persona):
    return _PERSONA_BLOCK.format(persona=persona) + "\n\n" + _OUTPUT_BLOCK_PERSONA


def _build_p1(persona):
    return (
        "A person with the following specific details is observing the image:\n\n"
        + persona
        + "\n\n"
        + _OUTPUT_BLOCK_PERSONA
    )


def _build_p2(persona):
    return (
        _OUTPUT_BLOCK_PERSONA
        + "\n\nNote: the viewer has the following background:\n\n"
        + _PERSONA_BLOCK.format(persona=persona)
    )


def _build_p3(persona):
    return (
        _PERSONA_BLOCK.format(persona=persona)
        + "\n\nAnswer only as this persona would, not as a general AI assistant.\n\n"
        + _OUTPUT_BLOCK_PERSONA
    )


def _build_p4(persona):
    # Drop the "Imagine you are a person..." preamble: P4 isolates *no-roleplay*
    # statistical framing, so the demographics are presented as external context
    # rather than an identity to adopt. (Same rationale as P1.)
    return (
        "Research shows that people with the following profile tend to perceive "
        "images distinctively. Following that pattern:\n\n"
        + persona
        + "\n\n"
        + _OUTPUT_BLOCK_PERSONA
    )


CONDITIONS = {
    "original_baseline": PromptCondition("original_baseline", False, _build_original_baseline),
    "D1": PromptCondition("D1", False, _build_d1),
    "D2": PromptCondition("D2", False, _build_d2),
    "original_persona": PromptCondition("original_persona", True, _build_original_persona),
    "P1": PromptCondition("P1", True, _build_p1),
    "P2": PromptCondition("P2", True, _build_p2),
    "P3": PromptCondition("P3", True, _build_p3),
    "P4": PromptCondition("P4", True, _build_p4),
}

NO_PERSONA_CONDITIONS = frozenset({"original_baseline", "D1", "D2"})
PERSONA_CONDITIONS = frozenset({"original_persona", "P1", "P2", "P3", "P4"})


def get_dataset_suffix(persona_set: Optional[str], condition: str) -> str:
    """Folder name under persona_analysis/{model_folder}/ for a (persona_set, condition).

    - `original_persona` keeps the existing layout (`{persona_set}` with no suffix)
      so historical embeddings remain reachable.
    - `original_baseline` lives under `defaults_baseline` only when invoked
      explicitly without a persona-set; when it appears as the `baseline` entry
      inside a persona-set JSON, that path is unchanged from history.
    - D1, D2 always live under `defaults_{condition}`.
    - P1-P4 live under `{persona_set}_{condition}`.
    """
    if condition not in CONDITIONS:
        raise KeyError(f"Unknown condition {condition!r}")
    if condition in NO_PERSONA_CONDITIONS:
        if condition == "original_baseline":
            return "defaults_baseline"
        return f"defaults_{condition}"
    if condition == "original_persona":
        if persona_set is None:
            raise ValueError("original_persona requires a persona_set")
        return persona_set
    if persona_set is None:
        raise ValueError(f"Condition {condition!r} requires a persona_set")
    return f"{persona_set}_{condition}"


def resolve_condition(
    explicit: Optional[str],
    use_baseline_prompt: bool,
    has_persona: bool,
) -> str:
    """Pick a condition id when callers haven't specified one.

    Preserves historical CLI semantics for extract_embeddings.py:
    - --use-baseline-prompt without --prompt-condition  →  original_baseline
    - --persona <txt>          without --prompt-condition  →  original_persona
    - both unset and no condition                          →  caller error
    """
    if explicit is not None:
        return explicit
    if use_baseline_prompt:
        return "original_baseline"
    if has_persona:
        return "original_persona"
    raise ValueError(
        "Cannot infer prompt condition: pass --prompt-condition, "
        "--use-baseline-prompt, or --persona."
    )
