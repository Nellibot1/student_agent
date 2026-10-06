"""Student-owned interpretation of visible tutor messages."""

from __future__ import annotations

import hashlib
import re

from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .conversation import ConversationMessage, ConversationRole

if TYPE_CHECKING:
    from .policy import SimulatedStudentState


class MessageType(StrEnum):
    QUESTION = "question"
    FEEDBACK = "feedback"
    FEEDBACK_AND_QUESTION = "feedback_and_question"
    EXPLANATION = "explanation"
    HINT = "hint"
    INSTRUCTION = "instruction"
    OTHER = "other"


class PerceivedFeedback(StrEnum):
    CORRECT = "correct"
    PARTIAL = "partial"
    INCORRECT = "incorrect"
    NONE = "none"


class StudentObservation(BaseModel):
    """The student's private interpretation, not authoritative tutor metadata."""

    model_config = ConfigDict(extra="forbid")

    tutor_message: str = Field(min_length=1)
    perceived_topic: str = "general discussion"
    topic_key: str = "general"
    message_type: MessageType = MessageType.OTHER
    perceived_feedback: PerceivedFeedback = PerceivedFeedback.NONE
    response_expected: bool = False
    perceived_difficulty: float = Field(0.5, ge=0.0, le=1.0)
    mentioned_terms: list[str] = Field(default_factory=list)
    turn_number: int = Field(1, ge=1)
    attempt_number: int = Field(0, ge=0)
    experiment_seed: str = ""

    def visible_prompt_block(self) -> str:
        """Render only the student's own interpretation."""
        return "\n".join(
            [
                f"perceived_topic: {self.perceived_topic}",
                f"message_type: {self.message_type.value}",
                f"perceived_feedback: {self.perceived_feedback.value}",
                f"response_expected: {self.response_expected}",
            ]
        )


class StudentPerception(Protocol):
    """Interpret visible conversation without access to tutor internals."""

    async def observe(
        self,
        *,
        tutor_message: str,
        history: Sequence[ConversationMessage],
        state: SimulatedStudentState,
    ) -> StudentObservation:
        """Return the student's private interpretation of the latest message."""


class RuleBasedPerception:
    """Deterministic default perception using only visible text and private memory."""

    async def observe(
        self,
        *,
        tutor_message: str,
        history: Sequence[ConversationMessage],
        state: SimulatedStudentState,
    ) -> StudentObservation:
        message = tutor_message.strip()
        if not message:
            raise ValueError("tutor_message cannot be empty")

        feedback = _infer_feedback(message)
        response_expected = _response_expected(message)
        message_type = _infer_message_type(message, feedback, response_expected)
        terms = _content_terms(message)
        perceived_topic = _infer_topic(message, terms, state.current_topic)
        topic_key = _topic_key(perceived_topic)
        turn_number = 1 + sum(message.role is ConversationRole.TUTOR for message in history)
        previous_attempts = state.attempts_by_topic.get(topic_key, 0)
        attempt_number = previous_attempts + 1 if response_expected else previous_attempts

        return StudentObservation(
            tutor_message=message,
            perceived_topic=perceived_topic,
            topic_key=topic_key,
            message_type=message_type,
            perceived_feedback=feedback,
            response_expected=response_expected,
            perceived_difficulty=_perceived_difficulty(message),
            mentioned_terms=terms,
            turn_number=turn_number,
            attempt_number=attempt_number,
        )


_PARTIAL_PATTERNS = (
    r"\balmost\b",
    r"\bpartly\b",
    r"\bpartially\b",
    r"\bclose\b",
    r"\bnot quite\b",
    r"\bmissing\b",
    r"\bon the right track\b",
)
_INCORRECT_PATTERNS = (
    r"\bincorrect\b",
    r"\bwrong\b",
    r"\bnot correct\b",
    r"\bthat is not right\b",
    r"\bconfus(?:e|es|ed|ing)\b",
)
_CORRECT_PATTERNS = (
    r"\bcorrect\b",
    r"\bexactly\b",
    r"\bthat(?:'s| is) right\b",
    r"\bwell done\b",
    r"\bgood job\b",
)
_RESPONSE_PATTERNS = (
    r"\?",
    r"\bcan you\b",
    r"\bcould you\b",
    r"\bplease\b",
    r"\btry again\b",
    r"\bexplain\b",
    r"\bcalculate\b",
    r"\bdescribe\b",
    r"\btell me\b",
    r"\bwhat do you\b",
    r"\bwhy do(?:es| you)\b",
    r"\bhow do(?:es| you)\b",
)
_HINT_PATTERNS = (r"\bhint\b", r"\bremember that\b", r"\blook at\b", r"\bthink about\b")
_EXPLANATION_PATTERNS = (r"\bbecause\b", r"\bmeans that\b", r"\bin other words\b", r"\bnotice that\b")
_TOPIC_TRANSITION_PATTERNS = (
    r"\bnew topic\b",
    r"\bmoving on to\b",
    r"\blet(?:'s| us) discuss\b",
    r"\bnow (?:consider|about)\b",
)
_STOP_WORDS = {
    "about",
    "again",
    "also",
    "almost",
    "and",
    "answer",
    "are",
    "because",
    "been",
    "being",
    "calculate",
    "can",
    "correct",
    "could",
    "describe",
    "does",
    "explain",
    "for",
    "from",
    "had",
    "has",
    "have",
    "into",
    "just",
    "missed",
    "might",
    "must",
    "please",
    "remember",
    "solve",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "think",
    "this",
    "try",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "you",
    "your",
}


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _infer_feedback(message: str) -> PerceivedFeedback:
    if _matches_any(message, _PARTIAL_PATTERNS):
        return PerceivedFeedback.PARTIAL
    if _matches_any(message, _INCORRECT_PATTERNS):
        return PerceivedFeedback.INCORRECT
    if _matches_any(message, _CORRECT_PATTERNS):
        return PerceivedFeedback.CORRECT
    return PerceivedFeedback.NONE


def _response_expected(message: str) -> bool:
    return _matches_any(message, _RESPONSE_PATTERNS)


def _infer_message_type(
    message: str,
    feedback: PerceivedFeedback,
    response_expected: bool,
) -> MessageType:
    if feedback is not PerceivedFeedback.NONE and response_expected:
        return MessageType.FEEDBACK_AND_QUESTION
    if feedback is not PerceivedFeedback.NONE:
        return MessageType.FEEDBACK
    if _matches_any(message, _HINT_PATTERNS):
        return MessageType.HINT
    if response_expected:
        return MessageType.QUESTION
    if _matches_any(message, _EXPLANATION_PATTERNS):
        return MessageType.EXPLANATION
    if re.search(r"\b(?:read|write|compare|use|consider|continue)\b", message, flags=re.IGNORECASE):
        return MessageType.INSTRUCTION
    return MessageType.OTHER


def _content_terms(message: str) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", message.lower()):
        if token in _STOP_WORDS or token in terms:
            continue
        terms.append(token)
        if len(terms) == 8:
            break
    return terms


def _infer_topic(message: str, terms: list[str], current_topic: str) -> str:
    transition = _matches_any(message, _TOPIC_TRANSITION_PATTERNS)
    if current_topic and not transition:
        return current_topic
    if terms:
        return " ".join(terms[:4])
    return current_topic or "general discussion"


def _topic_key(topic: str) -> str:
    normalized = " ".join(topic.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def _perceived_difficulty(message: str) -> float:
    words = message.split()
    long_words = sum(len(word.strip(".,?!:;()[]{}")) >= 10 for word in words)
    notation = sum(character in "=∫∑√^<>[]{}" for character in message)
    return min(1.0, 0.20 + len(words) / 120.0 + long_words * 0.025 + notation * 0.025)
