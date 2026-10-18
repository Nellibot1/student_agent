"""Offline replay evaluation for student-visible semantic perception.

Evaluation exercises only the perception boundary: labeled visible messages are replayed
and compared with structured observations. It does not generate student replies or expose
hidden tutor metadata to the perception adapter.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .conversation import ConversationMessage
from .perception import PerceivedFeedback, StudentObservation, StudentPerception
from .personas import DEFAULT_PERSONA
from .policy import SimulatedStudentState, evolve_simulated_state, initial_simulated_state


class ExpectedObservation(BaseModel):
    """Optional human labels for one visible tutor message."""

    model_config = ConfigDict(extra="forbid")

    feedback: PerceivedFeedback | None = None
    response_expected: bool | None = None
    correction_present: bool | None = None
    explanation_present: bool | None = None
    topic_group: str = ""


class PerceptionEvaluationCase(BaseModel):
    """One replayable visible-message case."""

    model_config = ConfigDict(extra="forbid")

    id: str
    conversation_id: str = "default"
    tutor_message: str = Field(min_length=1)
    history: list[ConversationMessage] = Field(default_factory=list)
    expected: ExpectedObservation = Field(default_factory=ExpectedObservation)


class PerceptionCaseResult(BaseModel):
    """The produced observation and any comparisons for one evaluation case."""

    model_config = ConfigDict(extra="forbid")

    id: str
    conversation_id: str
    observation: StudentObservation
    matches: dict[str, bool] = Field(default_factory=dict)


class PerceptionEvaluationReport(BaseModel):
    """Aggregate metrics plus per-case details for debugging mistakes."""

    model_config = ConfigDict(extra="forbid")

    case_count: int
    labeled_feedback_count: int
    feedback_accuracy: float | None = None
    response_expected_accuracy: float | None = None
    correction_accuracy: float | None = None
    explanation_accuracy: float | None = None
    confidence_brier_score: float | None = None
    topic_stability: float | None = None
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    cases: list[PerceptionCaseResult]


async def evaluate_perception(
    perception: StudentPerception,
    cases: list[PerceptionEvaluationCase],
) -> PerceptionEvaluationReport:
    """Replay labeled visible messages without tutor metadata entering perception."""
    # Each conversation ID receives independent learner memory. Cases with the same ID are
    # replayed in file order so later messages can reuse topics, tasks, and misconceptions.
    states: dict[str, SimulatedStudentState] = {}
    results: list[PerceptionCaseResult] = []
    field_scores: dict[str, list[bool]] = defaultdict(list)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    confidence_errors: list[float] = []
    topic_assignments: dict[tuple[str, str], list[str]] = defaultdict(list)

    for case in cases:
        state = states.setdefault(case.conversation_id, initial_simulated_state(DEFAULT_PERSONA))
        observation = await perception.observe(
            tutor_message=case.tutor_message,
            history=tuple(case.history),
            state=state.model_copy(deep=True),
        )
        matches: dict[str, bool] = {}
        expected = case.expected
        if expected.feedback is not None:
            correct = observation.perceived_feedback is expected.feedback
            matches["feedback"] = correct
            field_scores["feedback"].append(correct)
            confusion[expected.feedback.value][observation.perceived_feedback.value] += 1
            # The Brier score checks calibration: high confidence is rewarded only when
            # the categorical feedback label is actually correct.
            confidence_errors.append((observation.feedback_confidence - float(correct)) ** 2)
        for field_name in ("response_expected", "correction_present", "explanation_present"):
            expected_value = getattr(expected, field_name)
            if expected_value is not None:
                correct = getattr(observation, field_name) is expected_value
                matches[field_name] = correct
                field_scores[field_name].append(correct)
        if expected.topic_group:
            topic_assignments[(case.conversation_id, expected.topic_group)].append(observation.topic_key)

        results.append(
            PerceptionCaseResult(
                id=case.id,
                conversation_id=case.conversation_id,
                observation=observation,
                matches=matches,
            )
        )
        states[case.conversation_id] = evolve_simulated_state(state, DEFAULT_PERSONA, observation)

    # Topic stability measures whether messages labeled as the same human topic receive
    # the same learner-local topic key. It does not assume a curriculum topic identifier.
    stability_scores: list[float] = []
    for assigned_keys in topic_assignments.values():
        if assigned_keys:
            stability_scores.append(max(Counter(assigned_keys).values()) / len(assigned_keys))

    return PerceptionEvaluationReport(
        case_count=len(cases),
        labeled_feedback_count=len(field_scores["feedback"]),
        feedback_accuracy=_accuracy(field_scores["feedback"]),
        response_expected_accuracy=_accuracy(field_scores["response_expected"]),
        correction_accuracy=_accuracy(field_scores["correction_present"]),
        explanation_accuracy=_accuracy(field_scores["explanation_present"]),
        confidence_brier_score=(sum(confidence_errors) / len(confidence_errors) if confidence_errors else None),
        topic_stability=(sum(stability_scores) / len(stability_scores) if stability_scores else None),
        confusion_matrix={expected: dict(counts) for expected, counts in confusion.items()},
        cases=results,
    )


def load_evaluation_cases(path: str | Path) -> list[PerceptionEvaluationCase]:
    """Load JSONL evaluation cases."""
    source = Path(path).expanduser().resolve()
    cases: list[PerceptionEvaluationCase] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(PerceptionEvaluationCase.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"invalid evaluation case on line {line_number}: {exc}") from exc
    return cases


def save_evaluation_report(path: str | Path, report: PerceptionEvaluationReport) -> Path:
    """Atomically save a replay report."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(destination)
    return destination


def _accuracy(values: list[bool]) -> float | None:
    """Return no metric when a field had no human labels in the input corpus."""
    return sum(values) / len(values) if values else None
