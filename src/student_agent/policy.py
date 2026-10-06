"""Deterministic behavior policy for the simulated student agent.

The policy chooses a target response mode from persona parameters and tutor observation.
The LLM then realizes that mode in natural student language.
"""

from __future__ import annotations

import hashlib

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from .perception import PerceivedFeedback, StudentObservation
from .personas import StudentPersona


class SimulationMode(StrEnum):
    STATE_PROMPTED_STUDENT = "StatePromptedStudent"
    HIDDEN_STATE_STUDENT = "HiddenStateStudent"


class StudentResponseMode(StrEnum):
    ANSWER_CORRECT = "answer_correct"
    ANSWER_CORRECT_UNSURE = "answer_correct_unsure"
    CORRECT_WITH_OPEN_QUESTION = "correct_with_open_question"
    CORRECT_COMPUTATION_WRONG_CONCEPT = "correct_computation_wrong_concept"
    ANSWER_PARTIAL = "answer_partial"
    ANSWER_INCORRECT = "answer_incorrect"
    ASK_CLARIFICATION = "ask_clarification"
    REPEAT_UNDERSTANDING_QUESTION = "repeat_understanding_question"
    TOPIC_JUMP = "topic_jump"
    NUDGE = "nudge"


class SimulatedStudentState(BaseModel):
    """Private learner memory inferred only from the visible conversation."""

    model_config = ConfigDict(extra="forbid")

    current_topic: str = ""
    current_topic_key: str = ""
    mastery_by_topic: dict[str, float] = Field(default_factory=dict)
    skill_by_message_type: dict[str, float] = Field(default_factory=dict)
    attempts_by_topic: dict[str, int] = Field(default_factory=dict)
    misconception_strength_by_id: dict[str, float] = Field(default_factory=dict)
    active_misconceptions: list[str] = Field(default_factory=list)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    confusion: float = Field(0.3, ge=0.0, le=1.0)
    fatigue: float = Field(0.0, ge=0.0, le=1.0)
    motivation: float = Field(0.6, ge=0.0, le=1.0)
    help_seeking_tendency: float = Field(0.3, ge=0.0, le=1.0)
    abstraction_preference: float = Field(0.5, ge=0.0, le=1.0)
    verbosity: float = Field(0.4, ge=0.0, le=1.0)
    turns_seen: int = Field(0, ge=0)
    last_update_key: str = ""

    def mastery_for(self, topic_key: str, fallback: float) -> float:
        return self.mastery_by_topic.get(topic_key, fallback)

    def strongest_misconception(self, topic_key: str = "") -> tuple[str, float] | None:
        items = self.misconception_strength_by_id.items()
        if topic_key:
            items = ((k, v) for k, v in items if k.startswith(f"{topic_key}:"))
        strongest = max(items, key=lambda kv: kv[1], default=None)
        return strongest if strongest and strongest[1] > 0.05 else None

    def prompt_block(self, topic_key: str = "") -> str:
        mastery = self.mastery_by_topic.get(topic_key) if topic_key else None
        mastery_text = "(unknown)" if mastery is None else f"{mastery:.2f}"
        strongest = self.strongest_misconception(topic_key)
        misconception_text = "(none)" if strongest is None else f"{strongest[0]} strength {strongest[1]:.2f}"
        return "\n".join(
            [
                f"personal_understanding_for_topic: {mastery_text}",
                f"confidence: {self.confidence:.2f}",
                f"confusion: {self.confusion:.2f}",
                f"fatigue: {self.fatigue:.2f}",
                f"motivation: {self.motivation:.2f}",
                f"help_seeking_tendency: {self.help_seeking_tendency:.2f}",
                f"abstraction_preference: {self.abstraction_preference:.2f}",
                f"verbosity: {self.verbosity:.2f}",
                f"strongest_personal_misconception: {misconception_text}",
            ]
        )


class PolicyDecision(BaseModel):
    mode: StudentResponseMode
    reason: str
    repair_hint: str = ""

    def realization_hint(self) -> str:
        """Natural-language guidance that avoids exposing internal mode labels."""
        hints = {
            StudentResponseMode.ANSWER_CORRECT: (
                "Reply with a solid but student-like understanding. Include the key idea briefly."
            ),
            StudentResponseMode.ANSWER_CORRECT_UNSURE: (
                "Reply with the correct main idea, but sound unsure about whether the reasoning is sufficient. "
                "End with a brief confirmation question."
            ),
            StudentResponseMode.CORRECT_WITH_OPEN_QUESTION: (
                "Answer the tutor's immediate question correctly, then add one focused question about the same "
                "concept that shows your understanding is still unresolved. Do not change topics."
            ),
            StudentResponseMode.CORRECT_COMPUTATION_WRONG_CONCEPT: (
                "Give the correct computation or local fact, but attach one wrong conceptual interpretation. "
                "This should look like a student who can execute the procedure but still misunderstands what it means."
            ),
            StudentResponseMode.ANSWER_PARTIAL: (
                "Reply with an incomplete student answer. You MUST omit one required condition, leave one "
                "important detail unresolved, or make one slightly imprecise step. Do not accidentally give "
                "a fully correct answer."
            ),
            StudentResponseMode.ANSWER_INCORRECT: (
                "Reply with a plausible but wrong student answer. You MUST include exactly one concrete "
                "misconception or wrong claim, while keeping the rest natural and not absurd."
            ),
            StudentResponseMode.ASK_CLARIFICATION: (
                "Ask a focused clarification question about the part that still feels unclear."
            ),
            StudentResponseMode.REPEAT_UNDERSTANDING_QUESTION: (
                "Ask a focused understanding question about the same concept, even if the previous explanation "
                "was helpful. Make clear what still feels unresolved."
            ),
            StudentResponseMode.TOPIC_JUMP: (
                "Ask a short related aside that sounds like natural curiosity or mild distraction."
            ),
            StudentResponseMode.NUDGE: "Respond briefly so the tutor can continue.",
        }
        hint = hints[self.mode]
        if self.repair_hint:
            hint += f"\nAdditional retry constraint: {self.repair_hint}"
        return hint


def initial_simulated_state(persona: StudentPersona) -> SimulatedStudentState:
    controls = persona.control_profile()
    confidence = controls.confidence
    confusion = _clamp01(1.0 - controls.mastery + controls.misconception_rate * 0.15)
    help_seeking = controls.help_seeking
    if "abstract" in persona.description.lower():
        abstraction = 0.75
    elif "concrete" in persona.description.lower():
        abstraction = 0.35
    else:
        abstraction = 0.5
    verbosity = {"terse": 0.15, "concise": 0.35, "hesitant": 0.55, "verbose": 0.80}[persona.answer_style]
    return SimulatedStudentState(
        confidence=confidence,
        confusion=confusion,
        motivation=0.35 + controls.learning_rate * 0.45,
        help_seeking_tendency=help_seeking,
        abstraction_preference=abstraction,
        verbosity=verbosity,
        skill_by_message_type={},
    )


def evolve_simulated_state(
    simulated: SimulatedStudentState,
    persona: StudentPersona,
    observation: StudentObservation,
) -> SimulatedStudentState:
    """Update private learner state from the student's perception of visible text."""
    scalars = persona.policy_scalars()
    updated = simulated.model_copy(deep=True)
    update_key = f"{observation.turn_number}:{observation.topic_key}:{observation.perceived_feedback}"
    if updated.last_update_key == update_key:
        return updated

    updated.current_topic = observation.perceived_topic
    updated.current_topic_key = observation.topic_key
    if observation.response_expected:
        updated.attempts_by_topic[observation.topic_key] = observation.attempt_number

    mastery = updated.mastery_for(observation.topic_key, scalars.correct_probability)
    message_key = observation.message_type.value
    message_skill = updated.skill_by_message_type.get(message_key, scalars.correct_probability)

    if observation.perceived_feedback is PerceivedFeedback.CORRECT:
        mastery += (1.0 - mastery) * (0.10 + scalars.learning_gain)
        message_skill += (1.0 - message_skill) * (0.08 + scalars.learning_gain * 0.5)
        updated.confusion *= 0.85
        updated.confidence += (1.0 - updated.confidence) * 0.10
        updated.motivation += (1.0 - updated.motivation) * 0.04
        decay_factor = _misconception_decay_factor(persona)
        for mid, strength in list(updated.misconception_strength_by_id.items()):
            if mid.startswith(f"{observation.topic_key}:"):
                new_strength = strength * decay_factor
                if new_strength <= 0.05:
                    updated.misconception_strength_by_id.pop(mid)
                else:
                    updated.misconception_strength_by_id[mid] = new_strength
        updated.active_misconceptions = [
            m for m in updated.active_misconceptions if updated.misconception_strength_by_id.get(m, 0.0) > 0.05
        ]
    elif observation.perceived_feedback is PerceivedFeedback.PARTIAL:
        mastery += (1.0 - mastery) * (0.04 + scalars.learning_gain * 0.5)
        message_skill += (1.0 - message_skill) * (0.03 + scalars.learning_gain * 0.25)
        updated.confusion += (1.0 - updated.confusion) * 0.04
        updated.confidence += (1.0 - updated.confidence) * 0.02
        misconception = _misconception_id(observation)
        updated.misconception_strength_by_id[misconception] = _clamp01(
            updated.misconception_strength_by_id.get(misconception, scalars.misconception_probability * 0.4) + 0.10
        )
        if misconception not in updated.active_misconceptions:
            updated.active_misconceptions.append(misconception)
    elif observation.perceived_feedback is PerceivedFeedback.INCORRECT:
        mastery -= mastery * (0.04 + scalars.misconception_probability * 0.08)
        message_skill -= message_skill * (0.03 + scalars.misconception_probability * 0.05)
        updated.confusion += (1.0 - updated.confusion) * 0.10
        updated.confidence *= 0.92
        updated.motivation *= 0.97
        misconception = _misconception_id(observation)
        updated.misconception_strength_by_id[misconception] = _clamp01(
            updated.misconception_strength_by_id.get(misconception, scalars.misconception_probability) + 0.20
        )
        if misconception not in updated.active_misconceptions:
            updated.active_misconceptions.append(misconception)
    else:
        # Explanations and hints may make the learner slightly less confused, but
        # no authoritative correctness signal exists unless the student perceives one.
        if observation.message_type.value in {"explanation", "hint"}:
            updated.confusion *= 0.97

    updated.mastery_by_topic[observation.topic_key] = _clamp01(mastery)
    updated.skill_by_message_type[message_key] = _clamp01(message_skill)
    updated.confidence = _clamp01(updated.confidence)
    updated.confusion = _clamp01(updated.confusion)
    updated.fatigue = _clamp01(updated.fatigue + 0.01 + observation.perceived_difficulty * 0.01)
    updated.motivation = _clamp01(updated.motivation)
    updated.turns_seen += 1
    updated.last_update_key = update_key
    return updated


def choose_response_mode(
    persona: StudentPersona,
    observation: StudentObservation,
    *,
    mode: SimulationMode = SimulationMode.HIDDEN_STATE_STUDENT,
    simulated_state: SimulatedStudentState | None = None,
) -> PolicyDecision:
    """Choose the next student behavior reproducibly from persona + observation."""
    scalars = persona.policy_scalars()
    roll = _stable_unit(
        persona.id,
        observation.topic_key,
        observation.message_type.value,
        observation.tutor_message,
        observation.turn_number,
        observation.experiment_seed,
    )

    if roll < scalars.topic_jump_probability:
        return PolicyDecision(
            mode=StudentResponseMode.TOPIC_JUMP, reason="persona occasionally wanders to a related topic"
        )

    if not observation.tutor_message:
        return PolicyDecision(mode=StudentResponseMode.NUDGE, reason="no tutor message is available to answer")

    if not observation.response_expected:
        clarification_probability = scalars.clarification_probability
        if uses_simulated_state_policy(mode) and simulated_state is not None:
            clarification_probability = min(
                1.0,
                clarification_probability
                + simulated_state.help_seeking_tendency * 0.20
                + simulated_state.confusion * 0.10
                - simulated_state.confidence * 0.04,
            )
            repeat_probability = _repeat_understanding_probability(persona, observation, scalars, simulated_state)
            repeat_roll = _stable_unit(
                persona.id,
                observation.topic_key,
                observation.message_type.value,
                observation.tutor_message,
                observation.turn_number,
                "repeat_understanding",
                observation.experiment_seed,
            )
            if repeat_roll < repeat_probability:
                return _decision(
                    StudentResponseMode.REPEAT_UNDERSTANDING_QUESTION,
                    "student still has an unresolved understanding question",
                    observation,
                )
        if roll < scalars.topic_jump_probability + clarification_probability:
            return PolicyDecision(
                mode=StudentResponseMode.ASK_CLARIFICATION,
                reason="the student perceived no direct answer request and wants clarification",
            )
        return PolicyDecision(mode=StudentResponseMode.NUDGE, reason="continue naturally without forcing an answer")

    if uses_simulated_state_policy(mode) and simulated_state is not None:
        mastery = simulated_state.mastery_for(observation.topic_key, scalars.correct_probability)
        message_skill = simulated_state.skill_by_message_type.get(
            observation.message_type.value, scalars.correct_probability
        )
        strongest = simulated_state.strongest_misconception(observation.topic_key)
        misconception_pressure = strongest[1] if strongest else 0.0
        if strongest and strongest[1] > 0.45:
            repeat_roll = _stable_unit(
                persona.id,
                observation.topic_key,
                observation.message_type.value,
                observation.tutor_message,
                observation.turn_number,
                "repeat_misconception",
                observation.experiment_seed,
            )
            repeat_probability = min(0.85, scalars.misconception_probability + strongest[1] * 0.35)
            if repeat_roll < repeat_probability:
                return _decision(
                    StudentResponseMode.ANSWER_INCORRECT,
                    "active misconception persists after feedback",
                    observation,
                )
        adjusted_correct = _clamp01(
            0.50 * mastery
            + 0.25 * message_skill
            + 0.25 * scalars.correct_probability
            + observation.attempt_number * scalars.learning_gain * 0.5
            - simulated_state.confusion * 0.15
            - simulated_state.fatigue * 0.05
            - misconception_pressure * 0.12
        )
        adjusted_partial = min(
            1.0 - adjusted_correct,
            scalars.partial_probability + simulated_state.confusion * 0.16 + misconception_pressure * 0.16,
        )
    else:
        adjusted_correct = min(
            1.0,
            scalars.correct_probability
            + (observation.attempt_number * scalars.learning_gain)
            - (scalars.misconception_probability * 0.20),
        )
        adjusted_partial = min(1.0 - adjusted_correct, scalars.partial_probability)

    if roll < adjusted_correct:
        correct_mode = StudentResponseMode.ANSWER_CORRECT
        if uses_simulated_state_policy(mode) and simulated_state is not None:
            correct_mode = _correct_variant(persona, observation, simulated_state, scalars, misconception_pressure)
        return _decision(correct_mode, "persona is likely to answer this check correctly", observation)
    if roll < adjusted_correct + adjusted_partial:
        return _decision(
            StudentResponseMode.ANSWER_PARTIAL,
            "persona has a partial grasp or omits one important condition",
            observation,
        )
    return _decision(
        StudentResponseMode.ANSWER_INCORRECT,
        "persona is likely to reveal a misconception or fragile understanding",
        observation,
    )


def uses_simulated_state_policy(mode: SimulationMode) -> bool:
    return mode in {SimulationMode.STATE_PROMPTED_STUDENT, SimulationMode.HIDDEN_STATE_STUDENT}


def _decision(mode: StudentResponseMode, reason: str, _observation: StudentObservation) -> PolicyDecision:
    return PolicyDecision(mode=mode, reason=reason)


def _correct_variant(
    persona: StudentPersona,
    observation: StudentObservation,
    simulated_state: SimulatedStudentState,
    scalars,
    misconception_pressure: float,
) -> StudentResponseMode:
    """Sometimes make locally correct answers pedagogically unresolved."""
    uncertainty_pressure = _uncertainty_pressure(simulated_state, scalars, misconception_pressure)
    if uncertainty_pressure <= 0.15:
        return StudentResponseMode.ANSWER_CORRECT

    roll = _stable_unit(
        persona.id,
        observation.topic_key,
        observation.message_type.value,
        observation.tutor_message,
        observation.turn_number,
        "correct_uncertainty_variant",
        observation.experiment_seed,
    )
    if roll >= uncertainty_pressure:
        return StudentResponseMode.ANSWER_CORRECT

    conceptual_roll = _stable_unit(
        persona.id,
        observation.topic_key,
        observation.message_type.value,
        observation.tutor_message,
        observation.turn_number,
        "correct_wrong_concept_variant",
        observation.experiment_seed,
    )
    if misconception_pressure > 0.18 and conceptual_roll < min(0.55, scalars.misconception_probability + 0.25):
        return StudentResponseMode.CORRECT_COMPUTATION_WRONG_CONCEPT

    if simulated_state.help_seeking_tendency + simulated_state.confusion > 1.05:
        return StudentResponseMode.CORRECT_WITH_OPEN_QUESTION
    return StudentResponseMode.ANSWER_CORRECT_UNSURE


def _uncertainty_pressure(
    simulated_state: SimulatedStudentState,
    scalars,
    misconception_pressure: float,
) -> float:
    """Estimate how strongly a correct answer should still sound uncertain."""
    return _clamp01(
        simulated_state.confusion * 0.35
        + (1.0 - simulated_state.confidence) * 0.25
        + simulated_state.help_seeking_tendency * 0.20
        + misconception_pressure * 0.25
        + scalars.misconception_probability * 0.15
    )


def _repeat_understanding_probability(
    persona: StudentPersona,
    observation: StudentObservation,
    scalars,
    simulated_state: SimulatedStudentState,
) -> float:
    """Estimate whether the learner asks again because understanding is unresolved."""
    strongest = simulated_state.strongest_misconception()
    misconception_pressure = strongest[1] if strongest else 0.0
    weak_persistence_boost = 0.0
    if persona.persistence_style == "fragile":
        weak_persistence_boost += 0.14
    if persona.confidence_style == "low":
        weak_persistence_boost += 0.08
    if observation.perceived_feedback is PerceivedFeedback.CORRECT and simulated_state.confusion > 0.45:
        weak_persistence_boost += 0.12
    if observation.perceived_feedback is PerceivedFeedback.PARTIAL:
        weak_persistence_boost += 0.08
    return _clamp01(
        simulated_state.help_seeking_tendency * 0.24
        + simulated_state.confusion * 0.22
        + (1.0 - simulated_state.confidence) * 0.10
        + misconception_pressure * 0.18
        + scalars.misconception_probability * 0.10
        + weak_persistence_boost
    )


def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))


def _misconception_decay_factor(persona: StudentPersona) -> float:
    """Return the fraction of misconception strength retained after correction."""
    if persona.persistence_style == "stubborn":
        return 0.80
    if persona.persistence_style == "fragile":
        return 0.55
    return 0.65


def _misconception_id(observation: StudentObservation) -> str:
    base = observation.topic_key or "unknown_concept"
    digest = hashlib.sha256(observation.perceived_topic.encode("utf-8")).hexdigest()[:8]
    return f"{base}:perceived_gap_{digest}"


def _stable_unit(*parts: object) -> float:
    # Preserve the historical unseeded hash exactly. The reusable engine appends an
    # experiment seed as the final component only when callers provide one.
    if parts and parts[-1] == "":
        parts = parts[:-1]
    text = "\n".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)
