"""Deterministic behavior policy for the simulated student agent.

The policy chooses a target response mode from persona parameters and tutor observation.
The LLM then realizes that mode in natural student language. Keeping this choice in Python
makes behavior inspectable and reproducible instead of asking a language model to invent
both the learner state and the final wording at once.
"""

from __future__ import annotations

import hashlib

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from .perception import MessageType, PerceivedFeedback, StudentObservation
from .personas import StudentPersona


class SimulationMode(StrEnum):
    """Available strategies for turning private learner state into behavior."""

    HIDDEN_STATE_STUDENT = "HiddenStateStudent"


class StudentResponseMode(StrEnum):
    """Behavior categories selected before natural-language generation.

    These values describe what the reply should *do*. They are never intended to appear
    in the visible learner reply.
    """

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


class KnowledgeComponentState(BaseModel):
    """One proposition the learner encountered in the visible conversation.

    Exposure means the learner saw an explanation. Demonstration means later wording
    suggested the learner could use the idea. The two strengths stay separate because
    merely being told a fact is not evidence of mastery.
    """

    model_config = ConfigDict(extra="forbid")

    topic_key: str
    statement: str
    exposure_strength: float = Field(0.0, ge=0.0, le=1.0)
    demonstrated_strength: float = Field(0.0, ge=0.0, le=1.0)


class MisconceptionState(BaseModel):
    """One specific mistaken belief inferred from the visible conversation.

    A misconception can be linked to several tasks under the same topic. ``strength``
    controls how likely the policy is to repeat it, while ``status`` keeps resolved beliefs
    available for inspection without treating them as active.
    """

    model_config = ConfigDict(extra="forbid")

    topic_key: str
    task_key: str = ""
    task_keys: list[str] = Field(default_factory=list)
    belief: str
    strength: float = Field(0.0, ge=0.0, le=1.0)
    evidence: str = ""
    evidence_count: int = Field(0, ge=0)
    status: str = "active"


class SimulatedStudentState(BaseModel):
    """Everything the simulator remembers about one learner.

    This is private *student-side* memory. Topic and task keys are created from the
    visible conversation; they are not IDs copied from a tutor or course database.
    Values such as confidence and mastery are normalized to the range 0..1.
    """

    model_config = ConfigDict(extra="forbid")

    # Current focus and the student's local topic/task registry.
    current_topic: str = ""
    current_topic_key: str = ""
    current_task: str = ""
    current_task_key: str = ""
    topic_labels_by_key: dict[str, str] = Field(default_factory=dict)
    topic_aliases_by_key: dict[str, list[str]] = Field(default_factory=dict)
    task_labels_by_key: dict[str, str] = Field(default_factory=dict)
    task_topic_by_key: dict[str, str] = Field(default_factory=dict)
    topic_last_seen_turn_by_key: dict[str, int] = Field(default_factory=dict)
    task_last_seen_turn_by_key: dict[str, int] = Field(default_factory=dict)
    # Learning estimates accumulated across turns.
    mastery_by_topic: dict[str, float] = Field(default_factory=dict)
    instructional_exposure_by_topic: dict[str, float] = Field(default_factory=dict)
    skill_by_message_type: dict[str, float] = Field(default_factory=dict)
    attempts_by_topic: dict[str, int] = Field(default_factory=dict)
    attempts_by_task: dict[str, int] = Field(default_factory=dict)
    knowledge_components_by_id: dict[str, KnowledgeComponentState] = Field(default_factory=dict)
    misconceptions_by_id: dict[str, MisconceptionState] = Field(default_factory=dict)
    # Legacy mirrors retained so version-2 snapshots remain readable.
    misconception_strength_by_id: dict[str, float] = Field(default_factory=dict)
    active_misconceptions: list[str] = Field(default_factory=list)
    # Whole-learner tendencies that are not tied to one particular topic.
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
        """Return learned topic mastery or the persona baseline for a new topic."""
        return self.mastery_by_topic.get(topic_key, fallback)

    def strongest_misconception(self, topic_key: str = "") -> tuple[str, float] | None:
        """Return the strongest active belief, optionally restricted to one topic.

        The second lookup supports older version-2 snapshots that contain only the legacy
        strength mapping. New state should normally use ``misconceptions_by_id``.
        """
        semantic_items = self.misconceptions_by_id.values()
        if topic_key:
            semantic_items = (item for item in semantic_items if item.topic_key == topic_key)
        semantic_strongest = max(semantic_items, key=lambda item: item.strength, default=None)
        if semantic_strongest and semantic_strongest.strength > 0.05:
            return semantic_strongest.belief, semantic_strongest.strength

        items = self.misconception_strength_by_id.items()
        if topic_key:
            items = ((k, v) for k, v in items if k.startswith(f"{topic_key}:"))
        strongest = max(items, key=lambda kv: kv[1], default=None)
        return strongest if strongest and strongest[1] > 0.05 else None

class PolicyDecision(BaseModel):
    """The policy's intended behavior for one reply, before wording is generated."""

    mode: StudentResponseMode
    reason: str
    repair_hint: str = ""
    target_misconception: str = ""

    def realization_hint(self) -> str:
        """Natural-language guidance that avoids exposing internal mode labels."""
        # Instructions describe observable behavior rather than exposing enum names. This
        # gives the generator enough direction while keeping implementation language out
        # of the student's visible answer.
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
        if self.target_misconception and self.mode in {
            StudentResponseMode.ANSWER_INCORRECT,
            StudentResponseMode.CORRECT_COMPUTATION_WRONG_CONCEPT,
        }:
            hint += f"\nKeep the mistake consistent with this learner belief: {self.target_misconception}"
        if self.repair_hint:
            hint += f"\nAdditional retry constraint: {self.repair_hint}"
        return hint


def initial_simulated_state(persona: StudentPersona) -> SimulatedStudentState:
    """Translate a reusable persona into the starting state of one session."""
    controls = persona.control_profile()
    confidence = controls.confidence
    confusion = _clamp01(1.0 - controls.mastery + controls.misconception_rate * 0.15)
    help_seeking = controls.help_seeking
    # Abstraction preference currently has no dedicated public control. Preserve the
    # lightweight description-based default until it becomes an explicit persona field.
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
        motivation=controls.motivation,
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
    """Return the next private state after one perceived tutor message.

    The input is copied, so callers can discard this provisional result when generation
    or validation fails. Learning is based on what the student *thought* the tutor said,
    which may differ from the tutor's real intention.
    """
    scalars = persona.policy_scalars()
    updated = simulated.model_copy(deep=True)
    # ``update_key`` makes this calculation idempotent for generation retries. It is not a
    # database identifier; it only describes which perceived tutor turn was last applied.
    update_key = f"{observation.turn_number}:{observation.topic_key}:{observation.perceived_feedback}"
    # A prepared turn may be regenerated several times. Do not learn from the same
    # tutor message again on every retry.
    if updated.last_update_key == update_key:
        return updated

    # Maintain the student's own topic/task registry. Stable local keys let later
    # messages refer back to the same work even when the tutor uses different wording.
    updated.current_topic = observation.perceived_topic
    updated.current_topic_key = observation.topic_key
    updated.current_task = observation.perceived_task
    updated.current_task_key = observation.task_key
    updated.topic_labels_by_key.setdefault(observation.topic_key, observation.perceived_topic)
    aliases = updated.topic_aliases_by_key.setdefault(observation.topic_key, [])
    if observation.perceived_topic not in aliases:
        aliases.append(observation.perceived_topic)
    updated.task_labels_by_key.setdefault(observation.task_key, observation.perceived_task)
    updated.task_topic_by_key[observation.task_key] = observation.topic_key
    seen_turn = updated.turns_seen + 1
    updated.topic_last_seen_turn_by_key[observation.topic_key] = seen_turn
    updated.task_last_seen_turn_by_key[observation.task_key] = seen_turn
    if observation.response_expected:
        updated.attempts_by_topic[observation.topic_key] = observation.attempt_number
        updated.attempts_by_task[observation.task_key] = observation.attempt_number

    mastery = updated.mastery_for(observation.topic_key, scalars.correct_probability)
    message_key = observation.message_type.value
    message_skill = updated.skill_by_message_type.get(message_key, scalars.correct_probability)
    feedback_weight = (
        observation.feedback_confidence
        if observation.perceived_feedback is not PerceivedFeedback.NONE
        else 0.0
    )
    instructional = observation.correction_present or observation.explanation_present
    demonstrated = observation.demonstrated_understanding or (
        observation.perceived_feedback is PerceivedFeedback.CORRECT
    )

    # Exposure (the tutor explained it) and demonstrated ability (the learner showed it)
    # are deliberately separate. Hearing an answer is not the same as knowing it.
    _update_knowledge_components(updated, observation, demonstrated=demonstrated)
    if instructional:
        prior_exposure = updated.instructional_exposure_by_topic.get(observation.topic_key, 0.0)
        exposure_gain = (0.08 + scalars.learning_gain * 0.20) * max(feedback_weight, 0.5)
        updated.instructional_exposure_by_topic[observation.topic_key] = _clamp01(
            prior_exposure + (1.0 - prior_exposure) * exposure_gain
        )

    if observation.perceived_feedback is PerceivedFeedback.CORRECT:
        # Confirmation is mainly evidence/reinforcement, not a large learning event.
        independence_weight = _demonstration_independence_weight(observation.attempt_number)
        demonstration_gain = 0.04 + scalars.learning_gain * 0.30 if demonstrated else 0.01
        mastery += (1.0 - mastery) * demonstration_gain * feedback_weight * independence_weight
        message_skill += (
            (1.0 - message_skill)
            * (0.015 + scalars.learning_gain * 0.10)
            * feedback_weight
            * independence_weight
        )
        updated.confusion *= 1.0 - 0.15 * feedback_weight * independence_weight
        updated.confidence += (1.0 - updated.confidence) * 0.10 * feedback_weight
        updated.motivation += (1.0 - updated.motivation) * 0.04 * feedback_weight
        _decay_misconceptions(updated, observation, persona, feedback_weight)
    elif observation.perceived_feedback is PerceivedFeedback.PARTIAL:
        if instructional:
            # Partial work is weak evidence of competence. Most of the tutor's
            # explanation is recorded as exposure above and is not immediately
            # converted into mastery.
            mastery += (1.0 - mastery) * (0.005 + scalars.learning_gain * 0.10) * feedback_weight
            message_skill += (1.0 - message_skill) * (0.005 + scalars.learning_gain * 0.05) * feedback_weight
        if observation.explanation_present:
            updated.confusion *= 1.0 - 0.015 * feedback_weight
        else:
            updated.confusion += (1.0 - updated.confusion) * 0.04 * feedback_weight
        updated.confidence *= 1.0 - 0.03 * feedback_weight
        _record_misconception(
            updated,
            observation,
            gain=(0.06 if instructional else 0.10) * feedback_weight,
            seed=scalars.misconception_probability * 0.4 * feedback_weight,
        )
    elif observation.perceived_feedback is PerceivedFeedback.INCORRECT:
        # A correction is exposure, not evidence that the learner can now apply it.
        # Keep mastery stable until a later partial or correct response demonstrates
        # uptake. Repeated failure may remain confusing even when the explanation is clear.
        if observation.explanation_present:
            updated.confusion *= 1.0 - 0.01 * feedback_weight
            repetition_pressure = min(max(observation.attempt_number - 1, 0), 5) / 5
            updated.confusion += (1.0 - updated.confusion) * 0.02 * repetition_pressure * feedback_weight
        else:
            updated.confusion += (1.0 - updated.confusion) * 0.10 * feedback_weight
        updated.confidence *= 1.0 - 0.08 * feedback_weight
        updated.motivation *= 1.0 - 0.03 * feedback_weight
        _record_misconception(
            updated,
            observation,
            gain=(0.10 if instructional else 0.20) * feedback_weight,
            seed=scalars.misconception_probability * feedback_weight,
        )
    else:
        # Unassessed teaching increases exposure but not demonstrated mastery.
        if observation.explanation_present or observation.message_type.value in {"explanation", "hint"}:
            message_skill += (1.0 - message_skill) * scalars.learning_gain * 0.02
            updated.confusion *= 0.99

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
    """Choose the next student behavior reproducibly from persona + observation.

    This chooses a behavior category, not the final wording. The same inputs and seed
    produce the same pseudo-random roll, which makes experiment runs repeatable.
    """
    scalars = persona.policy_scalars()
    # A hash-based roll avoids mutable random-number-generator state.
    roll = _stable_unit(
        persona.id,
        observation.topic_key,
        observation.message_type.value,
        observation.tutor_message,
        observation.turn_number,
        observation.experiment_seed,
    )

    if not observation.tutor_message:
        return PolicyDecision(mode=StudentResponseMode.NUDGE, reason="no tutor message is available to answer")

    # Handle strong conversational constraints before ordinary probability sampling.
    # A learner cannot sensibly revise an answer when the tutor only says it is wrong.
    bare_negative_feedback = (
        observation.perceived_feedback is PerceivedFeedback.INCORRECT
        and observation.message_type is MessageType.FEEDBACK
        and not observation.correction_present
        and not observation.explanation_present
    )
    if bare_negative_feedback:
        strongest = simulated_state.strongest_misconception(observation.topic_key) if simulated_state else None
        if persona.persistence_style == "stubborn" and strongest and strongest[1] > 0.45:
            return PolicyDecision(
                mode=StudentResponseMode.ANSWER_INCORRECT,
                reason="stubborn learner maintains a strong belief after unexplained negative feedback",
                target_misconception=strongest[0],
            )
        return PolicyDecision(
            mode=StudentResponseMode.ASK_CLARIFICATION,
            reason="the tutor rejected the answer without providing a correction or explanation",
        )

    if roll < scalars.topic_jump_probability:
        return PolicyDecision(
            mode=StudentResponseMode.TOPIC_JUMP, reason="persona occasionally wanders to a related topic"
        )

    if not observation.response_expected:
        # Explanations and feedback often need only an acknowledgement. A confused or
        # help-seeking learner may still ask a question, but we do not force an answer to
        # a question the learner did not perceive.
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

    # For answerable questions, combine the stable persona with what this particular
    # session has learned. Confusion, fatigue, and active misconceptions reduce the
    # chance of a correct response; practice and learning gain increase it.
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
            repeat_probability = _misconception_repeat_probability(
                persona,
                misconception_probability=scalars.misconception_probability,
                misconception_strength=strongest[1],
                attempt_number=observation.attempt_number,
            )
            if repeat_roll < repeat_probability:
                return PolicyDecision(
                    mode=StudentResponseMode.ANSWER_INCORRECT,
                    reason="active misconception persists after feedback",
                    target_misconception=strongest[0],
                )
        # This weighted estimate blends long-term topic mastery, skill with this kind of
        # tutor message, and the persona baseline. Session conditions then move the chance
        # up or down without permanently changing the persona itself.
        adjusted_correct = _clamp01(
            0.50 * mastery
            + 0.25 * message_skill
            + 0.25 * scalars.correct_probability
            + min(observation.attempt_number, 3) * scalars.learning_gain * 0.15
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
            + (min(observation.attempt_number, 3) * scalars.learning_gain * 0.30)
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
    """Return whether this mode should use evolving per-session learner state."""
    return mode is SimulationMode.HIDDEN_STATE_STUDENT


def _decision(mode: StudentResponseMode, reason: str, observation: StudentObservation) -> PolicyDecision:
    target = ""
    if mode in {StudentResponseMode.ANSWER_INCORRECT, StudentResponseMode.CORRECT_COMPUTATION_WRONG_CONCEPT}:
        target = observation.misconception_statement
    return PolicyDecision(mode=mode, reason=reason, target_misconception=target)


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


def _demonstration_independence_weight(attempt_number: int) -> float:
    """Discount mastery evidence produced only after extensive scaffolding."""
    return max(0.35, 1.0 / (1.0 + max(attempt_number - 1, 0) * 0.25))


def _misconception_repeat_probability(
    persona: StudentPersona,
    *,
    misconception_probability: float,
    misconception_strength: float,
    attempt_number: int,
) -> float:
    """Return bounded persistence that eventually adapts after repeated correction."""
    dynamics = persona.learning_dynamics
    style_boost = {"fragile": 0.0, "steady": 0.03, "stubborn": 0.08}[persona.persistence_style]
    purpose_boost = 0.08 if dynamics.purpose == "tutor_stress_test" else 0.0
    cap = 0.90 if dynamics.purpose == "tutor_stress_test" else 0.80
    initial = min(
        cap,
        misconception_probability * 0.55 + misconception_strength * 0.30 + style_boost + purpose_boost,
    )
    excess_corrections = max(attempt_number - dynamics.corrections_before_adaptation, 0)
    return _clamp01(initial * dynamics.misconception_resistance**excess_corrections)


def _misconception_decay_factor(persona: StudentPersona) -> float:
    """Return the fraction of misconception strength retained after correction."""
    return persona.learning_dynamics.misconception_resistance


def _component_id(topic_key: str, statement: str) -> str:
    """Create a stable learner-local ID for equivalent normalized propositions."""
    normalized = " ".join(statement.lower().split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:10]
    return f"{topic_key}:component_{digest}"


def _misconception_id(topic_key: str, belief: str) -> str:
    """Create a stable learner-local ID when perception did not match an existing belief."""
    normalized = " ".join(belief.lower().split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:10]
    return f"{topic_key}:misconception_{digest}"


def _update_knowledge_components(
    state: SimulatedStudentState,
    observation: StudentObservation,
    *,
    demonstrated: bool,
) -> None:
    """Merge newly perceived propositions into the learner's private knowledge memory."""
    for statement in observation.knowledge_components:
        normalized = " ".join(statement.split()).strip()
        if not normalized:
            continue
        component_id = _component_id(observation.topic_key, normalized)
        component = state.knowledge_components_by_id.get(
            component_id,
            KnowledgeComponentState(topic_key=observation.topic_key, statement=normalized),
        ).model_copy(deep=True)
        if observation.correction_present or observation.explanation_present:
            component.exposure_strength = _clamp01(component.exposure_strength + 0.25)
        if demonstrated:
            component.demonstrated_strength = _clamp01(
                component.demonstrated_strength + 0.35 * max(observation.feedback_confidence, 0.5)
            )
        state.knowledge_components_by_id[component_id] = component


def _record_misconception(
    state: SimulatedStudentState,
    observation: StudentObservation,
    *,
    gain: float,
    seed: float,
) -> None:
    """Create or strengthen the misconception supported by the current observation."""
    belief = " ".join(observation.misconception_statement.split()).strip()
    if not belief:
        return
    # Prefer a semantic match returned by perception so paraphrases strengthen one belief
    # instead of creating duplicate records. Never accept a match from another topic.
    matched_id = observation.matched_misconception_id.strip()
    matched = state.misconceptions_by_id.get(matched_id)
    if matched is None or matched.topic_key != observation.topic_key:
        matched_id = ""
    misconception_id = matched_id or _misconception_id(observation.topic_key, belief)
    misconception = state.misconceptions_by_id.get(
        misconception_id,
        MisconceptionState(
            topic_key=observation.topic_key,
            task_key=observation.task_key,
            task_keys=[observation.task_key] if observation.task_key else [],
            belief=belief,
            strength=seed,
            evidence=observation.evidence,
        ),
    ).model_copy(deep=True)
    misconception.strength = _clamp01(misconception.strength + gain)
    misconception.task_key = misconception.task_key or observation.task_key
    if observation.task_key and observation.task_key not in misconception.task_keys:
        misconception.task_keys.append(observation.task_key)
    misconception.evidence = observation.evidence or misconception.evidence
    misconception.evidence_count += 1
    misconception.status = "active"
    state.misconceptions_by_id[misconception_id] = misconception
    state.misconception_strength_by_id[misconception_id] = misconception.strength
    if misconception.strength > 0.05 and misconception_id not in state.active_misconceptions:
        state.active_misconceptions.append(misconception_id)


def _decay_misconceptions(
    state: SimulatedStudentState,
    observation: StudentObservation,
    persona: StudentPersona,
    feedback_weight: float,
) -> None:
    """Weaken task-relevant misconceptions after demonstrated correct understanding."""
    decay_factor = _misconception_decay_factor(persona)
    weighted_decay = 1.0 - (1.0 - decay_factor) * feedback_weight
    for misconception_id, misconception in list(state.misconceptions_by_id.items()):
        if misconception.topic_key != observation.topic_key:
            continue
        # A correct answer should weaken the misconception relevant to the task
        # that was just demonstrated, not every misconception under the topic.
        misconception_tasks = set(misconception.task_keys)
        if misconception.task_key:
            misconception_tasks.add(misconception.task_key)
        if misconception_tasks and observation.task_key and observation.task_key not in misconception_tasks:
            continue
        updated = misconception.model_copy(deep=True)
        updated.strength *= weighted_decay
        if updated.strength <= 0.05:
            updated.status = "resolved"
        state.misconceptions_by_id[misconception_id] = updated
        state.misconception_strength_by_id[misconception_id] = updated.strength
    state.active_misconceptions = [
        misconception_id
        for misconception_id in state.active_misconceptions
        if state.misconception_strength_by_id.get(misconception_id, 0.0) > 0.05
    ]


def _stable_unit(*parts: object) -> float:
    """Map stable turn data to a reproducible pseudo-random number in the range 0..1."""
    # A content hash avoids mutable random-generator state, so saving and restoring a
    # session does not change later choices. Preserve the historical unseeded hash exactly:
    # the reusable engine appends an experiment seed only when callers provide one.
    if parts and parts[-1] == "":
        parts = parts[:-1]
    text = "\n".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)
