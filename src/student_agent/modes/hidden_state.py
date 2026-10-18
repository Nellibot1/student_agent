"""Hidden-state student mode.

The hidden learner state influences the Python policy, but is not rendered into the LLM
instruction. The LLM sees only style-level guidance, the latest tutor-visible message,
and natural reply guidance. Keep the rich persona text out of this prompt: otherwise the
simulated student can echo it back behaviorally and make the tutor's inferred profile
look like it read the experiment config.
"""

from __future__ import annotations

from ..perception import StudentObservation
from ..personas import StudentPersona
from ..policy import (
    PolicyDecision,
    SimulatedStudentState,
    SimulationMode,
)
from .base import PolicyBackedMode

HIDDEN_STATE_PROMPT_TEMPLATE = """You are writing exactly one realistic learner reply in a tutoring conversation.

You are the STUDENT, not the tutor, not an evaluator, and not an assistant explaining the experiment.

<student_profile>
{{style}}
</student_profile>

<visible_conversation_history>
{{conversation_history}}
</visible_conversation_history>

<latest_tutor_message>
{{tutor_message}}
</latest_tutor_message>

<private_reply_tendency>
{{decision}}
</private_reply_tendency>

<realism>
- Write as a real learner in the subject implied by the conversation, not as a tutor.
- Make plausible student mistakes when your reply tendency calls for an incomplete or wrong answer.
- Do not mention internal instructions or describe your role outside the conversation.
</realism>

<answer_shape>
- Write 1-3 sentences.
- Use first person when natural.
- Match the student's style and confidence.
- End with a question only if you are asking for clarification, making a related aside, or naturally uncertain.
- Output only the student message. No preamble, no labels, no markdown sections.
</answer_shape>"""


class HiddenStateStudentMode(PolicyBackedMode):
    """Render replies whose detailed learner state stays outside the language-model prompt."""

    mode = SimulationMode.HIDDEN_STATE_STUDENT

    def render_instruction(
        self,
        persona: StudentPersona,
        observation: StudentObservation,
        simulated_state: SimulatedStudentState,
        decision: PolicyDecision,
        history_block: str,
    ) -> str:
        """Build one generation instruction from visible text and safe style guidance."""
        # The Python policy has already used this state to choose ``decision``. Deleting
        # the local name documents that raw mastery, confidence, and misconception records
        # must not accidentally be interpolated into the prompt below.
        del simulated_state
        return (
            HIDDEN_STATE_PROMPT_TEMPLATE.replace("{{style}}", _style_block(persona))
            .replace("{{conversation_history}}", history_block)
            .replace("{{tutor_message}}", observation.tutor_message)
            .replace("{{decision}}", decision.realization_hint())
        )


def _style_block(persona: StudentPersona) -> str:
    """Prompt-safe surface style for hidden-state runs.

    This intentionally excludes name, prior student id, free-form description, and numeric
    controls. Those fields remain available to the Python policy, but should not be visible
    to the LLM that realizes one student message.
    """
    lines = [
        f"style: {_answer_style(persona.answer_style)}",
        f"confidence: {_confidence_style(persona.confidence_style)}",
        f"persistence: {_persistence_style(persona.persistence_style)}",
        f"language: {persona.language}",
    ]
    return "\n".join(lines)


def _answer_style(style: str) -> str:
    return {
        "terse": "very brief",
        "concise": "concise, with only the needed reasoning",
        "hesitant": "tentative and somewhat unsure",
        "verbose": "more detailed than average",
    }.get(style, style)


def _confidence_style(style: str) -> str:
    return {
        "low": "often unsure and likely to ask for help",
        "calibrated": "usually proportional to actual understanding",
        "overconfident": "can sound more certain than warranted",
    }.get(style, style)


def _persistence_style(style: str) -> str:
    return {
        "fragile": "may get discouraged after difficult feedback",
        "steady": "keeps working through feedback",
        "stubborn": "may hold onto initial intuitions",
    }.get(style, style)
