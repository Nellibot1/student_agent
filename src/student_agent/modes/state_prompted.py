"""State-prompted student mode.

This mode hides tutor internals, but still renders the simulator's own student state into
the prompt. Keep it for experiments where state-prompting is intentional.
"""

from __future__ import annotations

from ..perception import StudentObservation
from ..personas import StudentPersona
from ..policy import (
    PolicyDecision,
    SimulatedStudentState,
    SimulationMode,
)
from .base import PolicyBackedMode, fill_simulator_template

STATE_PROMPTED_RESPONSE_MODE_REFERENCE = """<reply_guidance_reference>
The hidden policy may ask for a solid answer, an incomplete answer, a plausible misconception,
a clarification question, a related aside, or a brief continuation. Follow that guidance
naturally, but never name the category or imply you know the tutor's grading state.
</reply_guidance_reference>"""


class StatePromptedStudentMode(PolicyBackedMode):
    mode = SimulationMode.STATE_PROMPTED_STUDENT

    def render_instruction(
        self,
        prompt_template: str,
        persona: StudentPersona,
        observation: StudentObservation,
        simulated_state: SimulatedStudentState,
        decision: PolicyDecision,
        history_block: str,
    ) -> str:
        return fill_simulator_template(
            prompt_template,
            mode=self.mode,
            persona_block=persona.prompt_block(),
            observation_block=observation.visible_prompt_block(),
            history_block=history_block,
            tutor_message=observation.tutor_message,
            decision_block=decision.realization_hint(),
            simulated_state_block=simulated_state.prompt_block(observation.topic_key),
            response_mode_reference=STATE_PROMPTED_RESPONSE_MODE_REFERENCE,
        )
