"""State-prompted student mode.

This mode hides tutor internals, but still renders the simulator's own student state into
the prompt. Keep it for experiments where state-prompting is intentional.
"""

from __future__ import annotations

from ..personas import StudentPersona
from ..policy import (
    PolicyDecision,
    SimulatedStudentState,
    SimulationContext,
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
        context: SimulationContext,
        simulated_state: SimulatedStudentState,
        decision: PolicyDecision,
    ) -> str:
        return fill_simulator_template(
            prompt_template,
            mode=self.mode,
            persona_block=persona.prompt_block(),
            context_block=context.visible_prompt_block(),
            decision_block=decision.realization_hint(),
            simulated_state_block=simulated_state.prompt_block(context.topic_id),
            response_mode_reference=STATE_PROMPTED_RESPONSE_MODE_REFERENCE,
        )
