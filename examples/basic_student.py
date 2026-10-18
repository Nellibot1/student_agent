"""Minimal offline example of embedding a simulated learner.

Run this file from the repository root with ``python examples/basic_student.py``. It uses
fixed test doubles instead of a hosted model, so it needs no API key or network connection.
Replace ``DemoPerception`` and ``DemoGenerator`` with real adapters in an application; the
``StudentSimulator`` setup and call pattern stay the same.
"""

from __future__ import annotations

import asyncio

from collections.abc import Sequence

from student_agent import (
    ConversationMessage,
    MessageType,
    SimulatedStudentState,
    StudentControlProfile,
    StudentObservation,
    StudentPersona,
    StudentSimulator,
)


class DemoGenerator:
    """Small offline test double for the component that writes the final reply."""

    async def generate(self, *, instruction: str, tutor_message: str) -> str:
        # A real generator sends ``instruction`` to a model. This fixed reply is useful for
        # learning the integration shape and for tests that must be deterministic.
        del instruction, tutor_message
        return "I can follow the first step, but I am not sure why the conclusion follows."


class DemoPerception:
    """Small offline test double for the component that reads tutor messages.

    A real integration should interpret the supplied text rather than returning fixed
    values. This explicit test double keeps the example runnable without an API key.
    """

    async def observe(
        self,
        *,
        tutor_message: str,
        history: Sequence[ConversationMessage],
        state: SimulatedStudentState,
    ) -> StudentObservation:
        # A real perception adapter derives these values from the message, visible history,
        # and student-owned state. This example returns a small valid observation directly.
        del history, state
        return StudentObservation(
            tutor_message=tutor_message,
            perceived_topic="reasoning",
            topic_key="reasoning",
            perceived_task="explain the reasoning",
            task_key="explain-reasoning",
            message_type=MessageType.QUESTION,
            response_expected=True,
        )


async def main() -> None:
    # Step 1: describe stable starting tendencies for this kind of learner. All numeric
    # controls use 0 for the lowest tendency and 1 for the highest.
    persona = StudentPersona(
        id="careful_beginner",
        answer_style="hesitant",
        confidence_style="low",
        persistence_style="steady",
        control=StudentControlProfile(
            mastery=0.35,
            confidence=0.30,
            learning_rate=0.45,
            misconception_rate=0.35,
            help_seeking=0.55,
            topic_drift=0.02,
        ),
    )
    # Step 2: create one simulator per learner conversation. The simulator owns behavior
    # and state; the application supplies one adapter that reads tutor text and another
    # that writes student text. The seed makes policy choices reproducible.
    student = StudentSimulator(
        DemoGenerator(),
        perception=DemoPerception(),
        persona=persona,
        seed=42,
    )

    # Step 3: pass only the text visible to the learner. ``respond`` interprets it, updates
    # a provisional state, chooses a response behavior, and generates the visible reply.
    reply = await student.respond("Can you explain the reasoning in your own words?")

    # Applications send ``reply.message`` back to the tutor. Observation and state are
    # private telemetry for researchers and debugging; they should not be shown to a tutor.
    print(f"Student: {reply.message}")
    print(f"Student perceived: {reply.observation.message_type}")
    print(f"Private state: {reply.state.model_dump()}")


if __name__ == "__main__":
    asyncio.run(main())
