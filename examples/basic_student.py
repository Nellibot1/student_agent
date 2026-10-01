"""Run a simulated learner without any application or model-provider dependency."""

from __future__ import annotations

import asyncio

from student_agent import (
    Grade,
    SimulationContext,
    StudentControlProfile,
    StudentPersona,
    StudentSimulator,
)


class DemoGenerator:
    """Replace this class with an adapter for the model used by your application."""

    async def generate(self, *, instruction: str, tutor_message: str) -> str:
        del instruction, tutor_message
        return "I can follow the first step, but I am not sure why the conclusion follows."


async def main() -> None:
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
    student = StudentSimulator(DemoGenerator(), persona=persona, seed=42)

    reply = await student.respond(
        "Can you explain the reasoning in your own words?",
        context=SimulationContext(
            topic_id="example-topic",
            task_type="explanation",
            expects_answer=True,
            turn_number=1,
            previous_grade=Grade.UNGRADED,
        ),
    )

    print(f"Student: {reply.message}")
    print(f"Intended behavior: {reply.decision.mode}")
    print(f"State: {reply.state.model_dump()}")


if __name__ == "__main__":
    asyncio.run(main())
