# student_agent

`student_agent` is a provider- and application-independent simulated learner engine.
It separates deterministic learner behavior from natural-language generation:

1. Python updates learner state and selects an intended response behavior.
2. An application-supplied text generator realizes that behavior as one learner message.
3. The caller receives the message together with the decision and updated state.

The package does not depend on Google ADK, an AI Tutor application, a web framework,
a database, or a particular model provider.

## Install

```bash
pip install student-agent
```

For development:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Minimal example

```python
import asyncio

from student_agent import SimulationContext, StudentSimulator


class MyModel:
    async def generate(self, *, instruction: str, tutor_message: str) -> str:
        # Replace this body with a call to any LLM provider or local model.
        return "I think I understand the first part, but why does the last step follow?"


async def main() -> None:
    student = StudentSimulator(MyModel(), seed=42)
    reply = await student.respond(
        "Can you explain what this result means?",
        context=SimulationContext(
            topic_id="example-topic",
            task_type="explanation",
            expects_answer=True,
            turn_number=1,
        ),
    )
    print(reply.message)
    print(reply.decision.mode)


asyncio.run(main())
```

The generator boundary is intentionally small:

```python
class StudentTextGenerator(Protocol):
    async def generate(self, *, instruction: str, tutor_message: str) -> str:
        ...
```

Provider-specific clients belong in the consuming application. This keeps the core
portable and prevents model SDKs from becoming mandatory dependencies.

## Context supplied by an application

`SimulationContext` is the normalized boundary between an application and the learner:

```python
from student_agent import Grade, SimulationContext

context = SimulationContext(
    topic_id="fractions",
    topic_label="Adding fractions",
    task_type="worked_example",
    learning_objective="Use a common denominator",
    expects_answer=True,
    attempt_number=2,
    turn_number=4,
    previous_grade=Grade.PARTIAL,
)
```

Each application is responsible for mapping its own session, exercise, and grading
objects into this neutral type. Unknown grade strings are rejected instead of being
silently interpreted as incorrect.

## Personas

Construct personas in code or load them from YAML:

```python
from student_agent import StudentControlProfile, StudentPersona

persona = StudentPersona(
    id="careful_beginner",
    answer_style="hesitant",
    confidence_style="low",
    persistence_style="steady",
    control=StudentControlProfile(
        mastery=0.30,
        confidence=0.25,
        learning_rate=0.45,
        misconception_rate=0.40,
        help_seeking=0.60,
        topic_drift=0.02,
    ),
)
```

## Save and restore a learner

Model clients are deliberately not serialized. Save a portable snapshot and inject the
appropriate generator when restoring it:

```python
from student_agent import StudentSimulator, StudentSnapshot

payload = student.snapshot().model_dump_json()

snapshot = StudentSnapshot.model_validate_json(payload)
restored = StudentSimulator.from_snapshot(snapshot, generator=MyModel())
```

The application decides whether to store the JSON in a file, database, cache, or
experiment record.

## Optional validation and retry

Applications can inject a `StudentResponseValidator`. If a generated message does not
match the selected policy behavior, the validator can reject it and provide a repair
hint. Set `max_generation_attempts` above one to enable retries. Validation is optional
because it may require an additional model call.

## Concurrency

`StudentSimulator` is a stateful session object. Create one instance per learner and do
not call the same instance concurrently. Web applications should load a learner
snapshot, process one turn, and save the returned snapshot under an application-level
lock or transaction.

## Package boundaries

The core package owns:

- personas and normalized controls;
- learner state evolution;
- deterministic response-mode selection;
- prompt rendering;
- portable snapshots;
- optional response validation hooks.

Consuming applications own:

- the model-provider adapter;
- grading and mapping to `Grade`;
- conversation-history storage;
- persistence and concurrency control;
- framework-specific integrations;
- timeout, rate-limit, and network retry behavior.
