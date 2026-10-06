# student_agent

`student_agent` is a reusable simulated learner that is independent of the AI Tutor
repository. Its public turn API receives only what a real student could see:

```python
reply = await student.respond("That is almost correct. Can you try again?")
```

There is no tutor-owned `topic_id`, task type, expected-answer flag, grade, lesson plan,
database object, or application session passed into `respond`. The student keeps its own
visible conversation history and forms a fallible interpretation of the tutor's words.

The package does not import Google ADK, the AI Tutor application, a web framework, a
database, OpenAI, Anthropic, or another model SDK. It uses two normal installable runtime
libraries: Pydantic for validated data models and PyYAML for persona files. “Standalone”
means the package can be installed and used outside the original repository; it does not
mean that all code must be Python-standard-library-only.

## Install and run

```bash
pip install student-agent
```

For local development:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Minimal example:

```python
import asyncio

from student_agent import StudentSimulator, load_builtin_persona


class MyModel:
    async def generate(self, *, instruction: str, tutor_message: str) -> str:
        # Send `instruction` to any hosted or local language model.
        return "I think I understand the first part, but why does the last step follow?"


async def main() -> None:
    student = StudentSimulator(
        MyModel(),
        persona=load_builtin_persona("luna"),
        seed=42,
    )

    reply = await student.respond("Can you explain what this result means?")
    print(reply.message)


asyncio.run(main())
```

The model boundary is intentionally small:

```python
class StudentTextGenerator(Protocol):
    async def generate(self, *, instruction: str, tutor_message: str) -> str: ...
```

The consuming application supplies that adapter, so the package works with an API model,
a local model, or a test double without taking a dependency on any provider SDK.

## How one turn works

For every call to `respond(tutor_message)`, the simulator performs five steps:

1. **Perception:** `RuleBasedPerception` reads the visible tutor message, the student's
   visible history, and the student's private memory. It guesses the topic, whether the
   message looks like a question, explanation, hint, instruction, or feedback, whether a
   reply is expected, and whether visible wording sounds correct/partial/incorrect.
2. **Private state update:** the inferred feedback changes the student's own mastery,
   confidence, confusion, misconceptions, attempts, fatigue, and motivation. No external
   grade is consulted.
3. **Policy decision:** the persona, private state, interpretation, and stable seed choose
   a response tendency such as correct, partial, misconception-shaped, clarification, or
   topic drift.
4. **Language generation:** the package builds an instruction containing only information
   owned by the student simulator. The injected text generator produces one natural reply.
5. **Commit:** after a non-empty reply passes optional validation, the new private state
   and the tutor/student messages are committed to the learner's memory. A failed turn does
   not mutate the session.

The default topic is not a curriculum ID. On the first message, the perception component
extracts a few meaningful words and hashes that local label into a stable private
`topic_key`. Later turns normally remain on that topic until visible transition language
such as “new topic” or “moving on to” appears. This is deliberately imperfect, just as a
student may misunderstand what topic a tutor is testing.

Likewise, `message_type` is inferred from surface wording. A question mark or phrases such
as “can you” and “explain” suggest a question. “Almost” or “not quite” suggests partial
feedback. “Correct” or “well done” suggests positive feedback. These are heuristics, not
hidden truth from the tutoring system.

For stronger semantic interpretation, inject another implementation of
`StudentPerception`. It may use a small local classifier or an LLM, but it must receive only
the same student-visible inputs:

```python
class StudentPerception(Protocol):
    async def observe(
        self,
        *,
        tutor_message: str,
        history: Sequence[ConversationMessage],
        state: SimulatedStudentState,
    ) -> StudentObservation: ...
```

This extension improves interpretation without coupling the student to a tutor database
or experiment runner.

## What validation means

Validation is optional and internal to the student. A `StudentResponseValidator` checks
whether generated text follows the already selected response tendency—for example, whether
a requested partial answer accidentally became fully correct. If rejected, it may return a
repair hint and the simulator generates again. It does **not** grade the student against a
tutor answer key, discover the real topic, or send data back to the tutor.

An external research evaluator may still grade transcripts after the fact. Keep that
evaluator outside `StudentSimulator`; do not feed its private labels into `respond` if the
goal is a black-box student.

## Personas

The six personas migrated from the earlier AI Tutor implementation are included:
`luna`, `matej`, `nika`, `rok`, `stress_zero`, and `tom`.

```python
from student_agent import available_builtin_personas, load_builtin_persona

print(available_builtin_personas())
persona = load_builtin_persona("luna")
```

Their behavior controls, descriptions, styles, and traits are preserved. The old
application-specific `prior_student_id` is retained as optional metadata where it existed,
not as a dependency on the prior repository.

You can construct a new profile or load a YAML file from any path:

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

## Save and restore

Snapshots contain the persona, private state, seed, mode, and visible conversation history.
Model clients and perception adapters are runtime dependencies and are injected again:

```python
from student_agent import StudentSimulator, StudentSnapshot

payload = student.snapshot().model_dump_json()
snapshot = StudentSnapshot.model_validate_json(payload)
restored = StudentSimulator.from_snapshot(snapshot, generator=MyModel())
```

Snapshot schema 2 belongs to version 0.2 and is intentionally incompatible with the old
metadata-driven schema 1.

## Boundaries and limitations

The package owns personas, student-visible history, private state evolution, perception,
policy selection, prompt rendering, snapshots, and optional response-policy validation.

The consuming application still owns the actual model adapter, persistence, concurrency,
timeouts, rate limits, and network retries. One `StudentSimulator` is one stateful learner
session; do not call the same instance concurrently.

Because the default perception uses lightweight rules, subtle or indirect feedback can be
misclassified, topic changes can be missed, and different topics with similar wording can
be grouped together. Those mistakes preserve isolation but reduce simulation accuracy. Use
a custom text-only perception adapter when semantic accuracy matters more than a zero-model
default.
