# Integration guide

## Required adapters

An application supplies two asynchronous adapters:

1. `StudentPerception.observe(...)` converts visible conversation text into a
   `StudentObservation`.
2. `StudentTextGenerator.generate(...)` converts a prepared instruction into one visible
   student reply.

`StudentResponseValidator` is optional and checks whether generated text follows the
simulator's already-selected behavior.

```python
student = StudentSimulator(
    generator=my_generator,
    perception=my_perception,
    persona=my_persona,
    seed="experiment-7",
)

for tutor_message in visible_tutor_messages:
    reply = await student.respond(tutor_message)
    send_to_tutor(reply.message)
```

Keep one simulator instance per conversation because `respond()` updates private state and
visible history.

## Protocol shapes

```python
class StudentTextGenerator(Protocol):
    async def generate(self, *, instruction: str, tutor_message: str) -> str: ...
```

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

The consuming application can use a hosted model, local model, classifier, or deterministic
test double. Keep provider imports inside the integration implementation.

## LiteLLM adapters

The bundled adapters are the quickest hosted-model implementation:

```python
from student_agent import StudentSimulator, load_builtin_persona
from student_agent.integrations import LiteLLMStudentGenerator, LiteLLMStudentPerception

model = "openai/gpt-4.1-mini"
student = StudentSimulator(
    LiteLLMStudentGenerator(model),
    perception=LiteLLMStudentPerception(model),
    persona=load_builtin_persona("luna"),
)
```

The CLI additionally configures a response validator for risky response modes. A normal
turn therefore uses two model calls, with a possible third validation call.

Reasoning effort is opt-in. Only set it when the selected model supports the parameter.

## Custom adapter checklist

1. Validate caller-controlled input before making a paid request.
2. Pass only learner-visible text and student-owned memory.
3. Return one validated observation or one non-empty student message.
4. Keep retries bounded and distinguish transient failures from configuration errors.
5. Do not retry authentication, permission, not-found, bad-request, or unsupported-parameter
   failures.
6. Inject a fake completion function in tests so unit tests do not contact providers.
7. Keep answer keys, official grades, and lesson plans in external evaluation code.

The bundled LiteLLM implementation demonstrates structured JSON validation, bounded prompt
memory, usage telemetry, retries, and provider diagnostics.

## Save and restore

Snapshots contain the persona, private state, seed, simulation mode, and visible history.
Runtime model clients are injected again after loading:

```python
from student_agent import StudentSimulator, StudentSnapshot

payload = student.snapshot().model_dump_json()
snapshot = StudentSnapshot.model_validate_json(payload)
restored = StudentSimulator.from_snapshot(
    snapshot,
    generator=MyModel(),
    perception=MySemanticPerception(),
)
```

The standalone CLI saves structured conversations under `data/conversations/` unless
`--no-save` is supplied.

## Data ownership

Only tutor text visible to the learner should enter the simulation. Observation and private
state are research telemetry and should not be exposed as tutor-side truth. If an experiment
needs authoritative grading, score the completed transcript outside `StudentSimulator`.
