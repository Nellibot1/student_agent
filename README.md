# student_agent

`student_agent` is a reusable independent simulated learner. Its public turn API receives only what a real student could see:

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

## Start here

The easiest way to understand the package is to separate **student behavior** from
**model calls**:

```text
visible tutor text
        |
        v
perception -> private state update -> policy decision -> text generation
                                                        |
                                                        v
                                               visible student reply
```

- **Perception** interprets the tutor's words from the student's point of view.
- **Private state** remembers the learner's confidence, confusion, topics, tasks, and
  misconceptions.
- **Policy** chooses a behavior such as a correct, partial, mistaken, or clarifying reply.
- **Generation** turns that behavior into natural student language.
- **Validation** is an optional final check that the generated wording matches the chosen
  behavior.

`StudentSimulator` coordinates those parts. Your application supplies a perception adapter
and a text-generation adapter. The included LiteLLM adapters are the quickest way to start;
custom adapters let you use any hosted model, local model, or test double.

Choose the path that matches what you want to do:

| Goal | Start with |
| --- | --- |
| Try a realistic student in a terminal | [Quick start](#quick-start), then run `student-agent chat` |
| Verify the package without an API key | Run `python examples/basic_student.py` |
| Embed the student in another application | [Use the Python API](#use-the-python-api) |
| Understand or modify the implementation | [Repository map](#repository-map), beginning with `engine.py` |
| Add a different model provider | [Implement a custom adapter](#implement-a-custom-adapter) |

## Quick start

Requirements: Python 3.11 or newer.

All commands in this README assume that the current directory is the repository root. Create
a virtual environment and install the package, the LiteLLM integration, and development
tools:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,litellm]"
cp .env.example .env
```

Before configuring a model provider, confirm that the core package works locally:

```bash
python examples/basic_student.py
```

This offline example prints a fixed student reply followed by the private observation and
state. It proves that installation and simulator wiring work; it does not test a hosted
model.

Edit `.env` and add the credentials required by your model provider. For example, the
default OpenAI-style setup expects `OPENAI_API_KEY`. Never commit `.env`.

Then start a conversation. Each line you enter at `Tutor:` is treated as a message that a
real student could see:

```bash
student-agent chat --model openai/gpt-4.1-mini --persona luna
```

Useful commands inside the chat are `/state` (inspect the learner's private state) and
`/quit`. Conversations are saved to `data/conversations/` by default.

To check the checkout before changing code:

```bash
pytest
ruff check src tests examples
```

### Use the Python API

The smallest real-model example is:

```python
import asyncio

from student_agent import StudentSimulator, load_builtin_persona
from student_agent.integrations import LiteLLMStudentGenerator, LiteLLMStudentPerception


async def main() -> None:
    model = "openai/gpt-4.1-mini"
    student = StudentSimulator(
        LiteLLMStudentGenerator(model),
        perception=LiteLLMStudentPerception(model),
        persona=load_builtin_persona("luna"),
        seed=42,
    )

    reply = await student.respond("Can you explain what this result means?")
    print(reply.message)


asyncio.run(main())
```

This example makes two model calls per turn: perception and response generation. The CLI
also configures a validator, which may make a third call for risky response types.

For an example that runs without credentials or network access, see
[`examples/basic_student.py`](examples/basic_student.py). It uses small test doubles for
both model boundaries.

### CLI options and persistence

If you installed only the core package, add the optional LiteLLM adapter before using the
CLI with a hosted or local model:

```bash
pip install "student-agent[litellm]"
```

Set the provider credentials required by your selected model, then start a terminal
session:

```bash
cp .env.example .env
# Edit .env and replace the API-key placeholder with a newly generated key.
student-agent chat --persona luna
```

The CLI automatically loads the nearest `.env` file from the current working directory
without overriding variables that are already exported. `.env` is ignored by Git; only
the placeholder-only `.env.example` belongs in source control.

Each line entered at `Tutor:` is the only application input passed into the student turn.
Use `/state` to inspect the local learner snapshot and `/quit` to end the session. To create
a persona on the spot:

```bash
student-agent chat \
  --persona-json '{"id":"careful_beginner","name":"Careful Beginner","answer_style":"hesitant","control":{"mastery":0.3,"confidence":0.25,"help_seeking":0.7}}'
```

Student generation uses a 1000-token completion budget by default so reasoning models have
room for internal reasoning before producing the intentionally concise visible answer. Override
it with `--max-tokens` when a particular provider needs a different limit.
All standalone model calls default to low reasoning effort. Override this for an experiment
with `--reasoning-effort none|low|medium|high|xhigh|max`, or set
`STUDENT_AGENT_REASONING_EFFORT` once in the environment.

For a non-interactive, one-message run:

```bash
student-agent chat \
  --message "Can you explain this result in your own words?" \
  --once
```

Standalone conversations are saved automatically as structured JSON under
`data/conversations/`. Each file contains the visible transcript, persona, private
observations and decisions, evolving student state, model usage, and a resumable snapshot.
Choose an explicit path or disable persistence with:

```bash
student-agent chat --persona luna --save data/conversations/probability.json
student-agent chat --persona luna --no-save
```

Continue a saved learner with its prior persona, state, and visible history:

```bash
student-agent chat --resume data/conversations/probability.json
```

List the packaged personas with `student-agent personas`. You can also set the model once
with the `STUDENT_AGENT_MODEL` environment variable.

The model boundary is intentionally small:

```python
class StudentTextGenerator(Protocol):
    async def generate(self, *, instruction: str, tutor_message: str) -> str: ...
```

The consuming application supplies that adapter, so the package works with an API model,
a local model, or a test double without taking a dependency on any provider SDK.
The optional `student_agent.integrations.LiteLLMStudentGenerator` is supplied for convenience;
LiteLLM is installed only when the `litellm` extra is selected.

The CLI uses semantic, text-only perception. A separate model call interprets the visible
tutor message as structured feedback before the response policy runs. It receives only the
visible conversation, never tutor grades or application state. Semantic adapters leave
temperature unset by default so reasoning models can use their required provider defaults.
Use a cheaper model with `--perception-model`:

```bash
student-agent chat --perception-model openai/gpt-4.1-mini
```

Malformed or failed semantic-perception calls are retried semantically with an adaptive output
budget: the default starts at 800 tokens and grows to 1600 for later attempts. Provider
finish reasons, refusals, token usage, and whether content was present are retained in usage
telemetry. They are never silently replaced with keyword-based guesses.

Interactive chat defaults to `--perception-failure-policy continue`. If all retries are
exhausted, the student preserves its complete learner state, records the turn as
`semantic_degraded`, and asks a clarification question rather than inventing an interpretation.
Use `--perception-failure-policy fail` for strict fail-fast behavior.
The CLI also enables semantic response validation and retries generation when the visible
reply does not match the selected student behavior. By default, `--validation-policy risky`
calls the validator only for partial, incorrect, wrong-concept, and topic-jump replies. Use
`always` for maximum checking or `off` to disable semantic validation. Configure the models
and retries separately with `--validator-model`, `--perception-attempts`, and
`--generation-attempts`.

Interactive chat also defaults to `--validation-failure-policy continue`. If all generation
repairs are exhausted, a non-empty reply that the validator marks safe to expose is committed
with `validation_status="degraded"` and a warning. Empty replies, replies marked unsafe, and
obvious simulator-instruction leakage still fail. The experiment runner uses strict `fail`
behavior by default.

## Repository map

Start with these files when you are new to the implementation:

| Path | Responsibility |
| --- | --- |
| `src/student_agent/engine.py` | Coordinates a complete turn and owns the public `StudentSimulator` API. Read this first. |
| `src/student_agent/perception.py` | Defines the structured interpretation produced from visible tutor text. |
| `src/student_agent/policy.py` | Stores learner state, evolves it, and chooses the next response behavior. |
| `src/student_agent/personas.py` | Defines persona files and loads the built-in personas. |
| `src/student_agent/modes/` | Converts a policy decision into the instruction used to generate student text. |
| `src/student_agent/integrations/` | Optional provider adapters. `litellm.py` implements perception, generation, and validation. |
| `src/student_agent/cli.py` | Wires adapters, persistence, and terminal input together. |
| `src/student_agent/persistence.py` | Saves and restores standalone conversations. |
| `examples/basic_student.py` | Offline, runnable integration example with test doubles. |
| `tests/` | Executable examples of expected behavior and edge cases. |

The dependency direction is important: the core package defines small protocols, while
provider-specific code implements them. Core modules must not import LiteLLM or application
code. This keeps the simulator usable in other repositories.

## Integrate it into another application

An application needs two adapters:

1. A `StudentPerception` implementation that converts visible conversation text into a
   `StudentObservation`.
2. A `StudentTextGenerator` implementation that converts the prepared instruction into one
   visible student message.

The optional `StudentResponseValidator` is useful when an experiment must distinguish, for
example, a deliberately partial answer from a fully correct one. The easiest working
implementation is the bundled LiteLLM integration shown in the quick start.

The normal application loop is deliberately small:

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

Keep one `StudentSimulator` instance for one learner session. Do not share the same instance
between simultaneous conversations, because `respond()` updates its state and history. To
move a learner between processes, serialize `student.snapshot()` and restore it with
`StudentSimulator.from_snapshot(...)`.

### Implement a custom adapter

You do not need to modify the simulator to use another model provider. Implement the two
small async protocols and inject the resulting objects:

1. `StudentPerception.observe(...)` receives the latest tutor text, visible history, and a
   copy of student-owned state. Return a validated `StudentObservation`.
2. `StudentTextGenerator.generate(...)` receives the prepared instruction and latest tutor
   message. Return exactly one visible student reply as a string.
3. Optionally implement `StudentResponseValidator.validate(...)` when your experiment needs
   semantic checking of generated behavior.
4. Keep provider imports in your integration module. Do not add them to `engine.py`,
   `policy.py`, or `perception.py`.
5. Pass only learner-visible text and student-owned memory. Never enrich perception with a
   tutor grade, answer key, lesson plan, or curriculum identifier.
6. Test the adapter with a fixed fake response before making live model calls.

The offline [`examples/basic_student.py`](examples/basic_student.py) shows the complete
shape with small deterministic adapters. The bundled
[`src/student_agent/integrations/litellm.py`](src/student_agent/integrations/litellm.py)
shows a production adapter with structured output validation, retries, token telemetry, and
bounded prompt memory.

### Data ownership rule

Pass only the tutor text that the simulated learner is allowed to see. Do not add answer
keys, grades, curriculum IDs, or a tutor's hidden plan to the perception or generation
inputs. If an experiment needs those values for scoring, evaluate the transcript outside
`StudentSimulator` after the reply has been produced.

## How one turn works

For every call to `respond(tutor_message)`, the simulator performs six steps:

1. **Perception:** an injected `StudentPerception` reads the visible tutor message, the
   student's visible history, and the student's private memory. The CLI supplies
   `LiteLLMStudentPerception`; direct API consumers must inject a semantic perception adapter.
   It estimates the local topic and task, message type, whether a reply is expected, feedback,
   taught propositions, demonstrated understanding, and any specific misconception. When the
   belief is a paraphrase of a relevant existing misconception, perception returns that supplied
   local ID so the state reinforces one canonical record instead of creating another wording copy.
   Exhausted perception can either fail the turn or enter the explicit degraded continuation
   path described above.
2. **Private state update:** confidence-weighted perceived feedback changes the student's
   confidence, confusion, contentful misconceptions, task attempts, fatigue, and motivation.
   Assessment, instruction, and demonstrated understanding have different effects. No external
   grade is consulted.
3. **Policy decision:** the persona, private state, interpretation, and stable seed choose
   a response tendency such as correct, partial, misconception-shaped, clarification, or
   topic drift. Bare negative feedback without a correction, explanation, or new question
   chooses clarification before generic answer sampling; a stubborn learner may instead
   maintain a strong active misconception.
4. **Language generation:** the package builds an instruction containing only information
   owned by the student simulator. The injected text generator produces one natural reply.
5. **Conditional semantic validation:** for risky modes, the configured validator checks that
   the visible reply realizes the intended partial, incorrect, wrong-concept, or topic-jump
   behavior. The same validation result records `observed_mode`, describing what the text
   actually did. A rejected reply is regenerated with a targeted repair hint. `always` and
   `off` policies are also available; skipped validation leaves `observed_mode` unset rather
   than making another model call.
6. **Commit:** after a non-empty reply passes validation, the new private state
   and the tutor/student messages are committed to the learner's memory. A failed turn does
   not mutate the session.

Perceived topics and tasks are not curriculum IDs. The student keeps a local registry and asks
semantic perception to reuse an existing key when wording refers to the same topic or task.
New labels receive stable private keys. Attempts are counted per task rather than across every
question that happens to share a broad topic. The complete registry remains in private state,
but each perception request receives only the current and five most recently relevant topics,
up to ten tasks from the current topic, five other recent tasks, and three aliases per topic.
It also receives at most eight active or recently relevant misconception candidates. This
bounds prompt growth without deleting long-term memory.

Alternative `StudentPerception` implementations may use a local semantic classifier or an
LLM, but must receive only the same student-visible inputs:

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

Validation is internal to the student. A `StudentResponseValidator` checks
whether generated text follows the already selected response tendency—for example, whether
a requested partial answer accidentally became fully correct. If rejected, it may return a
repair hint and the simulator generates again. It does **not** grade the student against a
tutor answer key, discover the real topic, or send data back to the tutor.

An external research evaluator may still grade transcripts after the fact. Keep that
evaluator outside `StudentSimulator`; do not feed its private labels into `respond` if the
goal is a black-box student.

## Offline perception evaluation

Create a JSONL corpus whose lines contain an id, visible tutor message, optional visible
history, and human labels under `expected`:

```json
{"id":"mixed-1","conversation_id":"lesson-1","tutor_message":"Almost; add the integral condition.","expected":{"feedback":"partial","correction_present":true,"explanation_present":false,"topic_group":"density"}}
```

Replay it without exposing tutor internals to perception:

```bash
student-agent evaluate-perception \
  --input data/perception-cases.jsonl \
  --output data/perception-report.json \
  --model openai/gpt-4.1-mini
```

The report includes feedback accuracy, a confusion matrix, confidence Brier score, field-level
accuracy, topic stability, and every structured observation.

## Personas

The six personas migrated from the earlier AI Tutor implementation are included:
`luna`, `matej`, `nika`, `rok`, `stress_zero`, and `tom`.

```python
from student_agent import available_builtin_personas, load_builtin_persona

print(available_builtin_personas())
persona = load_builtin_persona("luna")
```

You can also compile a natural-language learner description into a validated persona. The
command prints the generated controls and saves the concrete result as YAML, so later runs do
not reinterpret the prose:

```bash
student-agent generate-persona \
  "This student is very interested and eager to learn, but cautious when answering." \
  --id interested_cautious \
  --name "Interested, cautious learner" \
  --output personas/interested_cautious.yaml

student-agent chat --persona personas/interested_cautious.yaml
```

`STUDENT_AGENT_PERSONA_MODEL` selects a separate model for this step; otherwise the command
uses `STUDENT_AGENT_MODEL`. Existing output files are protected unless `--force` is supplied.
The generator keeps prior knowledge (`mastery`), learning speed (`learning_rate`), and
willingness to engage (`motivation`) separate. Characteristics not supported by the prose
remain close to the balanced defaults.

The same feature is available through the provider-neutral Python API:

```python
from student_agent import generate_persona_from_description, save_persona
from student_agent.integrations import LiteLLMPersonaGenerator

generator = LiteLLMPersonaGenerator("openai/gpt-4.1-mini")
persona = await generate_persona_from_description(
    "Interested and eager to learn, but cautious when answering.",
    generator=generator,
    persona_id="interested_cautious",
)
save_persona("personas/interested_cautious.yaml", persona)
```

Their behavior controls, descriptions, and styles are preserved. The old
application-specific `prior_student_id` is retained as optional metadata where it existed,
not as a dependency on the prior repository.

You can construct a new profile or load a YAML file from any path:

```python
from student_agent import LearningDynamics, StudentControlProfile, StudentPersona

persona = StudentPersona(
    id="careful_beginner",
    answer_style="hesitant",
    confidence_style="low",
    persistence_style="steady",
    learning_dynamics=LearningDynamics(
        purpose="realistic",
        misconception_resistance=0.65,
        corrections_before_adaptation=3,
    ),
    control=StudentControlProfile(
        mastery=0.30,
        confidence=0.25,
        learning_rate=0.45,
        motivation=0.80,
        misconception_rate=0.40,
        help_seeking=0.60,
        topic_drift=0.02,
    ),
)
```

`learning_dynamics` separates ordinary learner simulation from deliberate tutor stress
testing. A `tutor_stress_test` persona may keep misconceptions longer, while
`corrections_before_adaptation` and `misconception_resistance` still make repeated clear
correction gradually change its behavior instead of locking it into one answer forever.

## Save and restore

Snapshots contain the persona, private state, seed, mode, and visible conversation history.
Model clients and perception adapters are runtime dependencies and are injected again:

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

Snapshot schema 2 belongs to version 0.2 and is intentionally incompatible with the old
metadata-driven schema 1.

## Development workflow

Run commands from the repository root with the virtual environment active:

```bash
pytest                          # complete offline test suite
ruff check src tests examples  # lint and import-order checks
python examples/basic_student.py
python -m build                 # verify the distributable package
```

The unit tests use injected fakes and should not contact a model provider. Add tests when
changing policy probabilities, state evolution, retry behavior, snapshot schemas, or CLI
arguments. A fixed `seed` makes policy decisions reproducible, but model wording can still
vary unless the selected provider/model is deterministic.

When adding a provider integration, implement the small protocols in the core instead of
importing the provider SDK into `engine.py`, `policy.py`, or `perception.py`. When adding a
new state field, also consider snapshot compatibility and whether the field is safe to put
in a generation prompt.

## Troubleshooting

**`chat requires --model`**

Pass `--model <litellm-model-name>` or set `STUDENT_AGENT_MODEL` in `.env`.

**`LiteLLM support is not installed`**

Install the optional integration with `python -m pip install -e ".[litellm]"` from this
checkout, or `python -m pip install "student-agent[litellm]"` for an installed release.

**Authentication or provider errors**

Check that `.env` contains the credential expected by the model provider and that the model
name includes the provider prefix expected by LiteLLM. The CLI loads the nearest `.env`
without replacing environment variables that are already exported.

**The student behaves differently from the persona**

Inspect `/state` and the saved conversation's `observation`, `decision`, and
`validation_status`. The usual distinction is whether perception misunderstood the tutor,
the policy chose an unexpected mode, or generation failed to realize that mode.

**A failed turn changed the learner**

That is a bug. `respond()` is designed to commit state and history only after it has a usable
reply. Capture the smallest failing case and add it to `tests/test_student_agent.py`.

## Boundaries and limitations

The package owns personas, student-visible history, private state evolution, perception,
policy selection, prompt rendering, snapshots, and optional response-policy validation.

The consuming application still owns the actual model adapter, persistence, concurrency,
timeouts, rate limits, and network retries. One `StudentSimulator` is one stateful learner
session; do not call the same instance concurrently.

Semantic perception can still misinterpret ambiguous tutor wording. Such errors remain
student-side interpretation errors: no hidden tutor grade or application state is consulted.
