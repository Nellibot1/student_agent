# Architecture

## Processing pipeline

`StudentSimulator` coordinates a learner-owned processing pipeline:

```text
visible tutor text
        |
        v
perception -> private state update -> policy decision -> text generation
                                                        |
                                                        v
                                               visible student reply
```

- Perception interprets visible tutor text from the student's point of view.
- Private state tracks confidence, confusion, topics, tasks, and misconceptions.
- Policy chooses a behavior such as a correct, partial, mistaken, or clarifying response.
- Generation writes that behavior as natural student language.
- Optional validation checks that the wording realizes the selected behavior.

The simulator never receives a tutor's hidden answer key, grade, lesson plan, or curriculum
identifier. Those values belong in an external evaluator, not in learner perception.

## One turn

Every `respond(tutor_message)` call performs these steps:

1. An injected `StudentPerception` reads the tutor message, visible history, and student-owned
   memory and returns a validated `StudentObservation`.
2. The engine computes a provisional private-state update.
3. Deterministic policy combines that state, the persona, and the experiment seed to choose
   a `StudentResponseMode`.
4. The active simulation mode renders an instruction for the text generator.
5. For risky modes, an optional semantic validator checks whether the generated wording
   matches the selected behavior and can request a repair.
6. State and visible messages are committed only after a usable reply exists.

A failed turn does not mutate learner state or conversation history.

## Failure behavior

Semantic perception and validation can retry malformed model output and transient transport
failures. Authentication, permission, unsupported-parameter, and other configuration errors
are non-retryable and stop immediately.

Interactive chat normally continues conservatively after exhausted semantic perception: it
preserves learner state and asks for clarification. Strict integrations can select the
fail-fast policy. Validation has a similar explicit degraded-continuation policy for safe,
non-empty candidate replies.

## Memory boundaries

The simulator stores complete learner memory locally but sends a bounded relevant selection
to semantic perception. Topics and tasks use stable learner-local keys so paraphrases can
reuse existing memory without depending on curriculum IDs.

One `StudentSimulator` instance represents one learner conversation. Do not share an instance
between concurrent sessions. Use snapshots to move a learner between processes.

## Repository map

| Path | Responsibility |
| --- | --- |
| `src/student_agent/engine.py` | Coordinates complete turns and owns the public simulator API. |
| `src/student_agent/perception.py` | Structured learner interpretation of tutor text. |
| `src/student_agent/policy.py` | Learner state evolution and response-mode selection. |
| `src/student_agent/personas.py` | Persona models, files, and built-in loading. |
| `src/student_agent/modes/` | Converts policy decisions into generation instructions. |
| `src/student_agent/integrations/` | Optional provider adapters, including LiteLLM. |
| `src/student_agent/cli.py` | CLI setup, diagnostics, persistence, and terminal input. |
| `src/student_agent/persistence.py` | Conversation snapshots and restore support. |
| `examples/basic_student.py` | Offline example with deterministic test doubles. |
| `tests/` | Expected behavior and regression tests. |

Core modules define small provider-neutral protocols. Provider SDK imports belong in
integration modules, not in the engine, policy, or perception layers.

## Validation scope

Internal validation asks whether a generated reply follows the simulator's selected behavior.
For example, it can detect that a requested partial answer accidentally became fully correct.
It does not grade against an answer key or discover an authoritative topic.

Research scoring should evaluate completed transcripts outside the simulator. Feeding hidden
evaluation labels back into `respond()` would violate the student-visible boundary.
