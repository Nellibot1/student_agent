# Personas, persistence, and evaluation

## Built-in personas

The package includes `luna`, `matej`, `nika`, `rok`, `stress_zero`, and `tom`:

```bash
student-agent personas
student-agent chat --persona luna
```

Python callers can load the same files:

```python
from student_agent import available_builtin_personas, load_builtin_persona

print(available_builtin_personas())
persona = load_builtin_persona("luna")
```

## Generate a persona

Natural-language persona generation materializes a reusable YAML file:

```bash
student-agent generate-persona \
  "This student is interested and eager to learn, but cautious when answering." \
  --id interested_cautious \
  --name "Interested, cautious learner" \
  --output personas/interested_cautious.yaml

student-agent chat --persona personas/interested_cautious.yaml
```

`STUDENT_AGENT_PERSONA_MODEL` can select a separate model for this command; otherwise it uses
`STUDENT_AGENT_MODEL`. Existing output files are protected unless `--force` is supplied.

Persona controls keep prior knowledge (`mastery`), learning speed (`learning_rate`), and
willingness to engage (`motivation`) separate. `learning_dynamics` distinguishes ordinary
learner simulation from deliberate tutor stress testing.

## Persistence

Standalone conversations save the visible transcript, persona, private observations,
decisions, evolving state, model usage, and a resumable snapshot.

```bash
student-agent chat --persona luna --save data/conversations/probability.json
student-agent chat --resume data/conversations/probability.json
student-agent chat --persona luna --no-save
```

## Offline perception evaluation

Create a JSONL corpus whose rows contain visible tutor input and expected human labels:

```json
{"id":"mixed-1","conversation_id":"lesson-1","tutor_message":"Almost; add the integral condition.","expected":{"feedback":"partial","correction_present":true,"explanation_present":false,"topic_group":"density"}}
```

Replay it with:

```bash
student-agent evaluate-perception \
  --input data/perception-cases.jsonl \
  --output data/perception-report.json \
  --model openai/gpt-4.1-mini
```

The report contains feedback accuracy, a confusion matrix, confidence Brier score,
field-level accuracy, topic stability, and the structured observations.

Evaluation labels belong in this external workflow. They must not be passed into live
student perception as hidden truth.
