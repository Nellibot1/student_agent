# student_agent

`student_agent` simulates a stateful learner for tutoring applications and experiments. A
tutor sends visible text, and the package interprets it from the learner's perspective,
updates private learner state, chooses a response behavior, and generates a natural reply.


## First run from a fresh clone

### 1. Clone exactly one working copy

```bash
git clone https://github.com/Nellibot1/student_agent.git
cd student_agent
```

All commands below must run from this repository root. Avoid downloading a second ZIP copy;
having two folders with the same project makes it easy to edit one and execute the other.

### 2. Check Python and create a virtual environment

Python 3.11 or newer is required:

```bash
python3 --version
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

The terminal prompt should now include `(.venv)`. Confirm the selected interpreter:

```bash
python -c "import sys; print(sys.executable)"
```

It should point inside this repository's `.venv` directory.

### 3. Install the package and model integration

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[litellm]"
```

The editable installation creates the `student-agent` command and keeps it connected to the
source in this checkout.

### 4. Verify the package offline

This command requires no API key and makes no network request:

```bash
python examples/basic_student.py
```

If it prints a student reply and private state, the Python installation is working.

### 5. Configure a model provider

Create the local environment file:

```bash
cp .env.example .env
```

For the default OpenAI model, create an API key in the
[OpenAI API dashboard](https://platform.openai.com/api-keys), then edit `.env`:

```env
OPENAI_API_KEY=your-actual-api-key
STUDENT_AGENT_MODEL=openai/gpt-4.1-mini
```

Important:

- Put the real key only in `.env`, never in `.env.example`.
- `.env` is ignored by Git and must not be committed.
- The model is read from `STUDENT_AGENT_MODEL`, so `--model` is optional.
- A ChatGPT login and an API key are separate configuration concerns; this package calls the
  API through LiteLLM.

### 6. Run the setup check

```bash
student-agent doctor
```

A ready installation reports checks like:

```text
[OK] Python 3.11.7 (requires 3.11+)
[OK] Working directory: /path/to/student_agent
[OK] Environment file: /path/to/student_agent/.env
[OK] LiteLLM installed
[OK] Model: openai/gpt-4.1-mini
[OK] OPENAI_API_KEY is configured
[OK] Built-in persona: luna
Ready to run student-agent.
```

The command never prints the API key. Fix every `[ERROR]` item before continuing.

### 7. Start a conversation

```bash
student-agent chat --persona luna
```

Then enter a tutor message:

```text
Tutor: What is 1 + 1?
Student: 2.
```

Inside the chat, use `/state` to inspect private learner state and `/quit` to stop. Saved
conversations are written to `data/conversations/` by default.


## One-message smoke test

Use a non-interactive request to verify the complete provider path:

```bash
student-agent chat \
  --persona luna \
  --message "What is 1 + 1?" \
  --once \
  --no-save
```

Unlike the offline example, this command makes provider calls and may incur API usage.


## Troubleshooting

### `student-agent: command not found`

Activate `.venv` and reinstall the editable package:

```bash
source .venv/bin/activate
python -m pip install -e ".[litellm]"
```

### `chat requires --model`

Set `STUDENT_AGENT_MODEL` in `.env`, or pass a LiteLLM model explicitly with `--model`.

### `OPENAI_API_KEY is missing` or contains an example value

Edit the `.env` in the path printed by `student-agent doctor`. Do not edit `.env.example`.
Start a new command after saving the file.

### Authentication error

The request reached the provider, but the configured credential was rejected. Verify that
the correct key is in `.env`. Authentication and other non-retryable provider errors stop
immediately instead of being sent repeatedly.

### LiteLLM support is not installed

```bash
python -m pip install -e ".[litellm]"
```

### Unsupported `reasoning_effort`

Reasoning effort is omitted by default because many chat models do not accept it. Only pass
`--reasoning-effort` when the selected model supports that parameter.

### The wrong code seems to run

Check `pwd`, the Python executable, and `which student-agent`. This usually means the IDE and
terminal are using different checkouts or virtual environments.

## Common commands

```bash
# Use a built-in persona
student-agent chat --persona luna

# Do not save the conversation
student-agent chat --persona luna --no-save

# Save to a chosen file
student-agent chat --persona luna --save data/conversations/algebra.json

# Resume a saved learner
student-agent chat --resume data/conversations/algebra.json

# Show every CLI option
student-agent chat --help

# generate persona via prompt
student-agent generate-persona \
  "This student is interested and eager to learn, but cautious when answering." \
  --id interested_cautious \
  --name "Interested, cautious learner" \
  --output personas/interested_cautious.yaml

student-agent chat --persona personas/interested_cautious.yaml
```

## Minimal Python API example

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
    reply = await student.respond("What is 1 + 1?")
    print(reply.message)


asyncio.run(main())
```

The CLI can also use separate models for perception and validation. See
[`student-agent chat --help`](#common-commands) and the integration guide for details.

## Development setup

Contributors should install the development tools as well:

```bash
python -m pip install -e ".[dev,litellm]"
pytest
ruff check src tests examples
python -m build
```

The tests use injected model fakes and should not contact a provider.

## Further documentation

- [Architecture and turn lifecycle](docs/architecture.md)
- [Personas, persistence, and perception evaluation](docs/experiments.md)

The core package remains provider-neutral. LiteLLM is an optional convenience integration;
applications may inject their own perception, generation, and validation adapters.
