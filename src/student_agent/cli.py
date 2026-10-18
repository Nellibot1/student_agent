"""Command-line entry point for chat, persona generation, and perception evaluation.

This module is composition code: it turns command-line options into concrete adapters,
creates or restores a ``StudentSimulator``, and handles terminal input and persistence. The
simulation rules themselves belong in ``engine.py`` and ``policy.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

from .engine import (
    PerceptionFailurePolicy,
    StudentResponseValidator,
    StudentSimulator,
    StudentTextGenerator,
    ValidationFailurePolicy,
    ValidationPolicy,
    ValidationStatus,
)
from .errors import StudentAgentError
from .evaluation import evaluate_perception, load_evaluation_cases, save_evaluation_report
from .integrations import (
    DEFAULT_STUDENT_GENERATION_MAX_TOKENS,
    LiteLLMPersonaGenerator,
    LiteLLMStudentGenerator,
    LiteLLMStudentPerception,
    LiteLLMStudentResponseValidator,
)
from .perception import StudentPerception
from .persistence import (
    SavedStudentTurn,
    default_conversation_path,
    load_conversation,
    new_saved_conversation,
    save_conversation,
)
from .persona_generation import PersonaGenerator, generate_persona_from_description
from .personas import (
    DEFAULT_PERSONA,
    StudentPersona,
    available_builtin_personas,
    load_builtin_persona,
    load_persona,
    save_persona,
)
from .policy import SimulationMode

ReadLine = Callable[[str], str]
WriteLine = Callable[[str], None]
# Most chat models do not accept a reasoning_effort parameter. Keep it opt-in so
# ordinary model names such as openai/gpt-4.1-mini work without extra configuration.
DEFAULT_REASONING_EFFORT = ""
REASONING_EFFORT_CHOICES = ("none", "low", "medium", "high", "xhigh", "max")
API_KEY_PLACEHOLDER_MARKERS = (
    "replace-with",
    "replace_me",
    "replace-me",
    "your-",
    "your-api-key",
    "your-real-key",
    "your-openai-api-key",
    "placeholder",
)


def local_environment_path() -> Path | None:
    """Return the nearest .env file that the CLI would load, if one exists."""
    env_file = find_dotenv(usecwd=True)
    return Path(env_file).resolve() if env_file else None


def load_local_environment() -> bool:
    """Load the nearest .env from the working directory without replacing exported values."""
    # Start at the user's working directory rather than the installed package directory.
    # Exported environment variables win over values in the file (``override=False``).
    env_file = local_environment_path()
    return env_file is not None and load_dotenv(env_file, override=False)


def _looks_like_placeholder(value: str) -> bool:
    normalized = value.strip().strip("\"'").lower()
    return any(marker in normalized for marker in API_KEY_PLACEHOLDER_MARKERS)


def _openai_key_error(model: str) -> str:
    """Return actionable OpenAI credential guidance without exposing the key."""
    if not model.strip().lower().startswith("openai/"):
        return ""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    destination = local_environment_path() or (Path.cwd() / ".env").resolve()
    if not key:
        return f"OPENAI_API_KEY is missing. Add a valid API key to {destination}."
    if _looks_like_placeholder(key):
        return f"OPENAI_API_KEY still contains an example value. Replace it in {destination}."
    return ""


def validate_model_configuration(model: str) -> None:
    """Fail before interactive input when the selected provider cannot be used."""
    if find_spec("litellm") is None:
        raise RuntimeError(
            "LiteLLM support is not installed. Run `python -m pip install -e \".[litellm]\"`."
        )
    credential_error = _openai_key_error(model)
    if credential_error:
        raise ValueError(credential_error)


def doctor_report(model: str) -> tuple[list[str], bool]:
    """Build a secret-safe first-run diagnostic report."""
    lines: list[str] = []
    ready = True

    python_ok = sys.version_info >= (3, 11)
    lines.append(
        f"[{'OK' if python_ok else 'ERROR'}] Python "
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} (requires 3.11+)"
    )
    ready &= python_ok

    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").is_file():
        lines.append(f"[OK] Working directory: {cwd}")
    else:
        lines.append(f"[WARN] Working directory has no pyproject.toml: {cwd}")

    env_path = local_environment_path()
    if env_path is None:
        lines.append("[WARN] No .env file found; exported environment variables will be used")
    else:
        lines.append(f"[OK] Environment file: {env_path}")

    litellm_ok = find_spec("litellm") is not None
    lines.append(f"[{'OK' if litellm_ok else 'ERROR'}] LiteLLM {'installed' if litellm_ok else 'not installed'}")
    ready &= litellm_ok

    selected_model = model.strip()
    if selected_model:
        lines.append(f"[OK] Model: {selected_model}")
    else:
        lines.append("[ERROR] No model configured; set STUDENT_AGENT_MODEL in .env")
        ready = False

    credential_error = _openai_key_error(selected_model)
    if credential_error:
        lines.append(f"[ERROR] {credential_error}")
        ready = False
    elif selected_model.lower().startswith("openai/"):
        lines.append("[OK] OPENAI_API_KEY is configured")
    elif selected_model:
        lines.append("[OK] Provider credentials are delegated to LiteLLM")

    try:
        load_builtin_persona("luna")
    except (OSError, ValueError) as exc:
        lines.append(f"[ERROR] Built-in persona 'luna' could not be loaded: {exc}")
        ready = False
    else:
        lines.append("[OK] Built-in persona: luna")

    lines.append("Ready to run student-agent." if ready else "Setup is incomplete; fix the ERROR items above.")
    return lines, ready


def select_persona(value: str, persona_json: str) -> StudentPersona:
    """Resolve the CLI's mutually exclusive persona inputs in priority order."""
    if persona_json:
        return StudentPersona.model_validate_json(persona_json)
    if value in {"", "default", "balanced"}:
        return DEFAULT_PERSONA.model_copy(deep=True)
    path = Path(value)
    if path.is_file():
        return load_persona(path)
    return load_builtin_persona(value)


async def run_chat(
    args: argparse.Namespace,
    *,
    generator: StudentTextGenerator | None = None,
    perception: StudentPerception | None = None,
    validator: StudentResponseValidator | None = None,
    read_line: ReadLine = input,
    write_line: WriteLine = print,
) -> StudentSimulator:
    """Run a terminal conversation and return the resulting learner session.

    Optional injected adapters make this function easy to test without network calls.
    In normal CLI use they are omitted and the LiteLLM adapters are created here.
    """
    # A resumed file supplies defaults for models and learner data. Explicit command-line
    # values still take precedence, allowing a saved learner to continue with new adapters.
    saved = load_conversation(args.resume) if args.resume else None
    model = args.model or (saved.model if saved is not None else "")
    if not model:
        raise ValueError("chat requires --model or the STUDENT_AGENT_MODEL environment variable")
    # Tests inject small offline adapters. Normal CLI use constructs the corresponding
    # LiteLLM implementations from the parsed options.
    selected_generator = generator or LiteLLMStudentGenerator(
        model,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    perception_model = args.perception_model or (saved.perception_model if saved is not None else "") or model
    selected_perception = perception or LiteLLMStudentPerception(
        perception_model,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.perception_max_tokens,
        timeout=args.perception_timeout,
        max_attempts=args.perception_attempts,
    )
    validator_model = args.validator_model or (saved.validator_model if saved is not None else "") or model
    selected_validator = validator or LiteLLMStudentResponseValidator(
        validator_model,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.validation_max_tokens,
        timeout=args.validation_timeout,
    )
    if saved is None:
        # A new conversation gets a fresh state derived from the selected persona.
        persona = select_persona(args.persona, args.persona_json)
        student = StudentSimulator(
            selected_generator,
            persona=persona,
            mode=SimulationMode(args.mode),
            perception=selected_perception,
            seed=args.seed,
            validator=selected_validator,
            validation_policy=args.validation_policy,
            perception_failure_policy=args.perception_failure_policy,
            validation_failure_policy=args.validation_failure_policy,
            max_generation_attempts=args.generation_attempts,
        )
        turns: list[SavedStudentTurn] = []
    else:
        # Snapshots store data, not live model clients. Reinject the current runtime
        # adapters while restoring the persona, state, seed, and visible history.
        student = StudentSimulator.from_snapshot(
            saved.snapshot,
            generator=selected_generator,
            perception=selected_perception,
            validator=selected_validator,
            validation_policy=args.validation_policy,
            perception_failure_policy=args.perception_failure_policy,
            validation_failure_policy=args.validation_failure_policy,
            max_generation_attempts=args.generation_attempts,
        )
        persona = student.persona
        turns = [turn.model_copy(deep=True) for turn in saved.turns]

    save_path = _conversation_path(args, persona.id)

    def persist() -> None:
        """Checkpoint after each turn so an interrupted chat can be resumed."""
        if save_path is None:
            return
        save_conversation(
            save_path,
            new_saved_conversation(
                model=model,
                perception_model=perception_model,
                validator_model=validator_model,
                snapshot=student.snapshot(),
                turns=turns,
                generation_usage=_generation_usage(selected_generator),
                perception_usage=_generation_usage(selected_perception),
                validation_usage=_generation_usage(selected_validator),
            ),
        )

    write_line(f"Student persona: {persona.name or persona.id} ({persona.id})")
    if saved is not None:
        write_line(f"Resumed conversation: {Path(args.resume).expanduser().resolve()}")
    write_line("Enter a tutor message. Use /state for private state or /quit to stop.")

    # ``--message`` feeds the same loop as interactive input. Keeping one path ensures
    # one-shot and interactive turns have identical state, validation, and persistence.
    pending_message = args.message.strip()
    try:
        while True:
            if pending_message:
                tutor_message = pending_message
                pending_message = ""
                write_line(f"Tutor: {tutor_message}")
            else:
                try:
                    tutor_message = (await asyncio.to_thread(read_line, "Tutor: ")).strip()
                except (EOFError, KeyboardInterrupt):
                    write_line("")
                    break

            if not tutor_message:
                continue
            if tutor_message.lower() in {"/quit", "/exit"}:
                break
            if tutor_message.lower() == "/state":
                write_line(json.dumps(student.snapshot().model_dump(mode="json"), indent=2, ensure_ascii=False))
                continue

            reply = await student.respond(tutor_message)
            turns.append(SavedStudentTurn.from_reply(reply))
            persist()
            if reply.observation.perception_source == "semantic_degraded":
                write_line(
                    "[warning] Semantic perception failed; learner state was preserved and the student "
                    "continued by asking for clarification."
                )
            if reply.validation_status is ValidationStatus.DEGRADED:
                write_line(
                    "[warning] Response validation remained unresolved; the final safe, non-empty student "
                    "reply was kept so the interactive conversation could continue."
                )
            write_line(f"Student: {reply.message}")
            if args.once:
                break
    finally:
        persist()

    if save_path is not None:
        write_line(f"Conversation saved to: {save_path}")

    return student


def _conversation_path(args: argparse.Namespace, persona_id: str) -> Path | None:
    """Resolve no-save, explicit, resumed, and automatically generated save locations."""
    if args.no_save:
        return None
    if args.save is not None:
        return args.save.expanduser().resolve()
    if args.resume:
        return Path(args.resume).expanduser().resolve()
    return default_conversation_path(persona_id).resolve()


def _generation_usage(generator: StudentTextGenerator) -> list[dict[str, Any]]:
    """Convert adapter-specific usage records into JSON-compatible dictionaries."""
    serialized: list[dict[str, Any]] = []
    for item in getattr(generator, "usage", ()):
        if is_dataclass(item) and not isinstance(item, type):
            serialized.append(asdict(item))
        elif hasattr(item, "model_dump"):
            serialized.append(item.model_dump(mode="json"))
        elif isinstance(item, dict):
            serialized.append(dict(item))
    return serialized


async def run_perception_evaluation(args: argparse.Namespace) -> Path:
    """Replay a JSONL corpus through semantic perception and save metrics."""
    perception = LiteLLMStudentPerception(
        args.model,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_attempts=args.attempts,
    )
    report = await evaluate_perception(perception, load_evaluation_cases(args.input))
    return save_evaluation_report(args.output, report)


async def run_persona_generation(
    args: argparse.Namespace,
    *,
    generator: PersonaGenerator | None = None,
) -> tuple[Path, StudentPersona]:
    """Generate and persist one persona for later deterministic reuse."""
    selected_generator = generator or LiteLLMPersonaGenerator(
        args.model,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_attempts=args.attempts,
    )
    persona = await generate_persona_from_description(
        args.description,
        generator=selected_generator,
        persona_id=args.id,
        name=args.name,
        language=args.language,
    )
    destination = save_persona(args.output, persona, overwrite=args.force)
    return destination, persona


def build_parser() -> argparse.ArgumentParser:
    """Define all commands in one place so help text and runtime parsing stay aligned."""
    parser = argparse.ArgumentParser(prog="student-agent", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check a first-run installation without exposing secrets")
    doctor.add_argument(
        "--model",
        default=os.environ.get("STUDENT_AGENT_MODEL", ""),
        help="LiteLLM model name; defaults to STUDENT_AGENT_MODEL",
    )

    chat = subparsers.add_parser("chat", help="Chat with one simulated student")
    persona_group = chat.add_mutually_exclusive_group()
    persona_group.add_argument(
        "--persona",
        default="default",
        help="Built-in persona ID, persona YAML path, or 'default'",
    )
    persona_group.add_argument("--persona-json", default="", help="Inline StudentPersona JSON")
    persona_group.add_argument("--resume", type=Path, help="Resume a saved standalone conversation JSON file")
    chat.add_argument(
        "--model",
        default=os.environ.get("STUDENT_AGENT_MODEL", ""),
        help="LiteLLM model name; defaults to STUDENT_AGENT_MODEL",
    )
    chat.add_argument("--seed", default="42")
    chat.add_argument(
        "--mode",
        choices=[item.value for item in SimulationMode],
        default=SimulationMode.HIDDEN_STATE_STUDENT.value,
    )
    chat.add_argument("--temperature", type=float)
    _add_reasoning_effort_argument(chat)
    chat.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_STUDENT_GENERATION_MAX_TOKENS,
        help=f"Student generation token budget (default: {DEFAULT_STUDENT_GENERATION_MAX_TOKENS})",
    )
    chat.add_argument("--timeout", type=float, default=30.0)
    chat.add_argument(
        "--perception-model",
        default=os.environ.get("STUDENT_AGENT_PERCEPTION_MODEL", ""),
        help="LiteLLM model for semantic perception; defaults to the student response model",
    )
    chat.add_argument("--perception-max-tokens", type=int, default=800)
    chat.add_argument("--perception-timeout", type=float, default=20.0)
    chat.add_argument("--perception-attempts", type=int, default=3)
    chat.add_argument(
        "--perception-failure-policy",
        choices=[item.value for item in PerceptionFailurePolicy],
        default=PerceptionFailurePolicy.CONTINUE.value,
        help="Continue conservatively after exhausted perception retries (default), or fail the turn",
    )
    chat.add_argument(
        "--validator-model",
        default=os.environ.get("STUDENT_AGENT_VALIDATOR_MODEL", ""),
        help="LiteLLM model for semantic response validation; defaults to the response model",
    )
    chat.add_argument("--validation-max-tokens", type=int, default=250)
    chat.add_argument("--validation-timeout", type=float, default=20.0)
    chat.add_argument(
        "--validation-policy",
        choices=[item.value for item in ValidationPolicy],
        default=ValidationPolicy.RISKY.value,
        help="Validate every reply, only risky reply modes (default), or no replies",
    )
    chat.add_argument(
        "--validation-failure-policy",
        choices=[item.value for item in ValidationFailurePolicy],
        default=ValidationFailurePolicy.CONTINUE.value,
        help="Keep a safe non-empty reply after exhausted validation retries (default), or fail the turn",
    )
    chat.add_argument("--generation-attempts", type=int, default=2)
    chat.add_argument("--message", default="", help="Process this tutor message before prompting")
    chat.add_argument("--once", action="store_true", help="Exit after the first student response")
    save_group = chat.add_mutually_exclusive_group()
    save_group.add_argument(
        "--save",
        type=Path,
        help="Save to this JSON path instead of data/conversations/<persona>-<timestamp>.json",
    )
    save_group.add_argument("--no-save", action="store_true", help="Do not persist this conversation")

    subparsers.add_parser("personas", help="List packaged persona IDs")
    generate_persona = subparsers.add_parser(
        "generate-persona",
        help="Create a reusable persona YAML from a natural-language learner description",
    )
    generate_persona.add_argument("description", help="Natural-language description of the learner")
    generate_persona.add_argument("--id", default="generated_student", help="Stable persona identifier")
    generate_persona.add_argument("--name", default="", help="Display name; the model suggests one when omitted")
    generate_persona.add_argument("--language", default="match_tutor")
    generate_persona.add_argument("--output", type=Path, required=True, help="Destination YAML file")
    generate_persona.add_argument("--force", action="store_true", help="Replace an existing output file")
    generate_persona.add_argument(
        "--model",
        default=os.environ.get("STUDENT_AGENT_PERSONA_MODEL") or os.environ.get("STUDENT_AGENT_MODEL", ""),
        help="LiteLLM model; defaults to STUDENT_AGENT_PERSONA_MODEL or STUDENT_AGENT_MODEL",
    )
    generate_persona.add_argument("--temperature", type=float)
    _add_reasoning_effort_argument(generate_persona)
    generate_persona.add_argument("--max-tokens", type=int, default=900)
    generate_persona.add_argument("--timeout", type=float, default=30.0)
    generate_persona.add_argument("--attempts", type=int, default=3)
    evaluate = subparsers.add_parser("evaluate-perception", help="Replay a labeled JSONL perception corpus")
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument(
        "--model",
        default=os.environ.get("STUDENT_AGENT_PERCEPTION_MODEL") or os.environ.get("STUDENT_AGENT_MODEL", ""),
    )
    evaluate.add_argument("--max-tokens", type=int, default=800)
    _add_reasoning_effort_argument(evaluate)
    evaluate.add_argument("--timeout", type=float, default=20.0)
    evaluate.add_argument("--attempts", type=int, default=3)
    return parser


def _add_reasoning_effort_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        default=os.environ.get("STUDENT_AGENT_REASONING_EFFORT", DEFAULT_REASONING_EFFORT),
        help=(
            "Reasoning effort shared by model calls; omitted by default or read from "
            "STUDENT_AGENT_REASONING_EFFORT"
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the selected async workflow, and return a shell exit code."""
    load_local_environment()
    parser = build_parser()
    args = parser.parse_args(argv)
    # Handle commands with their own arguments first. If neither branch matches, the
    # parser's required subcommand guarantees that the remaining command is ``chat``.
    if args.command == "doctor":
        lines, ready = doctor_report(args.model)
        print("\n".join(lines))
        return 0 if ready else 1
    if args.command == "personas":
        print("\n".join(available_builtin_personas()))
        return 0
    if args.command == "generate-persona":
        if not args.model:
            parser.error("generate-persona requires --model or STUDENT_AGENT_PERSONA_MODEL")
        if args.max_tokens < 1 or args.timeout <= 0 or args.attempts < 1:
            parser.error("persona generation token, timeout, and attempt values must be positive")
        try:
            validate_model_configuration(args.model)
            destination, persona = asyncio.run(run_persona_generation(args))
        except (OSError, RuntimeError, StudentAgentError, ValueError) as exc:
            print(f"student-agent: error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(persona.model_dump(mode="json"), indent=2, ensure_ascii=False))
        print(f"Persona saved to: {destination.resolve()}")
        return 0
    if args.command == "evaluate-perception":
        if not args.model:
            parser.error("evaluate-perception requires --model or STUDENT_AGENT_PERCEPTION_MODEL")
        if args.max_tokens < 1 or args.timeout <= 0 or args.attempts < 1:
            parser.error("evaluation token, timeout, and attempt values must be positive")
        try:
            validate_model_configuration(args.model)
            destination = asyncio.run(run_perception_evaluation(args))
        except (OSError, RuntimeError, StudentAgentError, ValueError) as exc:
            print(f"student-agent: error: {exc}", file=sys.stderr)
            return 1
        print(f"Evaluation report: {destination}")
        return 0

    if not args.model and not args.resume:
        parser.error("chat requires --model or the STUDENT_AGENT_MODEL environment variable")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.perception_max_tokens < 1:
        parser.error("--perception-max-tokens must be at least 1")
    if args.perception_timeout <= 0:
        parser.error("--perception-timeout must be positive")
    if args.perception_attempts < 1:
        parser.error("--perception-attempts must be at least 1")
    if args.validation_max_tokens < 1:
        parser.error("--validation-max-tokens must be at least 1")
    if args.validation_timeout <= 0:
        parser.error("--validation-timeout must be positive")
    if args.generation_attempts < 1:
        parser.error("--generation-attempts must be at least 1")
    if args.once and not args.message:
        parser.error("--once requires --message")

    try:
        if args.model:
            validate_model_configuration(args.model)
        asyncio.run(run_chat(args))
    except (OSError, RuntimeError, StudentAgentError, ValueError) as exc:
        print(f"student-agent: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
