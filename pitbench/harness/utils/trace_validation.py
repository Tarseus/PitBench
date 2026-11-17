"""Evidence checks for recorded operations, separate from benchmark metrics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from pitbench.harness.utils.agent_trace import read_trace


def inspect_trace(path: Path) -> dict:
    counts = Counter()
    calls = {}
    states = set()
    gaps = []
    execution_finished = False
    for event in read_trace(path):
        kind = event["event"]
        counts[kind] += 1
        data = event["data"]
        if kind == "repository.checkpoint":
            states.add(data["state_id"])
        if kind == "execution.finished":
            execution_finished = True
        if kind == "collection.gap":
            gaps.append(data)
        group, _, phase = kind.partition(".")
        if group not in {"tool", "command", "terminal", "mcp"} or phase not in {
            "started",
            "result",
            "failed",
            "finished",
        }:
            continue
        call_id = event["call_id"]
        if call_id is None:
            gaps.append(
                {
                    "source": "operation",
                    "error": "missing_call_id",
                    "event_id": event["event_id"],
                }
            )
            continue
        call = calls.setdefault(call_id, {"call_id": call_id, "kind": group})
        call[phase] = data
    operations = []
    for call in calls.values():
        missing = []
        if "started" not in call:
            missing.append("request")
        if "finished" not in call:
            missing.append("finish")
        if "failed" not in call and (
            "result" not in call or call["result"].get("result_present") is False
        ):
            missing.append("result")
        finished = call.get("finished", {})
        for field in ("before_state_id", "after_state_id"):
            if finished.get(field) not in states:
                missing.append(field)
        if finished.get("missing_request"):
            missing.append("native_request")
        if finished.get("correlation") in {
            "unique_arguments_without_provider_id",
            "unresolved",
        }:
            missing.append("provider_call_id")
        operations.append(
            {
                "call_id": call["call_id"],
                "kind": call["kind"],
                "name": call.get("started", {}).get("name"),
                "producer": call.get("started", {}).get("producer", "pitbench"),
                "backend_call_id": finished.get("backend_call_id"),
                "missing": missing,
            }
        )
    return {
        "schema_version": 1,
        "scope": "recorded operations only; does not prove unexposed native activity absent",
        "execution_finished": execution_finished,
        "observed_operations_paired": bool(operations)
        and all(not op["missing"] for op in operations),
        "native_hooks_observed": counts["native.hook"] > 0,
        "event_counts": dict(counts),
        "collection_gaps": gaps,
        "operations": operations,
    }


def run_trace_smoke(args) -> int:
    """Execute and validate an isolated Agent tracing smoke test."""
    import asyncio
    import io
    import json
    import tarfile
    import uuid
    from types import SimpleNamespace

    import docker
    from pitbench.cli.evaluate_config import EvaluationConfig
    from pitbench.harness.agents import AgentFactory, AgentName
    from pitbench.harness.harness.harness import Harness
    from pitbench.harness.terminal.docker_compose_manager import DockerComposeManager
    from pitbench.harness.terminal.tmux_session import TmuxSession

    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("sessions", "agent-logs"):
        (args.output / name).mkdir()
    options = {
        key: value
        for key, value in EvaluationConfig.from_yaml(args.config)
        .kwargs_for(args.agent)
        .items()
        if value is not None
    }
    for argument in args.agent_kwarg:
        key, value = argument.split("=", 1)
        try:
            options[key] = json.loads(value)
        except json.JSONDecodeError:
            options[key] = value
    if args.model:
        options["model_name"] = args.model
    options["timeout_sec"] = args.timeout
    workdir = "/tmp/pitbench-trace-check"
    options["container_workdir"] = workdir
    report = {
        "agent": args.agent,
        "model": options.get("model_name"),
        "status": "started",
    }
    client = docker.from_env()
    nested = options.get("runner_backend") == "workspace"
    container = client.containers.run(
        args.image,
        ["sleep", "infinity"],
        detach=True,
        network_mode="none",
        user=args.user,
        name=f"pitbench-trace-check-{uuid.uuid4().hex[:12]}",
        **(DockerComposeManager.nested_sandbox_options() if nested else {}),
        volumes={
            str((args.output / name).resolve()): {"bind": target, "mode": "rw"}
            for name, target in (("sessions", "/logs"), ("agent-logs", "/agent-logs"))
        },
    )
    try:
        created = container.exec_run(["mkdir", "-p", workdir])
        if created.exit_code:
            raise RuntimeError(
                f"could not create smoke repository: {created.output.decode(errors='replace')}"
            )
        bundle = io.BytesIO()
        with tarfile.open(fileobj=bundle, mode="w") as archive:
            for name, content in {
                "answer.c": '#include <stdio.h>\nint main(void) { puts("1"); return 0; }\n',
                ".gitignore": "build/\n",
            }.items():
                info = tarfile.TarInfo(name)
                info.size = len(content.encode())
                # Files are created through the task user's own tar process.
                archive.addfile(info, io.BytesIO(content.encode()))
        uid = int(container.exec_run(["id", "-u"]).output)
        gid = int(container.exec_run(["id", "-g"]).output)
        owned_bundle = io.BytesIO()
        with (
            tarfile.open(fileobj=io.BytesIO(bundle.getvalue())) as source,
            tarfile.open(fileobj=owned_bundle, mode="w") as target,
        ):
            for member in source:
                member.uid, member.gid = uid, gid
                target.addfile(member, source.extractfile(member))
        container.put_archive(workdir, owned_bundle.getvalue())
        initialized = container.exec_run(
            [
                "sh",
                "-c",
                "git init -q && git add . && git -c user.name=Trace -c user.email=trace@example.invalid commit -qm initial",
            ],
            workdir=workdir,
        )
        if initialized.exit_code:
            raise RuntimeError(
                f"could not initialize smoke repository: {initialized.output.decode(errors='replace')}"
            )
        session = TmuxSession("trace-check", container, disable_recording=True)
        session.start()
        session.send_keys([f"cd {workdir}", "Enter"], block=True)
        agent = AgentFactory.get_agent(agent_name=AgentName(args.agent), **options)
        agent._trace_configuration = options
        agent.set_progress_callback(lambda detail: print(detail, flush=True))
        harness = Harness.__new__(Harness)
        harness._run_id = args.output.name
        result = asyncio.run(
            harness._run_agent_with_timeout(
                trial_handler=SimpleNamespace(
                    task_id="trace-check",
                    trial_name=args.agent,
                    instruction=(
                        "This is a small tracing integration check. Read answer.c. Change it to print 2. "
                        "Compile it to build/answer with cc and run it to verify the output. "
                        "Also run one shell command that exits with status 7, then continue and finish. "
                        "Keep build/answer as a required deliverable for independent checking. "
                        "Use the tools available to you. Keep Git HEAD unchanged. Do not install packages."
                        " Use standard shell or Python fallbacks if optional utilities such as rg are absent."
                        " If the agent tool-call interface itself denies or cannot execute requests, stop instead of retrying denied calls."
                    ),
                ),
                session=session,
                logging_dir=args.output / "agent-logs",
                timeout_sec=args.timeout + 30,
                agent=agent,
                portkey_metadata=None,
                portkey_trace_id=None,
            )
        )
        report["agent_result"] = result.model_dump(mode="json")
        verification = container.exec_run([str(Path(workdir) / "build/answer")])
        report["build_output"] = verification.output.decode(errors="replace")
        report["build_verified"] = (
            verification.exit_code == 0 and verification.output.strip() == b"2"
        )
        report["status"] = (
            "completed" if result.failure_mode.value == "none" else "failed"
        )
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        for path in (args.output / "agent-trace").glob("*/events.jsonl"):
            report["trace"] = str(path)
            report["inspection"] = inspect_trace(path)
        container.remove(force=True)
        (args.output / "validation.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "inspection"},
            indent=2,
        )
    )
    return int(report["status"] != "completed" or not report.get("build_verified"))
