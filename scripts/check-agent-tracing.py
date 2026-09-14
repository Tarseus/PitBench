#!/usr/bin/env python3
"""Run one configured agent on a tiny repository and inspect its actual trace."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import tarfile
import uuid
from pathlib import Path
from types import SimpleNamespace

import docker
from pitbench.cli.evaluate_config import EvaluationConfig
from pitbench.harness.agents import AgentFactory, AgentName
from pitbench.harness.harness.harness import Harness
from pitbench.harness.terminal.docker_compose_manager import DockerComposeManager
from pitbench.harness.terminal.tmux_session import TmuxSession
from pitbench.harness.utils.trace_validation import inspect_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent", required=True, choices=sorted(AgentFactory.AGENT_NAME_TO_IMPORT_PATH)
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--config", type=Path, default=Path("config/evaluate.local.yaml")
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--user", default="root", help="Task-container user for the smoke run"
    )
    parser.add_argument("--agent-kwarg", action="append", default=[])
    args = parser.parse_args()
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


if __name__ == "__main__":
    raise SystemExit(main())
