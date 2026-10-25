"""SSH deployment and checkpoint collection for the shared reference runner.

Credentials remain on the coordinating machine. Remote hosts execute independent
prepared shards, so losing the coordinator does not terminate their searches.
"""

from __future__ import annotations

import copy
import glob
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_HOST_KEYS_LOCK = threading.Lock()
_VERIFICATION_LOCK = threading.Lock()


def verify_collected_solution(
    instance: dict, solution: Path, record: dict, cache_root: Path
) -> dict:
    """Check the instance/witness pair, including witnesses shared across inputs."""
    from pitbench.evaluator.artifacts import write_json
    from pitbench.solver_drivers.reference import verify_reference_file

    cache = cache_root / (
        instance["sha256"] + "_" + record["solution_sha256"] + ".json"
    )
    with _VERIFICATION_LOCK:
        if cache.exists():
            checked = json.loads(cache.read_text())["verification"]
        else:
            if (
                hashlib.sha256(Path(instance["path"]).read_bytes()).hexdigest()
                != instance["sha256"]
            ):
                raise ValueError("local instance hash mismatch")
            if (
                hashlib.sha256(solution.read_bytes()).hexdigest()
                != record["solution_sha256"]
            ):
                raise ValueError("collected solution hash mismatch")
            checked = verify_reference_file(
                Path(instance["path"]), solution, instance["generator"]["kind"]
            )
            if not checked["feasible"] or checked["objective"] != record["objective"]:
                raise ValueError(
                    "remote solution failed independent local verification"
                )
            write_json(
                cache,
                {
                    "instance_sha256": instance["sha256"],
                    "solution_sha256": record["solution_sha256"],
                    "verification": checked,
                },
            )
        if not checked["feasible"] or checked["objective"] != record["objective"]:
            raise ValueError(
                "remote objective does not match verified instance/solution pair"
            )
    return checked


def partition_reference_jobs(
    jobs: list[dict], hosts: list[dict], completed: set[str], references: dict
) -> tuple[dict, list[dict]]:
    """Deterministic assignment by configured core-hours, retaining original IDs."""
    assigned = {host["host"]: [] for host in hosts}
    loads = {host["host"]: 0.0 for host in hosts}
    omitted = []
    pending = []
    for job in jobs:
        if job["id"] in completed:
            omitted.append({"job_id": job["id"], "reason": "completed"})
        elif references.get(job["instance"]["id"], {}).get("proven_optimal"):
            omitted.append({"job_id": job["id"], "reason": "proven_optimal"})
        else:
            pending.append(job)
    for job in sorted(pending, key=lambda j: (-j["seconds"] * j["threads"], j["id"])):
        eligible = [h for h in hosts if len(h["cpus"]) >= job["threads"]]
        if not eligible:
            raise ValueError(f"no host fits CPU request for {job['id']}")
        host = min(
            eligible, key=lambda h: (loads[h["host"]] / len(h["cpus"]), h["host"])
        )
        assigned[host["host"]].append(job)
        loads[host["host"]] += job["seconds"] * job["threads"]
    return assigned, omitted


class SSHHost:
    def __init__(self, host: dict, credentials: dict, known_hosts: Path):
        import paramiko

        known_hosts.parent.mkdir(parents=True, exist_ok=True)
        if not known_hosts.exists():
            known_hosts.touch(mode=0o600)

        class RememberKey(paramiko.MissingHostKeyPolicy):
            def missing_host_key(self, client, hostname, key):
                with _HOST_KEYS_LOCK:
                    keys = paramiko.HostKeys(str(known_hosts))
                    old = keys.get(hostname, {}).get(key.get_name())
                    if old is not None and old != key:
                        raise RuntimeError(f"SSH host key changed for {hostname}")
                    keys.add(hostname, key.get_name(), key)
                    temporary = known_hosts.with_suffix(".tmp")
                    keys.save(str(temporary))
                    temporary.chmod(0o600)
                    temporary.replace(known_hosts)

        self.client = paramiko.SSHClient()
        with _HOST_KEYS_LOCK:
            self.client.load_host_keys(str(known_hosts))
        self.client.set_missing_host_key_policy(RememberKey())
        self.client.connect(
            host["host"],
            username=credentials["username"],
            password=credentials["password"],
            timeout=20,
            banner_timeout=20,
            auth_timeout=20,
            allow_agent=False,
            look_for_keys=False,
        )
        self.sftp = self.client.open_sftp()

    def run(self, command: str, timeout: int = 60) -> str:
        _, stdout, stderr = self.client.exec_command(command, timeout=timeout)
        output = stdout.read().decode(errors="replace")
        error = stderr.read().decode(errors="replace")
        status = stdout.channel.recv_exit_status()
        if status:
            raise RuntimeError(
                f"remote exit {status}: {error[-1500:]} {output[-1500:]}"
            )
        return output

    def put(self, source: Path, destination: str) -> None:
        self.sftp.put(str(source), destination)

    def put_json(self, value: dict, destination: str) -> None:
        with self.sftp.open(destination, "w") as stream:
            stream.write(json.dumps(value, indent=2))

    def read_json(self, path: str) -> dict:
        with self.sftp.open(path) as stream:
            return json.loads(stream.read())

    def close(self) -> None:
        self.sftp.close()
        self.client.close()


class RemoteReferenceCollection:
    @staticmethod
    def prepare(config_path: Path, output: Path) -> dict:
        import yaml

        from pitbench.evaluator.artifacts import write_json
        from pitbench.evaluator.collection import ReferenceCollection

        config = yaml.safe_load(config_path.read_text())
        output = output.resolve()
        if (output / "distribution.json").exists():
            saved = json.loads((output / "distribution.json").read_text())
            if saved["config"] != config:
                raise ValueError("existing distributed configuration differs")
            return saved
        source = (ROOT / config["source_campaign"]).resolve()
        manifest = json.loads((source / "campaign.json").read_text())
        ReferenceCollection.status(source)
        references = json.loads((source / "bks.json").read_text())["references"]
        completed = set()
        for result in sorted((source / "jobs").glob("**/result.json")):
            record = json.loads(result.read_text())
            if record["execution_status"] in {"completed", "skipped_proven_optimal"}:
                completed.add(record["job_id"])
        inventory = json.loads((ROOT / config["inventory_file"]).read_text())
        hosts = []
        for entry in inventory:
            if entry.get("error"):
                continue
            count = min(
                config["max_physical_cores_per_host"],
                max(1, int(len(entry["physical_cpu_ids"]) * config["cpu_fraction"])),
            )
            stride = max(1, len(entry["physical_cpu_ids"]) // count)
            hosts.append(
                {
                    "host": entry["host"],
                    "root": entry["work_root"],
                    "cpus": entry["physical_cpu_ids"][::stride][:count],
                    "inventory": entry,
                }
            )
        if len(hosts) != config["expected_hosts"]:
            raise ValueError("inventory does not contain every requested host")
        assigned, omitted = partition_reference_jobs(
            manifest["jobs"], hosts, completed, references
        )
        output.mkdir(parents=True, exist_ok=False)
        files = {}
        for instance in manifest["instances"]:
            path = Path(instance["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != instance["sha256"]:
                raise ValueError("source instance hash mismatch")
            files[str(path)] = f"data/instances/{instance['sha256']}/{path.name}"
        for record in references.values():
            path = Path(record["solution"])
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != record["solution_sha256"]
            ):
                raise ValueError("source solution hash mismatch")
            files[str(path)] = f"data/references/{record['solution_sha256']}.json"
        runtime = (ROOT / config["python_runtime"]).resolve()
        bundle = output / "payload.tar.gz"
        with tarfile.open(bundle, "w:gz", compresslevel=1) as archive:

            def include(source: Path, destination: str):
                if source.is_file() or source.is_symlink():
                    archive.add(source, arcname=destination, recursive=False)

            for path in sorted((ROOT / "pitbench").rglob("*")):
                if "__pycache__" not in path.parts and path.suffix != ".pyc":
                    include(
                        path,
                        str(
                            Path("project/pitbench")
                            / path.relative_to(ROOT / "pitbench")
                        ),
                    )
            include(
                ROOT / "scripts/collect_reference_solutions.py",
                "project/scripts/collect_reference_solutions.py",
            )
            for path in sorted(runtime.rglob("*")):
                include(path, str(Path("runtime") / path.relative_to(runtime)))
            for path in sorted((ROOT / config["wheels"]).glob("*.whl")):
                include(path, "wheels/" + path.name)
            include(ROOT / config["requirements"], "requirements.txt")
            for source_path, destination in files.items():
                include(Path(source_path), "project/" + destination)
            for name, build in config["native_builds"].items():
                source_dir = ROOT / build["source"]
                for pattern in build["files"]:
                    for path in sorted(source_dir.glob(pattern)):
                        include(
                            path,
                            str(
                                Path("native_sources")
                                / name
                                / path.relative_to(source_dir)
                            ),
                        )
        distribution = {
            "schema_version": "1.0",
            "created_at": time.time(),
            "config": config,
            "source_campaign": str(source),
            "source_manifest_sha256": hashlib.sha256(
                (source / "campaign.json").read_bytes()
            ).hexdigest(),
            "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
            "hosts": hosts,
            "assignments": assigned,
            "omitted_jobs": omitted,
            "instances": manifest["instances"],
            "references": references,
            "files": files,
            "original_manifest": manifest,
        }
        write_json(output / "distribution.json", distribution)
        write_json(output / "bks.json", {"references": references})
        return distribution

    @staticmethod
    def host_template(distribution: dict, host: dict) -> dict:
        root = Path(host["root"])
        files = distribution["files"]

        def relocated(path):
            return str(root / "project" / files[str(path)])

        assigned = copy.deepcopy(distribution["assignments"][host["host"]])
        instance_ids = {job["instance"]["id"] for job in assigned}
        manifest = copy.deepcopy(distribution["original_manifest"])
        manifest["host_id"] = host["host"]
        manifest["source_campaign"] = distribution["source_campaign"]
        manifest["source_manifest_sha256"] = distribution["source_manifest_sha256"]
        manifest["instances"] = [
            copy.deepcopy(i)
            for i in distribution["instances"]
            if i["id"] in instance_ids
        ]
        for instance in manifest["instances"]:
            instance["source_path"] = instance["path"]
            instance["path"] = relocated(instance["path"])
        for job in assigned:
            job["instance"]["path"] = relocated(job["instance"]["path"])
        manifest["jobs"] = assigned
        manifest["cpus"] = host["cpus"]
        manifest["python"] = str(root / "env/bin/python")
        manifest["source_python_version"] = manifest["python_version"]
        manifest["config_path"] = str(root / "project/config.yaml")
        manifest["config"]["cpus"] = host["cpus"]
        manifest["config"]["physical_cores"] = len(host["cpus"])
        manifest["config"]["resource_guard"] = distribution["config"]["resource_guard"]
        references = {
            key: copy.deepcopy(value)
            for key, value in distribution["references"].items()
            if key in instance_ids
        }
        for record in references.values():
            record["source_solution"] = record["solution"]
            record["solution"] = relocated(record["solution"])
            record["instance_path"] = next(
                i["path"]
                for i in manifest["instances"]
                if i["id"] == record["instance"]
            )
        smokes = []
        candidates = distribution["instances"]
        for backend in manifest["backends"]:
            smoke_kind = backend.get("smoke_kind", backend["kinds"][0])
            instance = copy.deepcopy(
                next(
                    i
                    for i in candidates
                    if i["generator"]["kind"] == smoke_kind
                    and (
                        not backend.get("smoke_instance_id")
                        or i["id"] == backend["smoke_instance_id"]
                    )
                )
            )
            instance["path"] = relocated(instance["path"])
            smokes.append(
                {
                    "id": "smoke/" + backend["name"],
                    "instance": instance,
                    "backend": backend,
                    "seed": 0,
                    "seconds": 1,
                    "cpus": host["cpus"][: backend.get("threads", 1)],
                }
            )
        return {
            "manifest": manifest,
            "references": references,
            "smokes": smokes,
            "native_builds": distribution["config"]["native_builds"],
            "host": host,
            "bundle_sha256": distribution["bundle_sha256"],
        }

    @staticmethod
    def _credentials(config: dict) -> dict:
        path = ROOT / config["credentials_file"]
        if path.stat().st_mode & 0o077:
            raise ValueError("SSH credential file must have mode 0600")
        return {h["host"]: h for h in json.loads(path.read_text())["hosts"]}

    @staticmethod
    def deploy(config_path: Path, output: Path) -> dict:
        from pitbench.evaluator.artifacts import write_json

        distribution = RemoteReferenceCollection.prepare(config_path, output)
        config = distribution["config"]
        credentials = RemoteReferenceCollection._credentials(config)
        output = output.resolve()

        def install(host):
            connection = None
            try:
                connection = SSHHost(
                    host, credentials[host["host"]], ROOT / config["known_hosts_file"]
                )
                root = host["root"]
                connection.run("umask 077; mkdir -p " + shlex.quote(root))
                try:
                    ready = connection.read_json(root + "/ready.json")
                except FileNotFoundError:
                    ready = {}
                if ready.get("bundle_sha256") == distribution["bundle_sha256"]:
                    RemoteReferenceCollection.start(connection, host)
                    result = {
                        "host": host["host"],
                        "state": "started",
                        "root": root,
                        "cpus": host["cpus"],
                        "time": time.time(),
                        "reused_installation": True,
                    }
                    write_json(output / ("deploy-" + host["host"] + ".json"), result)
                    return result
                connection.put(output / "payload.tar.gz", root + "/payload.tar.gz")
                connection.put_json(
                    RemoteReferenceCollection.host_template(distribution, host),
                    root + "/template.json",
                )
                verify = (
                    "import hashlib;from pathlib import Path;p=Path("
                    + repr(root + "/payload.tar.gz")
                    + ");assert hashlib.sha256(p.read_bytes()).hexdigest()=="
                    + repr(distribution["bundle_sha256"])
                )
                connection.run("python3 -c " + shlex.quote(verify))
                connection.run(
                    "tar -xzf "
                    + shlex.quote(root + "/payload.tar.gz")
                    + " -C "
                    + shlex.quote(root),
                    timeout=180,
                )
                command = (
                    "cd "
                    + shlex.quote(root + "/project")
                    + "; PYTHONPATH="
                    + shlex.quote(root + "/project")
                    + " MPLCONFIGDIR="
                    + shlex.quote(root + "/matplotlib")
                    + " nice -n 10 taskset -c "
                    + ",".join(map(str, host["cpus"][:2]))
                    + " "
                    + shlex.quote(root + "/runtime/bin/python3")
                    + " "
                    + shlex.quote(root + "/project/pitbench/evaluator/remote.py")
                    + " --bootstrap "
                    + shlex.quote(root)
                    + " > "
                    + shlex.quote(root + "/bootstrap.log")
                    + " 2>&1"
                )
                connection.run(command, timeout=1200)
                RemoteReferenceCollection.start(connection, host)
                result = {
                    "host": host["host"],
                    "state": "started",
                    "root": root,
                    "cpus": host["cpus"],
                    "time": time.time(),
                }
            except Exception as error:
                result = {
                    "host": host["host"],
                    "state": "deploy_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "time": time.time(),
                }
            finally:
                if connection is not None:
                    connection.close()
            write_json(output / ("deploy-" + host["host"] + ".json"), result)
            return result

        with ThreadPoolExecutor(
            max_workers=config.get("deployment_workers", 4)
        ) as pool:
            results = list(pool.map(install, distribution["hosts"]))
        write_json(output / "deployment.json", {"time": time.time(), "hosts": results})
        return {"hosts": results}

    @staticmethod
    def start(connection: SSHHost, host: dict) -> None:
        root = host["root"]
        marker = root + "/campaign/STOP"
        repair_marker = root + "/campaign/NEEDS_REPAIR.json"
        code = f"import fcntl,os;from pathlib import Path;p=Path({root!r})/'campaign';f=(p/'controller.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);print('READY')"
        try:
            connection.run(
                shlex.quote(root + "/env/bin/python") + " -c " + shlex.quote(code)
            )
        except RuntimeError:
            return
        connection.run(
            "if [ ! -e "
            + shlex.quote(marker)
            + " ] && [ ! -e "
            + shlex.quote(root + "/campaign/DONE")
            + " ] && [ ! -e "
            + shlex.quote(repair_marker)
            + " ]; then cd "
            + shlex.quote(root + "/project")
            + "; nohup nice -n 10 "
            + shlex.quote(root + "/env/bin/python")
            + " -m scripts.collect_reference_solutions --campaign-config "
            + shlex.quote(root + "/project/config.yaml")
            + " --output-dir "
            + shlex.quote(root + "/campaign")
            + " >> "
            + shlex.quote(root + "/controller.log")
            + " 2>&1 < /dev/null & fi"
        )

    @staticmethod
    def sync(config_path: Path, output: Path) -> dict:
        import fcntl

        with (output / "cycle.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return RemoteReferenceCollection._sync_once(config_path, output)

    @staticmethod
    def _sync_once(config_path: Path, output: Path) -> dict:
        import yaml

        from pitbench.evaluator.artifacts import write_json

        config = yaml.safe_load(config_path.read_text())
        output = output.resolve()
        distribution = json.loads((output / "distribution.json").read_text())
        credentials = RemoteReferenceCollection._credentials(config)
        known = {i["id"]: i for i in distribution["instances"]}

        def collect(host):
            connection = None
            try:
                connection = SSHHost(
                    host, credentials[host["host"]], ROOT / config["known_hosts_file"]
                )
                status = connection.read_json(host["root"] + "/campaign/status.json")
                capacity_plan = host["root"] + "/campaign/capacity-upgrade-plan.json"
                capacity_stop = host["root"] + "/campaign/STOP"
                try:
                    connection.sftp.stat(capacity_plan)
                    connection.sftp.stat(capacity_stop)
                    watcher = host["root"] + "/capacity_watch.py"
                    log = host["root"] + "/capacity-watch.log"
                    singleton = host["root"] + "/campaign/capacity-upgrade.lock"
                    probe = (
                        "import fcntl\n"
                        + "f=open("
                        + repr(singleton)
                        + ',"a+")\n'
                        + "try:\n fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
                        + "except BlockingIOError:\n print('ACTIVE')\n"
                        + "else:\n fcntl.flock(f,fcntl.LOCK_UN)\n print('FREE')\n"
                    )
                    if (
                        connection.run("python3 -c " + shlex.quote(probe)).strip()
                        == "FREE"
                    ):
                        connection.run(
                            "nohup python3 -u "
                            + shlex.quote(watcher)
                            + " >> "
                            + shlex.quote(log)
                            + " 2>&1 < /dev/null &"
                        )
                except OSError:
                    pass
                references = connection.read_json(host["root"] + "/campaign/bks.json")[
                    "references"
                ]
                local = output / "hosts" / host["host"]
                local.mkdir(parents=True, exist_ok=True)
                accepted = {}
                for identity, record in references.items():
                    instance = known[identity]
                    if record["instance_sha256"] != instance["sha256"]:
                        raise ValueError("remote reference input hash mismatch")
                    digest = record["solution_sha256"]
                    if len(digest) != 64 or any(
                        c not in "0123456789abcdef" for c in digest
                    ):
                        raise ValueError("invalid remote solution digest")
                    solution = output / "solutions" / (digest + ".json")
                    solution.parent.mkdir(exist_ok=True)
                    if not solution.exists():
                        remote = record["solution"]
                        if (
                            not remote.startswith(host["root"] + "/")
                            or ".." in Path(remote).parts
                        ):
                            raise ValueError(
                                "remote solution is outside the deployment root"
                            )
                        temp = solution.with_suffix("." + host["host"] + ".tmp")
                        connection.sftp.get(remote, str(temp))
                        if hashlib.sha256(temp.read_bytes()).hexdigest() != digest:
                            raise ValueError("remote solution hash mismatch")
                        temp.replace(solution)
                    checked = verify_collected_solution(
                        instance, solution, record, output / "verifications"
                    )
                    accepted[identity] = {
                        **record,
                        "remote_host": host["host"],
                        "remote_solution": record["solution"],
                        "solution": str(solution),
                        "instance_path": instance["path"],
                        "locally_verified": True,
                        "local_verification": checked,
                    }
                write_json(local / "status.json", status)
                write_json(local / "bks.json", {"references": accepted})
                metadata_path = local / "artifact_versions.json"
                versions = (
                    json.loads(metadata_path.read_text())
                    if metadata_path.exists()
                    else {}
                )
                for remote_name, local_name in {
                    "campaign/attempts.jsonl": "attempts.jsonl",
                    "campaign/campaign.json": "campaign.json",
                    "campaign/resource_state.json": "resource_state.json",
                    "campaign/capacity-upgrade-state.json": "capacity-upgrade-state.json",
                    "campaign/NEEDS_REPAIR.json": "needs-repair.json",
                    "ready.json": "ready.json",
                    "controller.log": "controller.log",
                }.items():
                    remote_path = host["root"] + "/" + remote_name
                    try:
                        attributes = connection.sftp.stat(remote_path)
                    except FileNotFoundError:
                        continue
                    version = [attributes.st_size, attributes.st_mtime]
                    if versions.get(remote_name) != version:
                        temporary = local / (local_name + ".tmp")
                        connection.sftp.get(remote_path, str(temporary))
                        temporary.replace(local / local_name)
                        versions[remote_name] = version
                write_json(metadata_path, versions)
                result_state = "reachable"
                try:
                    connection.sftp.stat(host["root"] + "/campaign/NEEDS_REPAIR.json")
                    result_state = "needs_repair"
                except OSError:
                    pass
                if time.time() - status.get("updated_at", 0) > config.get(
                    "stale_heartbeat_sec", 180
                ):
                    deferred_statuses = {
                        "stopped_before_start",
                        "interrupted_by_host_restart",
                    }
                    terminal_campaign = (
                        status.get("running_jobs") == 0
                        and status.get("finished_jobs") == status.get("jobs")
                        and not deferred_statuses.intersection(
                            status.get("execution_counts", {})
                        )
                    )
                    try:
                        connection.sftp.stat(host["root"] + "/campaign/STOP")
                        stopping = True
                    except OSError:
                        stopping = False
                    if terminal_campaign and not stopping:
                        lock_path = host["root"] + "/campaign/controller.lock"
                        probe = (
                            "import fcntl\n"
                            + "f=open("
                            + repr(lock_path)
                            + ',"a+")\n'
                            + "try:\n fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
                            + "except BlockingIOError:\n print('ACTIVE')\n"
                            + "else:\n fcntl.flock(f,fcntl.LOCK_UN)\n print('FREE')\n"
                        )
                        if (
                            connection.run("python3 -c " + shlex.quote(probe)).strip()
                            == "FREE"
                        ):
                            errors = {
                                name: count
                                for name, count in status.get(
                                    "execution_counts", {}
                                ).items()
                                if name
                                in {
                                    "solver_error",
                                    "watchdog_timeout",
                                    "process_error",
                                    "collector_error",
                                }
                            }
                            if errors:
                                marker_path = (
                                    host["root"] + "/campaign/NEEDS_REPAIR.json"
                                )
                                payload = json.dumps(
                                    {
                                        "time": time.time(),
                                        "host": host["host"],
                                        "errors": errors,
                                        "state": "needs_repair",
                                    }
                                )
                                connection.run(
                                    "python3 -c "
                                    + shlex.quote(
                                        "from pathlib import Path; Path("
                                        + repr(marker_path)
                                        + ").write_text("
                                        + repr(payload)
                                        + ")"
                                    )
                                )
                                result_state = "needs_repair"
                            else:
                                done = host["root"] + "/campaign/DONE"
                                connection.run(
                                    "python3 -c "
                                    + shlex.quote(
                                        "from pathlib import Path; Path("
                                        + repr(done)
                                        + ").touch()"
                                    )
                                )
                                result_state = "complete"
                    else:
                        RemoteReferenceCollection.start(connection, host)
                        result_state = "reachable"
                else:
                    result_state = "reachable"
                return {"host": host["host"], "state": result_state, "status": status}
            except Exception as error:
                return {
                    "host": host["host"],
                    "state": "sync_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            finally:
                if connection is not None:
                    connection.close()

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(collect, distribution["hosts"]))
        best = copy.deepcopy(distribution["references"])
        for path in (output / "hosts").glob("*/bks.json"):
            for identity, record in json.loads(path.read_text())["references"].items():
                old = best.get(identity)
                if old is None or (
                    record["objective"],
                    not record["proven_optimal"],
                ) < (old["objective"], not old["proven_optimal"]):
                    best[identity] = record
        summary = {
            "updated_at": time.time(),
            "hosts": results,
            "instances": len(known),
            "assigned_jobs": sum(len(j) for j in distribution["assignments"].values()),
            "configured_cpu_slots": sum(
                len(result.get("status", {}).get("cpus", host["cpus"]))
                for host, result in zip(distribution["hosts"], results, strict=True)
            ),
            "cpu_slots_by_host": {
                result["host"]: len(result.get("status", {}).get("cpus", host["cpus"]))
                for host, result in zip(distribution["hosts"], results, strict=True)
            },
            "instances_with_reference": len(best),
            "proven_optimal": sum(r["proven_optimal"] for r in best.values()),
            "running_jobs": sum(
                r.get("status", {}).get("running_jobs", 0) for r in results
            ),
        }
        write_json(output / "bks.json", {"updated_at": time.time(), "references": best})
        write_json(output / "status.json", summary)
        return summary

    @staticmethod
    def sync_loop(config_path: Path, output: Path) -> None:
        import fcntl

        import yaml

        lock = (output / "sync.lock").open("a+")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = yaml.safe_load(config_path.read_text())
        while not (output / "STOP_SYNC").exists():
            summary = RemoteReferenceCollection.sync(config_path, output)
            print(
                json.dumps({k: v for k, v in summary.items() if k != "hosts"}),
                flush=True,
            )
            time.sleep(config.get("sync_interval_sec", 60))


def bootstrap_reference_host(root: Path) -> None:
    """Offline host setup; runs first with the portable base Python, then its venv."""
    root = root.resolve()
    environment = root / "env"
    if Path(sys.prefix) != environment:
        if not (environment / "bin/python").exists():
            subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
        subprocess.run(
            [
                str(environment / "bin/python"),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-cache-dir",
                "--find-links",
                str(root / "wheels"),
                "-r",
                str(root / "requirements.txt"),
            ],
            check=True,
        )
        os.execv(
            str(environment / "bin/python"),
            [str(environment / "bin/python"), __file__, "--bootstrap", str(root)],
        )
    import importlib.metadata

    import yaml

    from pitbench.evaluator.artifacts import write_json
    from pitbench.evaluator.collection import run_collection_process

    template = json.loads((root / "template.json").read_text())
    manifest = template["manifest"]
    binaries = {}
    for name, build in template["native_builds"].items():
        source = root / "native_sources" / name
        destination = root / "native" / name
        destination.parent.mkdir(exist_ok=True)
        sources = [
            p
            for pattern in build["sources"]
            for p in sorted(glob.glob(str(source / pattern)))
        ]
        command = [
            build.get("compiler", "g++"),
            *build["flags"],
            *sources,
            "-o",
            str(destination),
        ]
        write_json(
            destination.with_suffix(".build.json"),
            {"command": command, "source_identity": build},
        )
        subprocess.run(command, cwd=source, check=True)
        binaries[name] = destination
    identities = {}
    for backend in manifest["backends"]:
        name = backend["name"]
        if name in binaries:
            backend["source_binary_sha256"] = backend["binary_sha256"]
            backend["binary"] = str(binaries[name])
            backend["binary_sha256"] = hashlib.sha256(
                binaries[name].read_bytes()
            ).hexdigest()
        if backend.get("package"):
            actual = importlib.metadata.version(backend["package"])
            if actual != backend["package_version"]:
                raise ValueError(f"backend version mismatch: {name}")
        identities[name] = backend
    for job in manifest["jobs"]:
        job["backend"] = identities[job["backend"]["name"]]
    campaign = root / "campaign"
    campaign.mkdir(exist_ok=True)
    (root / "project/config.yaml").write_text(
        yaml.safe_dump(manifest["config"], sort_keys=False)
    )
    if not (campaign / "campaign.json").exists():
        shutil.copytree(root / "project/pitbench", campaign / "code_snapshot/pitbench")
        snapshot = campaign / "code_snapshot"
        manifest["source_hashes"] = {
            str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(snapshot.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts
        }
        manifest["python"] = sys.executable
        manifest["python_version"] = sys.version.split()[0]
        manifest["deployed_at"] = time.time()
        manifest["bundle_sha256"] = template["bundle_sha256"]
        write_json(campaign / "campaign.json", manifest)
        write_json(campaign / "prior_references.json", template["references"])
        write_json(campaign / "bks.json", {"references": template["references"]})
        write_json(
            campaign / "status.json",
            {"updated_at": time.time(), "state": "prepared", "running_jobs": 0},
        )
    for smoke in template["smokes"]:
        smoke["backend"] = identities[smoke["backend"]["name"]]
        directory = root / "smoke" / smoke["backend"]["name"]
        directory.mkdir(parents=True, exist_ok=True)
        smoke["directory"] = str(directory)
        smoke["host_id"] = manifest["host_id"]
        write_json(directory / "job.json", smoke)
        result = run_collection_process(
            [
                sys.executable,
                "-m",
                "pitbench.solver_drivers.reference",
                "--worker-job",
                str(directory / "job.json"),
            ],
            directory,
            {"job_id": smoke["id"]},
            timeout=90,
        )
        if result.get("execution_status") != "completed" or not result.get(
            "verified_feasible"
        ):
            raise RuntimeError(
                f"remote smoke failed: {smoke['backend']['name']}: {result}"
            )
    write_json(
        root / "ready.json",
        {
            "time": time.time(),
            "host": manifest["host_id"],
            "cpus": manifest["cpus"],
            "backends": identities,
            "bundle_sha256": template["bundle_sha256"],
        },
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=Path, required=True)
    bootstrap_reference_host(parser.parse_args().bootstrap)
