"""Execute the explicitly authorized research copies, retaining every outcome."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = BASE / ("run-" + stamp)
OUT.mkdir(exist_ok=False)
threads = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
env = os.environ.copy()
env.update(threads)
plan = json.loads((BASE / "plan.json").read_text())
authorization = json.loads((BASE / "authorization-addendum.json").read_text())
assert authorization["original_plan_sha256"] == sha(BASE / "plan.json")
manifest = json.loads((BASE / "manifest.json").read_text())
for path, digest in plan["source_hashes"].items():
    assert sha(BASE / "frozen" / path) == digest
for path, digest in manifest["candidate_hashes"].items():
    assert sha(BASE / "candidate" / path) == digest
corpus = json.loads((ROOT / "research/translation-first256-review/plan.json").read_text())["corpus"]
assert len(corpus) == 22
for path, digest in corpus.items():
    assert sha(ROOT / path) == digest

compiler = Path(subprocess.check_output(["which", "g++"], text=True).strip()).resolve()
deps = ROOT / "build/spatial/_deps"
libraries = [deps / f"bullet-build/src/{name}/lib{name}.a" for name in ("BulletDynamics", "BulletCollision", "LinearMath")]
libraries += [Path("/lib/x86_64-linux-gnu/liblapack.so.3"), Path("/lib/x86_64-linux-gnu/libblas.so.3")]
protected = sorted((ROOT / "spatial_backend").glob("*.h")) + sorted((ROOT / "spatial_backend").glob("*.cpp"))
protected += [ROOT / "spatial_backend/CMakeLists.txt", ROOT / "spatial_engine.py", ROOT / "build/spatial/spatial_runner"]
protected = [p for p in protected if p.is_file()]
guard_paths = protected + libraries + [compiler]
before = {str(p): sha(p) for p in guard_paths}
source_files = [BASE / "plan.json", BASE / "authorization-addendum.json", BASE / "manifest.json",
                BASE / "policy_checks.cpp", BASE / "replay_policy.cpp", Path(__file__)]
source_files += [p for dirname in ("frozen", "candidate") for p in sorted((BASE / dirname).rglob("*")) if p.is_file()]
provenance = {"published_source": authorization["published_prospective_source"], "pid": os.getpid(),
              "started_utc": stamp, "threads": threads, "compiler": str(compiler),
              "compiler_version": subprocess.check_output([str(compiler), "--version"], text=True).splitlines()[0],
              "source_hashes": {str(p.relative_to(BASE)): sha(p) for p in source_files},
              "corpus_hashes": corpus, "preexecution_guards": before,
              "scope": "Isolated scheduling controls/captured replays; concurrent cost descriptive, not ranking"}
(OUT / "provenance-before.json").write_text(json.dumps(provenance, indent=2) + "\n")
common = [str(compiler), "-std=c++17", "-O2", "-ffp-contract=off", "-DBT_USE_DOUBLE_PRECISION", "-DSPATIAL_LAPACK_RECOVERY=1",
          "-I" + str(deps / "bullet-src/src"), "-I" + str(deps / "json-src/single_include")]
compiled = {}
completion = {"outputs": str(OUT), "compile_receipts": [], "controls": [], "replays": []}
try:
    for name, source, headers, extra in (
        ("policy_checks", BASE / "policy_checks.cpp", BASE / "candidate/spatial_backend", []),
        ("replay_frozen", BASE / "replay_policy.cpp", BASE / "frozen/spatial_backend", []),
        ("replay_candidate", BASE / "replay_policy.cpp", BASE / "candidate/spatial_backend", ["-DEARLY_POLICY_CANDIDATE=1"]),
    ):
        binary = OUT / name
        command = common + extra + ["-I" + str(headers), str(source)] + [str(p) for p in libraries] + ["-o", str(binary)]
        receipt = {"name": name, "command": command, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        with (OUT / (name + ".compile.stdout")).open("w") as stdout, (OUT / (name + ".compile.stderr")).open("w") as stderr:
            result = subprocess.run(command, env=env, cwd=ROOT, stdout=stdout, stderr=stderr)
        receipt["returncode"] = result.returncode
        if result.returncode == 0:
            receipt["binary_sha256"] = sha(binary)
            compiled[name] = binary
        completion["compile_receipts"].append(receipt)
        (OUT / "compile-receipts.json").write_text(json.dumps(completion["compile_receipts"], indent=2) + "\n")
        print("COMPILE", name, result.returncode, flush=True)
        if result.returncode:
            raise RuntimeError("Compile failure retained; source must not be overwritten")
    receipt = {"binary": str(compiled["policy_checks"]), "binary_sha256": sha(compiled["policy_checks"])}
    with (OUT / "policy_checks.stdout").open("w") as stdout, (OUT / "policy_checks.stderr").open("w") as stderr:
        result = subprocess.run([str(compiled["policy_checks"])], cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
    receipt["returncode"] = result.returncode
    completion["controls"].append(receipt)
    print("CONTROL", result.returncode, flush=True)
    if result.returncode:
        raise RuntimeError("Control failure retained; do not overwrite prospective sources")
    with (OUT / "receipts.jsonl").open("w") as journal:
        for index, capture in enumerate(corpus):
            for label, binary, schedule in (
                ("frozen_default", compiled["replay_frozen"], "default"),
                ("candidate_default", compiled["replay_candidate"], "default"),
                ("candidate_early", compiled["replay_candidate"], "early"),
            ):
                stem = f"capture-{index:02d}-{label}"
                command = [str(binary), capture, "4096", schedule]
                started = time.perf_counter()
                with (OUT / (stem + ".json")).open("w") as stdout, (OUT / (stem + ".stderr")).open("w") as stderr:
                    result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
                receipt = {"index": index, "capture": capture, "label": label, "returncode": result.returncode,
                           "command": command, "binary_sha256": sha(binary), "elapsed_s_descriptive": time.perf_counter() - started,
                           "stdout_sha256": sha(OUT / (stem + ".json")), "stderr_sha256": sha(OUT / (stem + ".stderr"))}
                completion["replays"].append(receipt)
                journal.write(json.dumps(receipt) + "\n")
                journal.flush()
                print("REPLAY", index, label, result.returncode, flush=True)
                # Continue every declared input after failure. All failures stay in archive.
            (OUT / "progress.json").write_text(json.dumps({"completed_captures": index + 1, "replays": len(completion["replays"])}) + "\n")
except Exception as error:
    completion["error"] = str(error)
finally:
    after = {str(p): sha(p) for p in guard_paths}
    completion["postexecution_guards"] = after
    completion["guard_differences"] = {p: {"before": before[p], "after": after[p]} for p in before if before[p] != after[p]}
    completion["executables"] = {name: {"path": str(p), "sha256": sha(p)} for name, p in compiled.items()}
    completion["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    completion["original_plan_unchanged"] = sha(BASE / "plan.json") == authorization["original_plan_sha256"]
    (OUT / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    print("FINISHED", OUT, completion.get("error"), "guardchanges", len(completion["guard_differences"]), flush=True)
