"""SLURM experiment monitor — lightweight web GUI."""

import re
import subprocess
from pathlib import Path

from flask import Flask, Response, render_template, request

from slurm_tools.slurm import PROJECT_ROOT, load_config

config = load_config()

dir = Path(__file__).parent
app = Flask(
    __name__,
    template_folder=str(dir),
    static_folder=str(dir),
    static_url_path="/static",
)


def ssh(cmd: str, *, timeout: int = 10) -> str:
    result = subprocess.run(
        ["ssh", config.host, cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout


def parse_table(raw: str) -> tuple[list[str], list[list[str]]]:
    lines = [ln for ln in raw.strip().splitlines() if ln.strip()]
    if not lines:
        return [], []
    return lines[0].split(), [ln.split() for ln in lines[1:]]


# -- Routes ----------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


GRES_RE = re.compile(r"gpu:(?:[^:(]+:)?(\d+)", re.IGNORECASE)
GRES_TYPE_RE = re.compile(r"gpu:([^:(]+):(\d+)", re.IGNORECASE)
DOWN = {"down", "down*", "drain", "drain*", "drng", "maint"}
NODELIST_RE = re.compile(r"^(.+)\[(.+)]$")


def gpu_count(gres: str) -> int:
    m = GRES_RE.search(gres)
    return int(m.group(1)) if m else 0


def gpu_type_and_count(gres: str) -> tuple[str, int] | None:
    m = GRES_TYPE_RE.search(gres)
    return (m.group(1).upper(), int(m.group(2))) if m else None


def expand_nodelist(s: str) -> list[str]:
    m = NODELIST_RE.match(s)
    if not m:
        return [s]
    prefix, ranges = m.group(1), m.group(2)
    out: list[str] = []
    for part in ranges.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            out += [
                f"{prefix}{str(i).zfill(len(lo))}" for i in range(int(lo), int(hi) + 1)
            ]
        else:
            out.append(f"{prefix}{part}")
    return out


@app.route("/nodes")
def nodes():
    raw_nodes = ssh("sinfo -N -h -o '%N|%t|%G'")
    raw_jobs = ssh("squeue -h --states=running,completing -o '%N|%b'")

    # Per-node allocated GPU count from running jobs
    node_alloc: dict[str, int] = {}
    for line in raw_jobs.strip().splitlines():
        parts = line.split("|")
        if len(parts) != 2 or not parts[0].strip() or parts[1] in ("", "N/A"):
            continue
        for node in expand_nodelist(parts[0]):
            node_alloc[node] = node_alloc.get(node, 0) + gpu_count(parts[1])

    # Aggregate by GPU type
    seen: set[str] = set()
    totals: dict[str, int] = {}
    free: dict[str, int] = {}
    for line in raw_nodes.strip().splitlines():
        parts = line.split("|")
        if len(parts) != 3 or parts[0] in seen:
            continue
        seen.add(parts[0])
        parsed = gpu_type_and_count(parts[2])
        if not parsed:
            continue
        gpu_type, total = parsed
        state = parts[1]

        allocated = node_alloc.get(parts[0], 0)
        if not allocated and state in ("alloc", "resv"):
            allocated = total
        node_free = max(0, total - allocated) if state not in DOWN else 0

        totals[gpu_type] = totals.get(gpu_type, 0) + total
        free[gpu_type] = free.get(gpu_type, 0) + node_free

    gpus = []
    for gpu_type in sorted(totals):
        t, f = totals[gpu_type], free[gpu_type]
        gpus.append(
            {
                "type": gpu_type,
                "total": t,
                "used": t - f,
                "free": f,
                "pct": round((t - f) / t * 100) if t else 0,
            }
        )
    return render_template("nodes.html", gpus=gpus)


@app.route("/jobs")
def jobs():
    raw = ssh("squeue -u $USER -o '%.12i %.30j %.8T %.10M %.20b'")
    headers, rows = parse_table(raw)
    # Rename ugly TRES_PER_NODE header
    headers = ["GPU" if "TRES" in h else h for h in headers]
    # Format GPU column: "gpu:l40s:1" -> "l40s"
    gpu_idx = headers.index("GPU") if "GPU" in headers else None
    if gpu_idx is not None:
        for row in rows:
            if gpu_idx < len(row):
                m = GRES_TYPE_RE.search(row[gpu_idx])
                row[gpu_idx] = (
                    f"{m.group(1).upper()}:{m.group(2)}" if m else row[gpu_idx]
                )
    # Recent completed/failed/cancelled jobs from the last 7 days
    raw_closed = ssh(
        "sacct -u $USER -S now-7days --noheader --parsable2 -X "
        "-o 'JobID,JobName,State,Elapsed,AllocTRES' "
        "| grep -vE 'RUNNING|PENDING'"
    )
    closed_lines = [ln for ln in raw_closed.strip().splitlines() if ln.strip()]
    closed_rows = [ln.split("|") for ln in reversed(closed_lines)]
    for row in closed_rows:
        # Normalise e.g. "CANCELLED by 12345" or "CANCELLED+" to "CANCELLED"
        if len(row) > 2:
            row[2] = row[2].split()[0].rstrip("+")
        tres = row[4] if len(row) > 4 else ""
        gpu = ""
        for part in tres.split(","):
            if "gres/gpu:" in part:
                rest = part.split("gres/gpu:")[1]
                name, _, count = rest.partition("=")
                gpu = f"{name.upper()}:{count}" if count else name.upper()
                break
        row[4:] = [gpu]
    closed_headers = ["JOBID", "NAME", "STATE", "ELAPSED", "GPU"] if closed_rows else []
    return render_template(
        "jobs.html",
        headers=headers,
        rows=rows,
        closed_headers=closed_headers,
        closed_rows=closed_rows,
    )


@app.route("/cancel/<job_id>", methods=["POST"])
def cancel(job_id):
    if not job_id.isdigit():
        return "Bad job id", 400
    ssh(f"scancel {job_id}")
    return "", 204


@app.route("/fetch-results", methods=["PUT"])
def fetch_results():
    subprocess.run(
        [
            "rsync",
            "-avz",
            f"{config.host}:{config.remote_path}/outputs/eval_dist/",
            f"{PROJECT_ROOT}/outputs/eval_dist/",
        ],
        capture_output=True,
        timeout=120,
    )
    return "", 204


@app.route("/clean-logs", methods=["DELETE"])
def clean_logs():
    ssh(
        f"cd {config.remote_path}/slurm"
        " && running=$(squeue -u $USER -h -t R -o 'slurm-%i.out'); "
        'if [ -n "$running" ]; then '
        'ls slurm-*.out 2>/dev/null | grep -vF "$running" | xargs -r rm -f; '
        "else "
        "rm -f slurm-*.out; "
        "fi"
    )
    return "", 204


@app.route("/logs/<job_id>")
def logs(job_id):
    if not job_id.isdigit():
        return "Bad job id", 400

    follow = request.args.get("follow", "1") == "1"
    cmd = (
        f"tail -n +1 -f {config.remote_path}/slurm/slurm-{job_id}.out"
        if follow
        else f"cat {config.remote_path}/slurm/slurm-{job_id}.out"
    )

    def stream():
        proc = subprocess.Popen(
            ["ssh", config.host, cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            for line in proc.stdout or []:
                yield f"data: {line.rstrip()}\n\n"
        finally:
            proc.terminate()
            proc.wait()

    return Response(stream(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
