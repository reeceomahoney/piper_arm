"""SLURM experiment monitor — lightweight web GUI."""

import subprocess
from pathlib import Path

from flask import Flask, Response, render_template, request

from piper_arm.slurm import PROJECT_ROOT, REMOTE_HOST, REMOTE_PATH

_dir = Path(__file__).parent
app = Flask(
    __name__,
    template_folder=str(_dir),
    static_folder=str(_dir),
    static_url_path="/static",
)


def ssh(cmd: str, *, timeout: int = 10) -> str:
    result = subprocess.run(
        ["ssh", REMOTE_HOST, cmd],
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


@app.route("/nodes")
def nodes():
    raw = ssh(
        "sinfo -N -p short -o '%.20N %.5t %.15C %.10m %.20G'"
        " | awk 'NR==1 || /h100/ || /l40s/'"
    )
    headers, rows = parse_table(raw)
    return render_template("nodes.html", headers=headers, rows=rows)


@app.route("/jobs")
def jobs():
    raw = ssh("squeue -u $USER -o '%.12i %.30j %.8T %.10M %.20b'")
    headers, rows = parse_table(raw)
    # Rename ugly TRES_PER_NODE header
    headers = ["GPU" if "TRES" in h else h for h in headers]
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
                gpu = part.split("gres/gpu:")[1].split("=")[0]
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
            f"{REMOTE_HOST}:{REMOTE_PATH}/outputs/eval_dist/",
            f"{PROJECT_ROOT}/outputs/eval_dist/",
        ],
        capture_output=True,
        timeout=120,
    )
    return "", 204


@app.route("/clean-logs", methods=["DELETE"])
def clean_logs():
    ssh(
        f"cd {REMOTE_PATH} && running=$(squeue -u $USER -h -t R -o 'slurm-%i.out'); "
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
        f"tail -n +1 -f {REMOTE_PATH}/slurm-{job_id}.out"
        if follow
        else f"cat {REMOTE_PATH}/slurm-{job_id}.out"
    )

    def stream():
        proc = subprocess.Popen(
            ["ssh", REMOTE_HOST, cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            for line in proc.stdout:
                yield f"data: {line.rstrip()}\n\n"
        finally:
            proc.terminate()
            proc.wait()

    return Response(stream(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
