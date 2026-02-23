"""SLURM experiment monitor — lightweight web GUI."""

import os
import subprocess
from pathlib import Path

from flask import Flask, Response, render_template

_dir = Path(__file__).parent
app = Flask(
    __name__,
    template_folder=str(_dir),
    static_folder=str(_dir),
    static_url_path="/static",
)

REMOTE_HOST = os.environ.get("REMOTE_HOST", "htc")
REMOTE_PATH = os.environ.get("REMOTE_PATH", "$DATA/piper_arm")


def ssh(cmd: str, *, timeout: int = 10) -> str:
    result = subprocess.run(
        ["ssh", REMOTE_HOST, cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout


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
    lines = [ln for ln in raw.strip().splitlines() if ln.strip()]
    if not lines:
        return "<em>No data</em>"
    headers = lines[0].split()
    rows = [ln.split() for ln in lines[1:]]
    html = "<table><tr>"
    for h in headers:
        html += f"<th>{h}</th>"
    html += "</tr>"
    for row in rows:
        state = row[1] if len(row) > 1 else ""
        cls = f"state-{state}" if state else ""
        html += "<tr>"
        for cell in row:
            html += f'<td class="{cls}">{cell}</td>'
        html += "</tr>"
    html += "</table>"
    return html


@app.route("/jobs")
def jobs():
    raw = ssh("squeue -u $USER -o '%.12i %.30j %.8T %.10M %.20R'")
    lines = [ln for ln in raw.strip().splitlines() if ln.strip()]
    if not lines:
        return "<em>No jobs</em>"
    headers = lines[0].split()
    rows = [ln.split() for ln in lines[1:]]
    html = '<table id="jobs-table"><tr>'
    for h in headers:
        html += f"<th>{h}</th>"
    html += "<th></th></tr>"
    for row in rows:
        job_id = row[0] if row else ""
        state = row[2] if len(row) > 2 else ""
        cls = f"state-{state}" if state else ""
        html += "<tr>"
        for cell in row:
            html += f'<td class="{cls}">{cell}</td>'
        html += "<td>"
        if state == "RUNNING":
            html += (
                '<button class="btn btn-log"'
                f" onclick=\"openLog('{job_id}')\">logs</button> "
            )
        html += (
            '<button class="btn btn-cancel"'
            f" onclick=\"cancelJob('{job_id}')\">cancel</button>"
        )
        html += "</td></tr>"
    html += "</table>"
    return html


@app.route("/cancel/<job_id>", methods=["POST"])
def cancel(job_id):
    if not job_id.isdigit():
        return "Bad job id", 400
    ssh(f"scancel {job_id}")
    return "", 204


@app.route("/logs/<job_id>")
def logs(job_id):
    if not job_id.isdigit():
        return "Bad job id", 400

    def stream():
        proc = subprocess.Popen(
            ["ssh", REMOTE_HOST, f"tail -n +1 -f {REMOTE_PATH}/slurm-{job_id}.out"],
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
