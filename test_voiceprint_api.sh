#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${DEPLOY_HOST:-ubuntu@s2t.quickom.net}"
SSH_KEY="${DEPLOY_SSH_KEY:-$HOME/.ssh/bw-deploy-datagram}"

if [[ -n "${DEPLOY_SSH_AUTH_SOCK:-}" ]]; then
  export SSH_AUTH_SOCK="$DEPLOY_SSH_AUTH_SOCK"
fi

if [[ ! -r "$SSH_KEY" ]]; then
  echo "SSH key is not readable: $SSH_KEY" >&2
  exit 1
fi

ssh -A -i "$SSH_KEY" -o BatchMode=yes "$REMOTE_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail

service=quickom-asr-api.service
cd /home/ubuntu/qwen3-asr-toolkit
pid="$(systemctl show "$service" -p MainPID --value)"
if [[ -z "$pid" || "$pid" == 0 ]]; then
  echo "ASR API service has no running process." >&2
  exit 1
fi

QWEN3_ASR_API_PID="$pid" /home/ubuntu/qwen3-asr-toolkit/asr-env/bin/python3 - <<'PY'
import json
import os
import secrets
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path
from dotenv import dotenv_values, find_dotenv


def process_environment(pid):
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    return {
        key.decode(): value.decode()
        for item in raw.split(b"\0")
        if b"=" in item
        for key, value in [item.split(b"=", 1)]
    }


pid = os.environ["QWEN3_ASR_API_PID"]
env = process_environment(pid)
dotenv_path = find_dotenv(usecwd=True)
dotenv_env = dotenv_values(dotenv_path) if dotenv_path else {}
api_key = env.get("QWEN3_ASR_API_KEY") or dotenv_env.get("QWEN3_ASR_API_KEY")
db_path = env.get("VP_DB_PATH") or "/home/ubuntu/qwen3-asr-toolkit/qwen3_asr_toolkit/db/voiceprints.db"
room_ids = [
    f"__deployment_smoke_test_{i}__{int(time.time())}_{secrets.token_hex(4)}"
    for i in (1, 2)
]
rooms = [
    {
        "room_id": room_ids[0],
        "user_ids": [f"__deployment_smoke_user_{i}_{secrets.token_hex(3)}" for i in (1, 2)],
    },
    {
        "room_id": room_ids[1],
        "user_ids": [f"__deployment_smoke_user_3_{secrets.token_hex(3)}"],
    },
]
url = "http://127.0.0.1:8001/voiceprints/rooms/users"
headers = {"Content-Type": "application/json"}
if api_key:
    headers["X-Api-Key"] = api_key


def post_members():
    request = urllib.request.Request(
        url,
        data=json.dumps({"rooms": rooms}).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"Room membership request returned HTTP {error.code}: {detail}") from error


try:
    first_status, first = post_members()
    if first_status != 200 or first.get("rooms") != rooms or first.get("added_count") != 3:
        raise RuntimeError(f"Unexpected batch-add response: {first}")

    second_status, second = post_members()
    if second_status != 200 or second.get("rooms") != rooms or second.get("added_count") != 0:
        raise RuntimeError(f"Unexpected idempotency response: {second}")

    print("Voiceprint JSON batch-add and idempotency checks passed.")
finally:
    # Remove only this run's uniquely named smoke-test room from the store.
    with sqlite3.connect(db_path, timeout=10) as connection:
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'voiceprint_room_member'"
        ).fetchone()
        if table_exists:
            connection.execute(
                "DELETE FROM voiceprint_room_member WHERE room_id IN (?, ?)",
                room_ids,
            )
PY
REMOTE
