# Qwen3-ASR-Toolkit

- `source asr-env/bin/acticate`

- `pip install vllm`

- `vllm serve Qwen/Qwen3-ASR-1.7B --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.4 --max-model-len 4096 --max-num-seqs 24 --max-num-batched-tokens 2048 --no-enforce-eager`

- `pip install -e .`

- `QWEN3_ASR_API_URL="http://127.0.0.1:8000/v1/audio/transcriptions" QWEN3_ASR_HOST="0.0.0.0" QWEN3_ASR_PORT="8001" QWEN3_ASR_AUTO_CLEAN_CACHE="true" qwen3-asr-api`

## Voiceprint rooms

Add a user to a room (room membership is stored in the voiceprint SQLite database; repeating the request is safe):

```sh
curl -X POST "http://localhost:8001/voiceprints/rooms/team-a/users" \
  -H "X-Api-Key: $VOICEPRINT_API_KEY" \\
  -F "user_id=u_1"
```

Search only the enrolled members of that room:

```sh
curl -X POST "http://localhost:8001/voiceprints/search" \
  -H "X-Api-Key: $VOICEPRINT_API_KEY" \\
  -F "room_id=team-a" -F "file=@sample.wav"
```

`/voiceprints/search` still accepts repeated `user_ids` form fields for existing clients. Send either `room_id` or `user_ids`.

## Deploying to the ASR server

Deploy the latest `prod` commit to `s2t.quickom.net` and restart the API service:

```sh
DEPLOY_SSH_AUTH_SOCK="$SSH_AUTH_SOCK" ./deploy.sh
```

The script preserves untracked server data, stops if tracked server files have local edits, and leaves the vLLM process running.
