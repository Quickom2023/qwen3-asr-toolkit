# Qwen3-ASR-Toolkit
- `python -m venv asr-env`

- `source asr-env/bin/acticate`

- `pip install -e .`

- `pip install qwen3-asr[vllm]`

- `qwen-asr-serve Qwen/Qwen3-ASR-1.7B --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.3 --max-model-len 4096 --max-num-seqs 16 --max-num-batched-tokens 1024 --no-enforce-eager --allowed-local-media-path /tmp`

- `QWEN3_ASR_API_URL="http://127.0.0.1:8000/v1/audio/transcriptions" QWEN3_ASR_HOST="0.0.0.0" QWEN3_ASR_PORT="8001" QWEN3_ASR_AUTO_CLEAN_CACHE="true" qwen3-asr-api`