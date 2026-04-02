# Qwen3-ASR-Toolkit

- `source asr-env/bin/acticate`

- `pip install vllm`

- `vllm serve Qwen/Qwen3-ASR-1.7B --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.4 --max-model-len 4096 --max-num-seqs 24 --max-num-batched-tokens 2048 --no-enforce-eager`

- `pip install -e .`

- `QWEN3_ASR_API_URL="http://127.0.0.1:8000/v1/audio/transcriptions" QWEN3_ASR_HOST="0.0.0.0" QWEN3_ASR_PORT="8001" QWEN3_ASR_AUTO_CLEAN_CACHE="true" qwen3-asr-api`