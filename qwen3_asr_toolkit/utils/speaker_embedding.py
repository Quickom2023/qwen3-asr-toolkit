"""Speaker embeddings from WeSpeaker ResNet34-LM (ONNX, CPU), and the speech they are taken from."""

import threading

import numpy as np
import onnxruntime as ort
import torch
import torchaudio.compliance.kaldi as kaldi
from silero_vad import get_speech_timestamps

from qwen3_asr_toolkit.utils.audio_tools import WAV_SAMPLE_RATE


MODEL_REPO = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
MODEL_FILE = "voxceleb_resnet34_LM.onnx"
MODEL_VERSION = "wespeaker-voxceleb-resnet34-LM"
# One thread per inference: concurrency comes from the request executor.
ORT_THREADS = 1
# Exact-zero runs this long are padding (WebRTC gaps, muted input), not speech.
ZERO_RUN_MIN_MS = 10


def fbank(samples: np.ndarray) -> np.ndarray:
    """80-bin Kaldi fbank of 16 kHz audio, mean-normalized per utterance: (frames, 80)."""
    waveform = torch.from_numpy(np.asarray(samples, dtype=np.float32) * (1 << 15)).unsqueeze(0)
    feats = kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=WAV_SAMPLE_RATE,
        window_type="hamming",
    )
    feats = feats - feats.mean(dim=0, keepdim=True)
    return feats.numpy().astype(np.float32)


class Embedder:
    """One shared ONNX session; `embed` is safe to call from several threads."""

    def __init__(self, model_path: str) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = ORT_THREADS
        options.inter_op_num_threads = ORT_THREADS
        self._session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name
        self._output = self._session.get_outputs()[0].name

    def embed(self, samples: np.ndarray) -> np.ndarray:
        """L2-normalized 256-d embedding of 16 kHz mono audio."""
        feats = fbank(samples)[None, :, :]
        vector = self._session.run([self._output], {self._input: feats})[0][0]
        return (vector / np.linalg.norm(vector)).astype(np.float32)


def _drop_zero_runs(samples: np.ndarray, min_len: int) -> np.ndarray:
    is_zero = samples == 0
    if not is_zero.any():
        return samples
    edges = np.flatnonzero(np.diff(np.concatenate(([0], is_zero.astype(np.int8), [0]))))
    starts, ends = edges[0::2], edges[1::2]
    long_runs = (ends - starts) >= min_len
    keep = np.ones(len(samples), dtype=bool)
    for start, end in zip(starts[long_runs], ends[long_runs]):
        keep[start:end] = False
    return samples[keep]


def speech_only(samples: np.ndarray, vad_model, vad_lock: threading.Lock) -> np.ndarray:
    """The speech in `samples`: long zero runs removed, then Silero VAD segments joined."""
    samples = _drop_zero_runs(
        np.asarray(samples, dtype=np.float32),
        ZERO_RUN_MIN_MS * WAV_SAMPLE_RATE // 1000,
    )
    if len(samples) == 0:
        return samples
    # The VAD model keeps state between windows, so one call at a time.
    with vad_lock:
        timestamps = get_speech_timestamps(
            torch.from_numpy(samples),
            vad_model,
            sampling_rate=WAV_SAMPLE_RATE,
            return_seconds=False,
        )
    if not timestamps:
        return samples[:0]
    return np.concatenate([samples[item["start"]:item["end"]] for item in timestamps])
