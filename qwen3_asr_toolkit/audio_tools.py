import os
import librosa
import subprocess
import tempfile
import numpy as np
import soundfile as sf
import traceback

from silero_vad import get_speech_timestamps


WAV_SAMPLE_RATE = 16000


def _debug_enabled() -> bool:
    return os.getenv("QWEN3_ASR_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _debug_log(message: str) -> None:
    if _debug_enabled():
        print(f"[qwen3-asr-debug][pid={os.getpid()}] {message}", flush=True)


def _validate_librosa_audio(wav_data: np.ndarray) -> None:
    if wav_data is None or len(wav_data) == 0:
        raise ValueError("librosa returned empty audio.")
    if np.isnan(wav_data).any() or np.isinf(wav_data).any():
        raise ValueError("librosa returned NaN/Inf values.")
    if np.all(wav_data == 0):
        raise ValueError("librosa returned all-zero audio.")


def load_audio(file_path: str) -> np.ndarray:
    recovered_path = None
    try:
        _debug_log(f"Using librosa to load audio from local file")
        wav_data, _ = librosa.load(file_path, sr=WAV_SAMPLE_RATE, mono=True)
        _validate_librosa_audio(wav_data)
        return wav_data
    except Exception as librosa_e:
        try:
            _debug_log(f"librosa decode rejected: {librosa_e}")
            with tempfile.NamedTemporaryFile(suffix="_recovered.wav", delete=False) as tmp:
                recovered_path = tmp.name
            _debug_log(f"Using ffmpeg to recreate audio")
            command = [
                "ffmpeg",
                "-y",
                "-i", file_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", str(WAV_SAMPLE_RATE),
                "-ac", "1",
                recovered_path,
            ]
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if process.returncode != 0:
                stderr_text = process.stderr.decode("utf-8", errors="ignore")
                raise RuntimeError(f"FFmpeg error recreating audio: {stderr_text}")

            wav_data, _ = sf.read(recovered_path, dtype="float32")
            if isinstance(wav_data, np.ndarray) and wav_data.ndim > 1:
                wav_data = wav_data.mean(axis=1)
            return wav_data
        except Exception as ffmpeg_e:
            raise RuntimeError(
                f"Failed to load audio from '{file_path}'. "
                f"librosa_error={librosa_e}; ffmpeg_error={ffmpeg_e}"
            )
        finally:
            if recovered_path and os.path.exists(recovered_path):
                try:
                    os.remove(recovered_path)
                except OSError:
                    pass


def process_vad(wav: np.ndarray, worker_vad_model, segment_threshold_s: int = 120, max_segment_threshold_s: int = 180) -> list[np.ndarray]:
    try:
        _debug_log(
            f"process_vad start: wav_len={len(wav)}, segment_threshold_s={segment_threshold_s}, "
            f"max_segment_threshold_s={max_segment_threshold_s}"
        )
        vad_params = {
            'sampling_rate': WAV_SAMPLE_RATE,
            'return_seconds': False,
            'min_speech_duration_ms': 1500,
            'min_silence_duration_ms': 500
        }

        speech_timestamps = get_speech_timestamps(
            wav,
            worker_vad_model,
            **vad_params
        )
        _debug_log(f"speech_timestamps count={len(speech_timestamps) if speech_timestamps else 0}")
        if speech_timestamps:
            preview = speech_timestamps[:5]
            _debug_log(f"speech_timestamps preview={preview}")

        if not speech_timestamps:
            raise ValueError("No speech segments detected by VAD.")

        potential_split_points_s = {0.0, len(wav)}
        for i in range(len(speech_timestamps)):
            start_of_next_s = speech_timestamps[i]['start']
            if start_of_next_s is None:
                _debug_log(f"speech_timestamps[{i}]['start'] is None; full_item={speech_timestamps[i]}")
            potential_split_points_s.add(start_of_next_s)
        sorted_potential_splits = sorted(list(potential_split_points_s))

        final_split_points_s = {0.0, len(wav)}
        segment_threshold_samples = segment_threshold_s * WAV_SAMPLE_RATE
        target_time = segment_threshold_samples
        while target_time < len(wav):
            closest_point = min(sorted_potential_splits, key=lambda p: abs(p - target_time))
            final_split_points_s.add(closest_point)
            target_time += segment_threshold_samples
        final_ordered_splits = sorted(list(final_split_points_s))

        max_segment_threshold_samples = max_segment_threshold_s * WAV_SAMPLE_RATE
        new_split_points = [0.0]

        # Make sure that each audio segment does not exceed max_segment_threshold_s
        for i in range(1, len(final_ordered_splits)):
            start = final_ordered_splits[i - 1]
            end = final_ordered_splits[i]
            segment_length = end - start

            if segment_length <= max_segment_threshold_samples:
                new_split_points.append(end)
            else:
                num_subsegments = int(np.ceil(segment_length / max_segment_threshold_samples))
                subsegment_length = segment_length / num_subsegments

                for j in range(1, num_subsegments):
                    split_point = start + j * subsegment_length
                    new_split_points.append(split_point)

                new_split_points.append(end)

        segmented_wavs = []
        for i in range(len(new_split_points) - 1):
            start_sample = int(new_split_points[i])
            end_sample = int(new_split_points[i + 1])
            segmented_wavs.append((start_sample, end_sample, wav[start_sample:end_sample]))
        return segmented_wavs

    except Exception as e:
        _debug_log(f"process_vad exception: {e.__class__.__name__}: {e}")
        _debug_log(traceback.format_exc())
        segmented_wavs = []
        total_samples = len(wav)
        max_chunk_size_samples = max_segment_threshold_s * WAV_SAMPLE_RATE
        _debug_log(
            f"fallback chunking: total_samples={total_samples}, max_chunk_size_samples={max_chunk_size_samples}"
        )

        for start_sample in range(0, total_samples, max_chunk_size_samples):
            end_sample = min(start_sample + max_chunk_size_samples, total_samples)
            segment = wav[start_sample:end_sample]
            if len(segment) > 0:
                segmented_wavs.append((start_sample, end_sample, segment))
        _debug_log(f"fallback chunking produced segments={len(segmented_wavs)}")

        return segmented_wavs


def save_audio_file(wav: np.ndarray, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sf.write(file_path, wav, WAV_SAMPLE_RATE)
