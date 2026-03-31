import argparse
import os
import srt

from tqdm import tqdm
from datetime import timedelta
from urllib.parse import urlparse
from qwen3_asr_toolkit.audio_tools import WAV_SAMPLE_RATE
from qwen3_asr_toolkit.pipeline import run_transcription_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Python toolkit for the Qwen3-ASR API—parallel high‑throughput calls, robust long‑audio transcription, multi‑sample‑rate support."
    )
    parser.add_argument("--input-file", '-i', type=str, required=True, help="Input media file path")
    parser.add_argument("--context", '-c', type=str, default="Transcribe with punctuation. Preserve sentence meaning across pauses.", help="Optional text context for ASR")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/v1/audio/transcriptions", help="ASR API endpoint (OpenAI-compatible)")
    parser.add_argument("--model", type=str, default=None, help="Model name for the API")
    parser.add_argument("--api-timeout", type=int, default=300, help="API request timeout (seconds)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--skip-failed", action="store_true", help="Skip failed segments instead of aborting")
    parser.add_argument("--max-retries", type=int, default=10, help="Max retries per segment")
    parser.add_argument("--num-threads", '-j', type=int, default=4, help="Number of threads to use for parallel calls")
    parser.add_argument("--vad-segment-threshold", '-d', type=int, default=60, help="Segment threshold seconds for VAD")
    parser.add_argument("--max-segment-seconds", type=int, default=120, help="Hard limit for segment length (seconds)")
    parser.add_argument("--tmp-dir", '-t', type=str, default=os.path.join(os.path.expanduser("~"), "qwen3-asr-cache"), help="Temp directory path")
    parser.add_argument("--save-srt", '-srt', action="store_true", help="Save SRT subtitle file")
    parser.add_argument("--silence", '-s', action="store_true", help="Reduce the output info on the terminal")
    return parser.parse_args()


def main():
    args = parse_args()
    input_file = args.input_file
    context = args.context
    api_url = args.api_url
    model = args.model
    api_timeout = args.api_timeout
    temperature = args.temperature
    skip_failed = args.skip_failed
    max_retries = args.max_retries
    num_threads = args.num_threads
    vad_segment_threshold = args.vad_segment_threshold
    max_segment_seconds = args.max_segment_seconds
    tmp_dir = args.tmp_dir
    save_srt = args.save_srt
    silence = args.silence

    pbar_state = {"bar": None, "done": 0}

    def _log_callback(message: str):
        if not silence:
            print(message)

    def _progress_callback(done: int, total: int):
        if silence:
            return
        if pbar_state["bar"] is None:
            pbar_state["bar"] = tqdm(total=total, desc="Calling Qwen3-ASR API")
        delta = done - pbar_state["done"]
        if delta > 0:
            pbar_state["bar"].update(delta)
            pbar_state["done"] = done

    try:
        pipeline_result = run_transcription_pipeline(
            input_file=input_file,
            context=context,
            api_url=api_url,
            model=model,
            api_timeout=api_timeout,
            temperature=temperature,
            skip_failed=skip_failed,
            max_retries=max_retries,
            num_threads=num_threads,
            vad_segment_threshold=vad_segment_threshold,
            max_segment_seconds=max_segment_seconds,
            tmp_dir=tmp_dir,
            progress_callback=_progress_callback,
            log_callback=_log_callback,
        )
    finally:
        if pbar_state["bar"] is not None:
            pbar_state["bar"].close()

    full_text = pipeline_result.full_text
    language = pipeline_result.detected_language

    if not silence:
        print(f"Detected Language: {language}")
        print(f"Full Transcription: {full_text}")

    # Save full text to local file
    if os.path.exists(input_file):
        save_file = os.path.splitext(input_file)[0] + ".txt"
    else:
        save_file = os.path.splitext(urlparse(input_file).path)[0].split('/')[-1] + '.txt'

    with open(save_file, 'w') as f:
        f.write(language + '\n')
        f.write(full_text + '\n')

    print(f"Full transcription of \"{input_file}\" saved to \"{save_file}\"!")

    # Save subtitles to local SRT file
    if args.save_srt:
        subtitles = []
        for idx, segment in enumerate(pipeline_result.segments):
            start_time = segment.start_sample / WAV_SAMPLE_RATE
            end_time = segment.end_sample / WAV_SAMPLE_RATE
            content = segment.text
            subtitles.append(srt.Subtitle(
                index=idx,
                start=timedelta(seconds=start_time),
                end=timedelta(seconds=end_time),
                content=content
            ))
        final_srt_content = srt.compose(subtitles)
        srt_path = os.path.splitext(save_file)[0] + ".srt"
        with open(srt_path, 'w') as f:
            f.write(final_srt_content)
        print(f"SRT subtitles of \"{input_file}\" saved to \"{srt_path}\"!")


if __name__ == '__main__':
    main()
