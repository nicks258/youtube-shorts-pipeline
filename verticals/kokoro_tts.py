"""Kokoro TTS provider — local open-source neural TTS (Apache 2.0).

Generates high-quality speech locally using the 82M-parameter Kokoro model
via KPipeline at 24 kHz, converted to MP3 for the Shorts pipeline.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path

try:
    from .log import log
except ImportError:
    def log(msg: str):
        print(msg)

_PIPELINES: dict = {}


def _get_pipeline(lang_code: str):
    if lang_code not in _PIPELINES:
        from kokoro import KPipeline
        _PIPELINES[lang_code] = KPipeline(lang_code=lang_code)
    return _PIPELINES[lang_code]


def _split_and_merge_sentences(script: str) -> list[str]:
    """Split script into sentences after . ! ? followed by space, merging sentences < 4 words."""
    raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", script.strip()) if s.strip()]
    if not raw_sentences:
        return []

    merged: list[str] = []
    buffer = ""
    for s in raw_sentences:
        candidate = f"{buffer} {s}".strip() if buffer else s
        if len(candidate.split()) < 4:
            buffer = candidate
        else:
            merged.append(candidate)
            buffer = ""

    if buffer:
        if merged:
            merged[-1] = f"{merged[-1]} {buffer}"
        else:
            merged.append(buffer)

    return merged


def _trim_silence(audio, threshold: float = 0.005):
    """Trim silence from the start and end of audio array."""
    import numpy as np
    abs_audio = np.abs(audio)
    max_val = np.max(abs_audio) if len(abs_audio) > 0 else 0.0
    if max_val == 0.0:
        return audio
    thresh = min(threshold, float(max_val * 0.1))
    indices = np.where(abs_audio > thresh)[0]
    if len(indices) == 0:
        return audio
    return audio[indices[0]:indices[-1] + 1]


def _stretch_audio(audio, factor: float, sample_rate: int = 24000):
    """Time-stretch audio with ffmpeg's atempo filter."""
    if abs(factor - 1.0) < 0.001:
        return audio

    import numpy as np
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as in_file, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_file:
        in_path = in_file.name
        out_path = out_file.name

    try:
        sf.write(in_path, audio, sample_rate)
        subprocess.run(
            [
                "ffmpeg", "-i", in_path,
                "-filter:a", f"atempo={factor:.4f}",
                out_path, "-y", "-loglevel", "error"
            ],
            check=True,
        )
        stretched, _ = sf.read(out_path)
        return stretched.astype(np.float32)
    finally:
        Path(in_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)


def generate_kokoro(script: str, out_dir: Path, lang: str = "en") -> Path:
    """Generate voiceover via Kokoro TTS.

    Args:
        script: The text to speak.
        out_dir: Directory where the output mp3 will be saved.
        lang: Language identifier (default "en").

    Returns:
        Path to out_dir/voiceover_{lang}.mp3.
    """
    import numpy as np
    import soundfile as sf

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    voice = os.environ.get("KOKORO_VOICE", "am_michael").strip() or "am_michael"
    speed = float(os.environ.get("KOKORO_SPEED", "1.0"))

    # "a" for American English, "b" for British English (or KOKORO_LANG if set)
    lang_code = os.environ.get("KOKORO_LANG", "").strip() or ("b" if voice.startswith("b") else "a")

    if lang_code == "b":
        log(f"Generating {lang} voiceover via Kokoro (voice: {voice}, speed: {speed}) -- British pronunciation mode (lang_code \"b\")...")
    else:
        log(f"Generating {lang} voiceover via Kokoro (voice: {voice}, speed: {speed})...")

    sentences = _split_and_merge_sentences(script)
    if not sentences:
        raise ValueError("Script contains no speakable sentences")

    pipeline = _get_pipeline(lang_code)

    sample_rate = 24000
    sentence_records = []

    for sentence in sentences:
        chunks = []
        generator = pipeline(sentence, voice=voice, speed=speed)
        for result in generator:
            audio = getattr(result, "audio", None)
            if audio is None and isinstance(result, (tuple, list)) and len(result) >= 3:
                audio = result[2]
            if audio is not None:
                if hasattr(audio, "detach"):
                    audio = audio.detach().cpu().numpy()
                elif hasattr(audio, "numpy"):
                    audio = audio.numpy()
                chunks.append(audio)

        if not chunks:
            continue

        raw_sentence_audio = np.concatenate(chunks)
        trimmed_audio = _trim_silence(raw_sentence_audio)

        words = len(sentence.split())
        seconds = max(len(trimmed_audio) / sample_rate, 0.01)
        rate = words / seconds

        sentence_records.append({
            "text": sentence,
            "audio": trimmed_audio,
            "words": words,
            "rate": rate,
        })

    if not sentence_records:
        raise RuntimeError("Kokoro produced no audio")

    # Determine target rate
    env_wps = os.environ.get("KOKORO_WPS", "").strip()
    if env_wps:
        target_rate = float(env_wps)
    else:
        all_rates = [rec["rate"] for rec in sentence_records]
        target_rate = float(np.median(all_rates))
        target_rate = max(2.4, min(3.0, target_rate))

    # Time-stretch sentences if rate differs from target rate
    processed_sentences = []
    for i, rec in enumerate(sentence_records, start=1):
        orig_rate = rec["rate"]
        factor = target_rate / orig_rate
        clamped_factor = max(0.85, min(1.15, factor))

        stretched_audio = _stretch_audio(rec["audio"], clamped_factor, sample_rate)
        final_seconds = max(len(stretched_audio) / sample_rate, 0.01)
        final_rate = rec["words"] / final_seconds

        log(f"sentence {i}: {orig_rate:.1f} words/s -> {final_rate:.1f} words/s")

        processed_sentences.append({
            "text": rec["text"],
            "audio": stretched_audio,
        })

    # Join sentences with silence:
    # 0.15s start and end, 0.35s after ? or !, 0.25s after others
    silence_start = np.zeros(int(round(0.15 * sample_rate)), dtype=np.float32)
    silence_end = np.zeros(int(round(0.15 * sample_rate)), dtype=np.float32)

    pieces = [silence_start]
    for i, rec in enumerate(processed_sentences):
        pieces.append(rec["audio"])
        if i < len(processed_sentences) - 1:
            clean_text = rec["text"].strip().rstrip("\"'”’")
            gap_sec = 0.35 if (clean_text.endswith("?") or clean_text.endswith("!")) else 0.22
            gap_audio = np.zeros(int(round(gap_sec * sample_rate)), dtype=np.float32)
            pieces.append(gap_audio)
    pieces.append(silence_end)

    full_audio = np.concatenate(pieces)

    wav_path = out_dir / f"_temp_kokoro_{lang}.wav"
    mp3_path = out_dir / f"voiceover_{lang}.mp3"

    sf.write(str(wav_path), full_audio, sample_rate)
    try:
        subprocess.run(
            [
                "ffmpeg", "-i", str(wav_path),
                "-filter:a", "loudnorm=I=-14:TP=-1.5",
                "-codec:a", "libmp3lame", "-b:a", "192k",
                str(mp3_path), "-y", "-loglevel", "error"
            ],
            check=True,
        )
    finally:
        wav_path.unlink(missing_ok=True)

    log(f"Kokoro voiceover saved: {mp3_path.name}")
    return mp3_path
