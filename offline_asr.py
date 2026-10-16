"""Bounded local audio decoding and sequential ASR with durable partial output."""
import json
import time
from pathlib import Path
from uuid import uuid4

import av
import numpy as np
from faster_whisper.vad import VadOptions, get_speech_timestamps

SAMPLE_RATE = 16000
MAX_SECONDS = 900
MAX_BYTES = 200 * 1024 * 1024
LANGUAGES = ('Indonesian', 'English')


class AudioInputError(ValueError):
    pass


def decode_audio_file(filename):
    path = Path(filename)
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_BYTES:
        raise AudioInputError('Pilih file audio yang tidak kosong, maksimal 200 MB.')
    parts = []
    count = 0
    try:
        with av.open(str(path), mode='r') as container:
            if not container.streams.audio:
                raise AudioInputError('File tidak memiliki track audio.')
            resampler = av.AudioResampler(format='fltp', layout='mono', rate=SAMPLE_RATE)

            def append(frame):
                nonlocal count
                values = frame.to_ndarray().reshape(-1).astype(np.float32, copy=False)
                count += len(values)
                if count > SAMPLE_RATE * MAX_SECONDS:
                    raise AudioInputError('Durasi audio maksimal 15 menit. Potong file yang lebih panjang terlebih dahulu.')
                if not np.isfinite(values).all():
                    raise AudioInputError('Audio berisi sampel yang tidak valid.')
                parts.append(values.copy())

            for frame in container.decode(audio=0):
                for converted in resampler.resample(frame):
                    append(converted)
            for converted in resampler.resample(None):
                append(converted)
    except AudioInputError:
        raise
    except Exception as error:
        raise AudioInputError('Audio tidak bisa dibaca. Coba WAV, MP3, FLAC, M4A, OGG, atau WebM yang valid.') from error
    if not count:
        raise AudioInputError('Audio tidak berisi sampel suara.')
    return np.concatenate(parts)


def iter_chunks(audio, sample_rate=SAMPLE_RATE, chunk_seconds=30):
    maximum = int(sample_rate * chunk_seconds)
    if maximum < 1:
        raise ValueError('Chunk size must be positive')
    start = 0
    while start < len(audio):
        end = min(start + maximum, len(audio))
        if end < len(audio):
            # Choose a quiet boundary in the final second; never overlap/drop samples.
            left = max(start + 1, end - sample_rate)
            region = np.abs(audio[left:end])
            window = max(1, sample_rate // 10)
            if len(region) >= window:
                cumulative = np.concatenate(([0.0], np.cumsum(region, dtype=np.float64)))
                energy = (cumulative[window:] - cumulative[:-window]) / window
                quiet = int(np.argmin(energy))
                if energy[quiet] < 0.003:
                    end = left + quiet + window // 2
        yield start, end, audio[start:end]
        start = end


def detect_speech(audio):
    return get_speech_timestamps(audio, VadOptions(
        threshold=0.5, min_speech_duration_ms=120,
        min_silence_duration_ms=300, speech_pad_ms=400,
    ))


def write_atomic(path, content):
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(content, encoding='utf8')
    temporary.replace(path)


def process_audio(audio, language, recognize, output_root, speech_detector=detect_speech, should_stop=lambda: False):
    if language not in LANGUAGES:
        raise AudioInputError('Pilih bahasa Indonesia atau English.')
    if audio.ndim != 1 or not 0 < len(audio) <= SAMPLE_RATE * MAX_SECONDS:
        raise AudioInputError('Durasi audio maksimal 15 menit dan tidak boleh kosong.')
    job = Path(output_root) / uuid4().hex
    job.mkdir(parents=True, exist_ok=False)
    text_path, status_path = job / 'transcript.txt', job / 'status.json'
    started = time.perf_counter()
    state = {'job_id': job.name, 'status': 'processing', 'language': language,
        'duration_seconds': len(audio) / SAMPLE_RATE, 'processed_seconds': 0,
        'chunks_processed': 0, 'speech_chunks': 0, 'text': '',
        'text_path': str(text_path), 'status_path': str(status_path), 'segments': []}
    paragraphs = []

    def persist():
        state['text'] = '\n\n'.join(paragraphs)
        state['elapsed_seconds'] = round(time.perf_counter() - started, 3)
        write_atomic(text_path, state['text'])
        write_atomic(status_path, json.dumps(state, ensure_ascii=False, indent=2))

    persist()
    try:
        for start, end, chunk in iter_chunks(audio):
            if should_stop():
                state['status'] = 'cancelled'
                persist()
                yield dict(state, segments=[dict(segment) for segment in state['segments']])
                return
            regions = speech_detector(chunk)
            if regions:
                first = max(0, int(regions[0]['start']))
                last = min(len(chunk), int(regions[-1]['end']))
                if last > first:
                    text = recognize(chunk[first:last], language)
                    if not isinstance(text, str):
                        raise RuntimeError('ASR returned invalid text')
                    state['speech_chunks'] += 1
                    if text.strip():
                        paragraphs.append(text.strip())
                        state['segments'].append({'chunk_start_seconds': start / SAMPLE_RATE,
                            'chunk_end_seconds': end / SAMPLE_RATE, 'text': text.strip()})
            state['chunks_processed'] += 1
            state['processed_seconds'] = end / SAMPLE_RATE
            if end == len(audio):
                state['status'] = 'completed'
            elif should_stop():
                state['status'] = 'cancelled'
            persist()
            yield dict(state, segments=[dict(segment) for segment in state['segments']])
            if state['status'] == 'cancelled':
                return
    except GeneratorExit:
        if state['status'] != 'completed':
            state['status'] = 'cancelled'
            persist()
        raise
    except Exception:
        state['status'] = 'failed'
        persist()
        raise
