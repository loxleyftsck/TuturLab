"""Standalone, local-only Qwen3 ASR application. No BaliCall services required."""
import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import threading
import time

ROOT = Path(__file__).resolve().parent
for name, value in {
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
    'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'GRADIO_ANALYTICS_ENABLED': 'False',
    'DO_NOT_TRACK': '1', 'GRADIO_TEMP_DIR': str(ROOT / 'cache' / 'uploads'),
    'HF_HOME': str(ROOT / 'cache' / 'huggingface'),
}.items():
    os.environ[name] = value

import gradio as gr
import torch
from qwen_asr import Qwen3ASRModel

from offline_asr import AudioInputError, LANGUAGES, MAX_BYTES, decode_audio_file, process_audio


def make_handler(recognize, output_root):
    """Stream durable partial results; keep the UI usable after input/inference errors."""
    active_jobs = {}
    jobs_lock = threading.Lock()

    def stop_job(request: gr.Request):
        with jobs_lock:
            event = active_jobs.get(request.session_hash)
            if event is None:
                return 'Tidak ada proses aktif pada sesi ini.'
            event.set()
        return 'Penghentian diminta. Tunggu potongan yang sedang berjalan selesai. Hasil yang sudah tersimpan tetap ada di folder output.'

    def transcribe(filename, language, request: gr.Request, progress=gr.Progress()):
        latest = None
        iterator = None
        stop_event = threading.Event()
        session = request.session_hash
        with jobs_lock:
            active_jobs[session] = stop_event
        try:
            yield '', 'Membaca audio dan memeriksa durasi…', gr.update(value=None, interactive=False), {}
            if not filename:
                raise AudioInputError('Unggah audio atau rekam suara terlebih dahulu.')
            audio = decode_audio_file(filename)
            duration = len(audio) / 16000
            yield '', f'Audio {duration / 60:.2f} menit siap. Memproses potongan pertama…', gr.update(value=None, interactive=False), {'duration_seconds': duration}
            iterator = process_audio(audio, language, recognize, output_root, should_stop=stop_event.is_set)
            for latest in iterator:
                fraction = latest['processed_seconds'] / latest['duration_seconds']
                progress(fraction, desc=f"{fraction:.0%} audio diperiksa")
                detail = {key: value for key, value in latest.items() if key not in ('text', 'segments', 'status_path', 'text_path')}
                if latest['status'] == 'completed':
                    status = (f"Selesai · {latest['duration_seconds'] / 60:.2f} menit audio · "
                              f"waktu proses {latest['elapsed_seconds']:.1f} detik.")
                    if not latest['text']:
                        status += ' Tidak ada ucapan yang berhasil dikenali.'
                elif latest['status'] == 'cancelled':
                    status = f"Dihentikan pada {latest['processed_seconds'] / 60:.2f} menit. Hasil parsial tersimpan dan bisa diunduh."
                else:
                    status = (f"Memproses {fraction:.0%} · {latest['processed_seconds'] / 60:.2f} / "
                              f"{latest['duration_seconds'] / 60:.2f} menit audio. Hasil parsial tersimpan.")
                yield latest['text'], status, gr.update(value=latest['text_path'], interactive=True), detail
        except AudioInputError as error:
            yield '', str(error), gr.update(value=None, interactive=False), {'status': 'invalid_input'}
        except Exception:
            logging.exception('Transcription failed')
            detail = {'status': 'failed'}
            if latest:
                detail.update(job_id=latest['job_id'], processed_seconds=latest['processed_seconds'])
            yield (latest['text'] if latest else ''), ('Pemrosesan gagal. Hasil potongan yang sudah selesai tetap tersimpan di folder output. '
                'Coba audio lebih pendek atau periksa logs/app.log.'), gr.update(
                    value=latest['text_path'] if latest else None, interactive=bool(latest)), detail
        finally:
            if iterator is not None:
                iterator.close()
            with jobs_lock:
                if active_jobs.get(session) is stop_event:
                    active_jobs.pop(session, None)
    return transcribe, stop_job


def main():
    parser = argparse.ArgumentParser(description='ASR Offline, Qwen3-ASR-0.6B')
    parser.add_argument('--port', type=int)
    args = parser.parse_args()
    config = json.loads((ROOT / 'config.json').read_text(encoding='utf8'))
    for folder in ('output', 'logs', 'cache/uploads'):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, force=True, handlers=[
        RotatingFileHandler(ROOT / 'logs' / 'app.log', maxBytes=2_000_000, backupCount=2, encoding='utf8'),
        logging.StreamHandler(),
    ], format='%(asctime)s %(levelname)s %(message)s')
    torch.set_num_threads(max(1, min(8, int(config.get('cpu_threads', 4)))))
    logging.info('Loading local Qwen3-ASR-0.6B on CPU/float32')
    started = time.perf_counter()
    model = Qwen3ASRModel.from_pretrained(
        str(ROOT / 'models' / 'Qwen3-ASR-0.6B'), dtype=torch.float32,
        device_map='cpu', attn_implementation='eager', max_inference_batch_size=1,
        max_new_tokens=512, local_files_only=True,
    )
    logging.info('Model ready in %.1f seconds', time.perf_counter() - started)
    inference_lock = threading.Lock()

    def recognize(audio, language):
        # A cancelled UI request may finish its current CPU call; never overlap models.
        with inference_lock, torch.inference_mode():
            return model.transcribe(audio=(audio, 16000), language=language, return_time_stamps=False)[0].text

    with gr.Blocks(title='ASR Offline', analytics_enabled=False) as app:
        gr.Markdown('# ASR Offline\nUbah suara menjadi teks dengan **Qwen3-ASR-0.6B** di komputer ini.')
        gr.Markdown('**Maksimal 15 menit / 200 MB per file.** Audio diproses berurutan dalam potongan hingga 30 detik; '
            'hasil tersimpan setiap potongan. Waktu proses CPU dapat melebihi durasi audio.')
        audio = gr.Audio(sources=['upload', 'microphone'], type='filepath', format=None, label='Unggah atau rekam audio')
        language = gr.Dropdown(list(LANGUAGES), value='Indonesian', label='Bahasa ucapan')
        with gr.Row():
            start = gr.Button('Transkripsikan', variant='primary')
            stop = gr.Button('Hentikan proses', variant='stop')
        status = gr.Markdown('Siap · model lokal · tidak memerlukan layanan BaliCall.', elem_id='process-status')
        result = gr.Textbox(label='Hasil transkripsi', lines=10, max_lines=20, interactive=False)
        download = gr.DownloadButton('Unduh teks (.txt)', interactive=False)
        with gr.Accordion('Detail proses', open=False):
            metadata = gr.JSON(label='Informasi pekerjaan')
        gr.Markdown('Hasil lengkap/parsial: folder **output** di sebelah aplikasi. Hentikan proses bekerja setelah potongan '
            'yang sedang berjalan selesai; teks yang sudah tersimpan tetap tersedia.\n\n'
            'Aplikasi ini menghasilkan teks; belum menyediakan nama pembicara, diarization, atau timestamp per kata. '
            'Berkas unggahan dan hasil tetap tersimpan lokal sampai Anda menghapusnya sendiri.')
        transcribe, stop_job = make_handler(recognize, ROOT / 'output')
        start.click(transcribe, [audio, language],
            [result, status, download, metadata], concurrency_limit=1, concurrency_id='asr', api_name='transcribe')
        stop.click(stop_job, outputs=status, queue=False, api_name='stop')
    port = args.port or int(config.get('port', 5197))
    print(f'Buka http://127.0.0.1:{port}/ - tutup terminal untuk menghentikan aplikasi.', flush=True)
    app.queue(default_concurrency_limit=1, max_size=2).launch(
        server_name='127.0.0.1', server_port=port, share=False, inbrowser=False,
        max_file_size=MAX_BYTES, enable_monitoring=False, footer_links=[],
        allowed_paths=[str(ROOT / 'output')],
        blocked_paths=[str(ROOT / 'models'), str(ROOT / '.venv'), str(ROOT / 'logs')],
        theme=gr.themes.Soft(font=['Arial', 'sans-serif'], font_mono=['Consolas', 'monospace']),
    )


if __name__ == '__main__':
    main()
