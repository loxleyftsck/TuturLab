import io
import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from offline_asr import AudioInputError, decode_audio_file, iter_chunks, process_audio


def make_wav(path, seconds, rate=16000, channels=1):
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        remaining = int(seconds * rate)
        block = bytes(rate * channels * 2)
        while remaining:
            count = min(remaining, rate)
            stream.writeframes(block[:count * channels * 2])
            remaining -= count


class PipelineTests(unittest.TestCase):
    def test_accepts_fifteen_minutes_and_rejects_one_second_more(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'long.wav'
            make_wav(filename, 900)
            audio = decode_audio_file(filename)
            self.assertEqual(len(audio), 14_400_000)
            make_wav(filename, 901)
            with self.assertRaisesRegex(AudioInputError, '15 menit'):
                decode_audio_file(filename)

    def test_resamples_stereo_and_rejects_corrupt_or_empty_files(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'stereo.wav'
            make_wav(filename, 1, rate=48000, channels=2)
            self.assertEqual(decode_audio_file(filename).shape, (16000,))
            filename.write_bytes(b'not audio')
            with self.assertRaises(AudioInputError):
                decode_audio_file(filename)
            make_wav(filename, 0)
            with self.assertRaises(AudioInputError):
                decode_audio_file(filename)

    def test_chunks_preserve_every_sample_and_never_exceed_thirty_seconds(self):
        audio = np.arange(1001, dtype=np.float32)
        chunks = list(iter_chunks(audio, sample_rate=10, chunk_seconds=30))
        np.testing.assert_array_equal(np.concatenate([chunk for _, _, chunk in chunks]), audio)
        self.assertTrue(all(0 < len(chunk) <= 300 for _, _, chunk in chunks))
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], 1001)
        self.assertTrue(all(chunks[i][1] == chunks[i + 1][0] for i in range(len(chunks) - 1)))

    def test_long_silence_finishes_without_calling_asr_and_saves_empty_result(self):
        with tempfile.TemporaryDirectory() as directory:
            def forbidden(*args):
                self.fail('Silent audio reached the model')
            updates = list(process_audio(np.zeros(16000 * 65, dtype=np.float32), 'Indonesian', forbidden, Path(directory)))
            final = updates[-1]
            self.assertEqual(final['status'], 'completed')
            self.assertEqual(final['processed_seconds'], 65)
            self.assertEqual(final['text'], '')
            self.assertEqual(Path(final['text_path']).read_text(encoding='utf8'), '')

    def test_partial_results_are_persisted_and_report_complete_only_after_last_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = iter(['Bagian pertama.', 'Bagian kedua.', 'Bagian terakhir.'])
            updates = list(process_audio(np.ones(16000 * 61, dtype=np.float32), 'Indonesian',
                lambda *_: next(calls), Path(directory), speech_detector=lambda audio: [{'start': 0, 'end': len(audio)}]))
            self.assertEqual(updates[-1]['text'], 'Bagian pertama.\n\nBagian kedua.\n\nBagian terakhir.')
            self.assertEqual(len(updates[0]['segments']), 1)
            self.assertEqual(len(updates[-1]['segments']), 3)
            self.assertTrue(all(update['status'] == 'processing' for update in updates[:-1]))
            self.assertEqual(updates[-1]['processed_seconds'], 61)
            self.assertEqual(json.loads(Path(updates[-1]['status_path']).read_text())['status'], 'completed')

    def test_failure_preserves_finished_text_and_never_reports_success(self):
        with tempfile.TemporaryDirectory() as directory:
            def recognize(*args):
                if not hasattr(recognize, 'called'):
                    recognize.called = True
                    return 'Hasil pertama.'
                raise RuntimeError('Fixture inference failed')
            updates = process_audio(np.ones(16000 * 61, dtype=np.float32), 'Indonesian', recognize,
                Path(directory), speech_detector=lambda audio: [{'start': 0, 'end': len(audio)}])
            first = next(updates)
            with self.assertRaises(RuntimeError):
                list(updates)
            self.assertEqual(Path(first['text_path']).read_text(encoding='utf8'), 'Hasil pertama.')
            self.assertEqual(json.loads(Path(first['status_path']).read_text())['status'], 'failed')

    def test_cancellation_keeps_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            updates = process_audio(np.ones(16000 * 61, dtype=np.float32), 'English', lambda *_: 'Saved words.',
                Path(directory), speech_detector=lambda audio: [{'start': 0, 'end': len(audio)}])
            first = next(updates)
            updates.close()
            self.assertEqual(Path(first['text_path']).read_text(encoding='utf8'), 'Saved words.')
            self.assertEqual(json.loads(Path(first['status_path']).read_text())['status'], 'cancelled')

    def test_stop_requested_during_inference_finishes_current_chunk_and_saves_cancelled_state(self):
        with tempfile.TemporaryDirectory() as directory:
            stopped = []
            def recognize(*args):
                stopped.append(True)
                return 'Current chunk saved.'
            updates = list(process_audio(np.ones(16000 * 61, dtype=np.float32), 'English', recognize,
                Path(directory), speech_detector=lambda audio: [{'start': 0, 'end': len(audio)}],
                should_stop=lambda: bool(stopped)))
            self.assertEqual(len(stopped), 1)
            self.assertEqual(updates[-1]['status'], 'cancelled')
            self.assertEqual(updates[-1]['processed_seconds'], 30)
            self.assertEqual(Path(updates[-1]['text_path']).read_text(encoding='utf8'), 'Current chunk saved.')
            self.assertEqual(json.loads(Path(updates[-1]['status_path']).read_text())['status'], 'cancelled')


if __name__ == '__main__':
    unittest.main()
