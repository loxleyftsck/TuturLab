import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import wave

import numpy as np

from app import make_handler


class HandlerTests(unittest.TestCase):
    def test_stop_is_scoped_to_browser_session_and_next_job_can_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_path = root / 'silence.wav'
            with wave.open(str(audio_path), 'wb') as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(bytes(16000 * 65 * 2))
            handler, stop = make_handler(lambda *_: self.fail('Silence reached ASR'), root / 'output')
            request = SimpleNamespace(session_hash='first-session')
            other = SimpleNamespace(session_hash='other-session')
            job = handler(str(audio_path), 'English', request, progress=lambda *a, **kw: None)
            next(job)
            self.assertIn('Tidak ada', stop(other))
            self.assertIn('Penghentian diminta', stop(request))
            updates = list(job)
            final = updates[-1]
            self.assertEqual(final[3]['status'], 'cancelled')
            self.assertEqual(final[3]['processed_seconds'], 0)
            self.assertTrue(Path(final[2]['value']).is_file())
            self.assertIn('Tidak ada', stop(request))
            another = list(handler(str(audio_path), 'English', request, progress=lambda *a, **kw: None))
            self.assertEqual(another[-1][3]['status'], 'completed')
            self.assertEqual(another[-1][0], '')

    def test_missing_audio_is_friendly_and_cleans_active_job(self):
        with tempfile.TemporaryDirectory() as directory:
            handler, stop = make_handler(lambda *_: '', Path(directory))
            request = SimpleNamespace(session_hash='missing-audio')
            updates = list(handler(None, 'Indonesian', request))
            self.assertEqual(updates[-1][3]['status'], 'invalid_input')
            self.assertIn('Unggah audio', updates[-1][1])
            self.assertIn('Tidak ada', stop(request))


if __name__ == '__main__':
    unittest.main()
