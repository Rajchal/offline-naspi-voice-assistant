import os
import sys
import struct
import subprocess
import time
import signal
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
from queue import Queue
try:
    import pyaudio
    import pvporcupine
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Install with: pip3 install pyaudio pvporcupine")
    sys.exit(1)

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    frame_length: int = 512
    channels: int = 1
    format: int = pyaudio.paInt16
    recording_duration: int = 5
    audio_timeout: int = 30

@dataclass
class PorcupineConfig:
    access_key: str ="CnNEQfm996S877kY+Ml+GSSqdOb/IgW5CKVUSXzasBWK8+SRlwfeDg=="
    keyword_file: str = ""  # Leave blank to use built-in keywords
    fallback_keyword: str = "computer"
    sensitivity: float = 0.5

@dataclass
class WhisperConfig:
    possible_paths: List[str] = field(default_factory=lambda: [
        "~/whisper.cpp/build/bin/whisper-cli",
        "../whisper.cpp/build/bin/whisper-cli",
        "./whisper.cpp/build/bin/whisper-cli",
        "/usr/local/bin/whisper-cli"
    ])
    possible_models: List[str] = field(default_factory=lambda: [
        "~/whisper.cpp/models/ggml-tiny.en.bin",
        "./whisper.cpp/models/ggml-tiny.en.bin",
        "/usr/local/share/whisper/ggml-tiny.en.bin"
    ])
    timeout: int = 30

class VoiceAssistantError(Exception): pass
class AudioError(VoiceAssistantError): pass
class TranscriptionError(VoiceAssistantError): pass
class TTSError(VoiceAssistantError): pass

class VoiceAssistant:
    def __init__(self, audio_config: Optional[AudioConfig] = None,
                 porcupine_config: Optional[PorcupineConfig] = None,
                 whisper_config: Optional[WhisperConfig] = None):
        self.audio_config = audio_config or AudioConfig()
        self.porcupine_config = porcupine_config or PorcupineConfig()
        self.whisper_config = whisper_config or WhisperConfig()
        self.porcupine = None
        self.pa = None
        self.audio_stream = None
        self.is_running = False
        self.wake_word = "computer"
        self.whisper_path = None
        self.whisper_model = None
        self._setup_logging()
        self.audio_queue = Queue()
        try:
            self._setup_whisper()
            self._setup_porcupine()
            self._setup_audio()
        except Exception as e:
            self.cleanup()
            raise VoiceAssistantError(f"Initialization failed: {e}")

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger("VoiceAssistant")

    def _setup_whisper(self) -> None:
        for path in self.whisper_config.possible_paths:
            expanded_path = Path(path).expanduser()
            if expanded_path.exists():
                self.whisper_path = str(expanded_path)
                break
        if not self.whisper_path:
            raise VoiceAssistantError("Whisper executable not found")
        for model_path in self.whisper_config.possible_models:
            expanded_path = Path(model_path).expanduser()
            if expanded_path.exists():
                self.whisper_model = str(expanded_path)
                break
        if not self.whisper_model:
            raise VoiceAssistantError("Whisper model not found")

    def _setup_porcupine(self) -> None:
        # Use built-in 'computer' keyword
        self.porcupine = pvporcupine.create(
            access_key=self.porcupine_config.access_key,
            keywords=[self.porcupine_config.fallback_keyword],
            sensitivities=[self.porcupine_config.sensitivity]
        )
        self.wake_word = self.porcupine_config.fallback_keyword
        self.audio_config.sample_rate = self.porcupine.sample_rate
        self.audio_config.frame_length = self.porcupine.frame_length

    def _setup_audio(self) -> None:
        self.pa = pyaudio.PyAudio()
        self.audio_stream = self.pa.open(
            rate=self.audio_config.sample_rate,
            channels=self.audio_config.channels,
            format=self.audio_config.format,
            input=True,
            frames_per_buffer=self.audio_config.frame_length
        )

    @contextmanager
    def _audio_stream_context(self):
        try:
            if self.audio_stream and not self.audio_stream.is_active():
                self.audio_stream.start_stream()
            yield
        finally:
            pass

    def listen_for_hotword(self) -> None:
        self.logger.info(f"Listening for wake word '{self.wake_word}'...")
        self.is_running = True
        try:
            with self._audio_stream_context():
                while self.is_running:
                    pcm = self.audio_stream.read(
                        self.audio_config.frame_length,
                        exception_on_overflow=False
                    )
                    pcm = struct.unpack_from("h" * self.audio_config.frame_length, pcm)
                    keyword_index = self.porcupine.process(pcm)
                    if keyword_index >= 0:
                        print("✅ Wake word 'computer' detected! Listening for your question...")
                        self.handle_voice_command()
                        print("Listening for wake word 'computer'...")
                    time.sleep(0.1)
        except Exception as e:
            self.logger.error(f"Fatal error in hotword detection: {e}")
            raise

    def handle_voice_command(self) -> None:
        audio_file = "command.wav"
        try:
            print("🔴 Recording your question (5 seconds)...")
            if self._record_command(audio_file):
                print("🎙️ Question recorded. Transcribing...")
                transcription = self._transcribe_audio(audio_file)
                if transcription:
                    print(f"🗣️ You said: {transcription}")
                    response = self._get_llm_response(transcription)
                    if response:
                        print("🤖 Generating response...")
                        self._speak_response(response)
                    else:
                        self._speak_response("I'm sorry, I couldn't process that.")
                else:
                    self._speak_response("I didn't catch that. Could you repeat?")
            else:
                self._speak_response("Sorry, I couldn't record your question.")
        except Exception as e:
            self.logger.error(f"Error handling voice command: {e}")
            self._speak_response("I encountered an error processing your request.")
        finally:
            if Path(audio_file).exists():
                try:
                    Path(audio_file).unlink()
                except Exception:
                    pass

    def _record_command(self, filename: str) -> bool:
        cmd = [
            'arecord', '-D', 'default', '-f', 'S16_LE', '-r', '16000', '-c', '1',
            '-t', 'wav', '-d', str(self.audio_config.recording_duration), filename
        ]
        try:
            self.audio_stream.stop_stream()
        except Exception:
            pass
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.audio_config.recording_duration + 2
            )
            if self.audio_stream:
                self.audio_stream.start_stream()
            if result.returncode == 0 and Path(filename).exists() and Path(filename).stat().st_size > 1000:
                return True
        except Exception as e:
            self.logger.warning(f"Recording error: {e}")
        return False

    def _transcribe_audio(self, audio_file: str) -> Optional[str]:
        try:
            if not Path(audio_file).exists():
                return None
            cmd = [
                self.whisper_path,
                '-m', self.whisper_model,
                '-f', audio_file,
                '--output-txt',
                '--no-prints'
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.whisper_config.timeout
            )
            if result.returncode == 0:
                txt_file = Path(audio_file).with_suffix('.txt')
                if txt_file.exists():
                    transcription = txt_file.read_text().strip()
                    txt_file.unlink()
                    return transcription if transcription else None
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
        return None

    def _get_llm_response(self, question: str) -> Optional[str]:
        cmd = ['ollama', 'run', 'phi3:0.6', question]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                response = result.stdout.strip()
                return response if response else None
            else:
                self.logger.warning(f"Ollama error: {result.stderr}")
        except Exception as e:
            self.logger.warning(f"Ollama error: {e}")
        return None

    def _speak_response(self, text: str) -> None:
        try:
            print("🔊 Speaking response...")
            clean_text = text.replace('\n', ' ').strip()[:500]
            clean_text = clean_text.encode('ascii', errors='ignore').decode('ascii')
            cmd = ['espeak', '-s', '150', '-v', 'en', clean_text]
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
        except Exception as e:
            print(f"⚠️ TTS error: {e}")
            print(f"🤖 Assistant: {text}")

    def signal_handler(self, sig, frame) -> None:
        self.is_running = False
        self.cleanup()
        sys.exit(0)

    def cleanup(self) -> None:
        try:
            if self.audio_stream:
                if self.audio_stream.is_active():
                    self.audio_stream.stop_stream()
                self.audio_stream.close()
        except Exception:
            pass
        try:
            if self.pa:
                self.pa.terminate()
        except Exception:
            pass
        try:
            if self.porcupine:
                self.porcupine.delete()
        except Exception:
            pass

    def run(self) -> None:
        print("=" * 60)
        print("🤖 RASPBERRY PI OFFLINE VOICE ASSISTANT")
        print("=" * 60)
        print(f"Wake word: '{self.wake_word}'")
        print("Press Ctrl+C to exit")
        print("=" * 60)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        self.listen_for_hotword()

def main():
    print("🍓 Raspberry Pi Voice Assistant Setup")
    print("=" * 50)
    try:
        assistant = VoiceAssistant()
        assistant.run()
    except VoiceAssistantError as e:
        print(f"\n❌ Voice Assistant Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check audio devices: arecord -l")
        print("2. Test microphone: arecord -d 3 test.wav && aplay test.wav")
        print("3. Verify Ollama: ollama list")
        print("4. Check Whisper installation")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
