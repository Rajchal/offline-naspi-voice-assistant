

import os
import sys
import struct
import subprocess
import time
import signal
import threading
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
from queue import Queue, Empty


try:
    import pyaudio
    import pvporcupine
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Install with: pip3 install pyaudio pvporcupine")
    sys.exit(1)




@dataclass
class AudioConfig:
    """Audio configuration settings"""
    sample_rate: int = 16000
    frame_length: int = 512
    channels: int = 1
    format: int = pyaudio.paInt16
    recording_duration: int = 5
    audio_timeout: int = 30




@dataclass
class PorcupineConfig:
    """Porcupine hotword detection configuration"""
    access_key: str = "CnNEQfm996S877kY+Ml+GSSqdOb/IgW5CKVUSXzasBWK8+SRlwfeDg=="
    keyword_file: str = "Hey-Raspberry-Pi_en_raspberry-pi_v3_0_0.ppn"
    fallback_keyword: str = "computer"
    sensitivity: float = 0.5




@dataclass
class WhisperConfig:
    """Whisper transcription configuration"""
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




class VoiceAssistantError(Exception):
    """Base exception for voice assistant errors"""
    pass




class AudioError(VoiceAssistantError):
    """Audio-related errors"""
    pass




class TranscriptionError(VoiceAssistantError):
    """Transcription-related errors"""
    pass




class TTSError(VoiceAssistantError):
    """Text-to-speech errors"""
    pass




class VoiceAssistant:
    """Professional Raspberry Pi Voice Assistant with proper error handling"""
    
    def __init__(self, audio_config: Optional[AudioConfig] = None,
                 porcupine_config: Optional[PorcupineConfig] = None,
                 whisper_config: Optional[WhisperConfig] = None):
        self.audio_config = audio_config or AudioConfig()
        self.porcupine_config = porcupine_config or PorcupineConfig()
        self.whisper_config = whisper_config or WhisperConfig()
        
        # Initialize components
        self.porcupine = None
        self.pa = None
        self.audio_stream = None
        self.is_running = False
        self.wake_word = ""
        self.whisper_path = None
        self.whisper_model = None
        
        # Setup logging
        self._setup_logging()
        
        # Audio processing queue for thread safety
        self.audio_queue = Queue()
        
        self.logger.info("Initializing Voice Assistant for Raspberry Pi...")
        
        try:
            self._validate_environment()
            self._setup_whisper()
            self._setup_porcupine()
            self._setup_audio()
            self.logger.info("Voice Assistant initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Voice Assistant: {e}")
            self.cleanup()
            raise VoiceAssistantError(f"Initialization failed: {e}")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('voice_assistant.log', mode='a')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _validate_environment(self) -> None:
        """Validate the environment and dependencies"""
        self.logger.info("Validating environment...")
        
        # Check if running on Raspberry Pi
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'Raspberry Pi' in cpuinfo:
                    self.logger.info("✓ Running on Raspberry Pi")
                else:
                    self.logger.warning("⚠ Not detected as Raspberry Pi")
        except FileNotFoundError:
            self.logger.warning("⚠ Could not detect system type")
        
        # Check required commands
        required_commands = ['arecord', 'espeak', 'ollama']
        missing_commands = []
        
        for cmd in required_commands:
            if not self._command_exists(cmd):
                missing_commands.append(cmd)
        
        if missing_commands:
            raise VoiceAssistantError(f"Missing required commands: {missing_commands}")
        
        # Check Ollama model
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
            if 'phi' not in result.stdout:
                self.logger.warning("⚠ Phi model not found. Run: ollama pull phi")
        except subprocess.TimeoutExpired:
            raise VoiceAssistantError("Ollama service timeout")
        except Exception as e:
            raise VoiceAssistantError(f"Ollama not available: {e}")
    
    def _command_exists(self, command: str) -> bool:
        """Check if a command exists in the system"""
        try:
            subprocess.run([command, '--help'], capture_output=True, timeout=5)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _setup_whisper(self) -> None:
        """Setup Whisper paths and validate installation"""
        self.logger.info("Setting up Whisper...")
        
        # Find Whisper executable
        for path in self.whisper_config.possible_paths:
            expanded_path = Path(path).expanduser()
            if expanded_path.exists():
                self.whisper_path = str(expanded_path)
                break
        
        if not self.whisper_path:
            raise VoiceAssistantError("Whisper executable not found")
        
        # Find Whisper model
        for model_path in self.whisper_config.possible_models:
            expanded_path = Path(model_path).expanduser()
            if expanded_path.exists():
                self.whisper_model = str(expanded_path)
                break
        
        if not self.whisper_model:
            raise VoiceAssistantError("Whisper model not found")
        
        self.logger.info(f"✓ Whisper executable: {self.whisper_path}")
        self.logger.info(f"✓ Whisper model: {self.whisper_model}")
    
    def _setup_porcupine(self) -> None:
        """Initialize Porcupine hotword detection with error handling"""
        self.logger.info("Setting up Porcupine hotword detection...")
        
        try:
            keyword_path = Path(self.porcupine_config.keyword_file)
            
            if keyword_path.exists():
                self.porcupine = pvporcupine.create(
                    access_key=self.porcupine_config.access_key,
                    keyword_paths=[str(keyword_path)],
                    sensitivities=[self.porcupine_config.sensitivity]
                )
                self.wake_word = "Hey Raspberry Pi"
                self.logger.info(f"✓ Using custom wake word: '{self.wake_word}'")
            else:
                self.porcupine = pvporcupine.create(
                    access_key=self.porcupine_config.access_key,
                    keywords=[self.porcupine_config.fallback_keyword],
                    sensitivities=[self.porcupine_config.sensitivity]
                )
                self.wake_word = self.porcupine_config.fallback_keyword
                self.logger.warning(f"⚠ Using fallback wake word: '{self.wake_word}'")
            
            # Update audio config based on Porcupine requirements
            self.audio_config.sample_rate = self.porcupine.sample_rate
            self.audio_config.frame_length = self.porcupine.frame_length
            
        except Exception as e:
            raise VoiceAssistantError(f"Failed to initialize Porcupine: {e}")
    
    def _setup_audio(self) -> None:
        """Initialize PyAudio with comprehensive error handling"""
        self.logger.info("Setting up audio...")
        
        try:
            self.pa = pyaudio.PyAudio()
            device_index = self._find_best_audio_device()
            
            self.audio_stream = self.pa.open(
                rate=self.audio_config.sample_rate,
                channels=self.audio_config.channels,
                format=self.audio_config.format,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.audio_config.frame_length,
                stream_callback=None
            )
            
            self.logger.info("✓ Audio stream initialized successfully")
            
        except Exception as e:
            raise AudioError(f"Failed to initialize audio: {e}")
    
    def _find_best_audio_device(self) -> Optional[int]:
        """Find and return the best available audio input device"""
        self.logger.info("Scanning audio devices...")
        
        devices = []
        for i in range(self.pa.get_device_count()):
            try:
                info = self.pa.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    devices.append((i, info))
                    self.logger.info(f"  Device {i}: {info['name']} "
                                   f"(inputs: {info['maxInputChannels']})")
            except Exception as e:
                self.logger.warning(f"Error getting device {i} info: {e}")
        
        if not devices:
            raise AudioError("No audio input devices found")
        
        # Return default device (None) or first available device
        return None
    
    @contextmanager
    def _audio_stream_context(self):
        """Context manager for audio stream operations"""
        try:
            if self.audio_stream and not self.audio_stream.is_active():
                self.audio_stream.start_stream()
            yield
        finally:
            if self.audio_stream and self.audio_stream.is_active():
                pass  # Keep stream running for continuous listening
    
    def listen_for_hotword(self) -> None:
        """Listen continuously for the hotword with improved error handling"""
        self.logger.info(f" Listening for wake word '{self.wake_word}'...")
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        try:
            with self._audio_stream_context():
                while self.is_running:
                    try:
                        pcm = self.audio_stream.read(
                            self.audio_config.frame_length,
                            exception_on_overflow=False
                        )
                        pcm = struct.unpack_from("h" * self.audio_config.frame_length, pcm)
                        
                        keyword_index = self.porcupine.process(pcm)
                        
                        if keyword_index >= 0:
                            self.logger.info(f" Wake word '{self.wake_word}' detected!")
                            print(" Wake word 'Hey Raspberry Pi' detected! Listening for your question...")
                            self.handle_voice_command()
                            self.logger.info(f" Resuming listening for '{self.wake_word}'...")
                            print(" Resuming listening for 'Hey Raspberry Pi'...")
                        
                        consecutive_errors = 0  # Reset error counter on success
                        
                    except Exception as e:
                        consecutive_errors += 1
                        self.logger.error(f"Error in hotword detection: {e}")
                        
                        if consecutive_errors >= max_consecutive_errors:
                            raise VoiceAssistantError(f"Too many consecutive errors: {e}")
                        
                        time.sleep(0.1)  # Brief pause before retry
                        
        except Exception as e:
            self.logger.error(f"Fatal error in hotword detection: {e}")
            raise
    
    def save_answer(self, question: str, answer: str) -> None:
        """Save the question and LLM answer to a file"""
        try:
            with open("llm_responses.txt", "a") as file:
                file.write(f"Question: {question}\nAnswer: {answer}\n\n")
            self.logger.info("✓ Question and answer saved to llm_responses.txt")
        except Exception as e:
            self.logger.error(f"Error saving question and answer: {e}")
    
    def handle_voice_command(self) -> None:
        """Handle voice command with comprehensive error handling"""
        audio_file = None
        try:
            audio_file = "command.wav"
            self.logger.info("Recording your question...")
            print(" Recording your question (5 seconds)...")
            
            if self._record_command(audio_file):
                print(" Question recorded. Transcribing...")
                transcription = self._transcribe_audio(audio_file)
                
                if transcription:
                    self.logger.info(f"Transcription: {transcription}")
                    print(f" You said: {transcription}")
                    response = self._get_llm_response(transcription)
                    
                    if response:
                        self.logger.info(f"Response: {response}")
                        print(" Generating response...")
                        self.save_answer(transcription, response)
                        self._speak_response(response)
                    else:
                        self._speak_response("I'm sorry, I couldn't process that.")
                else:
                    self.logger.error("Transcription failed, no valid transcription returned")
                    self._speak_response("I didn't catch that. Could you repeat?")
            else:
                self._speak_response("Sorry, I couldn't record your question.")
                
        except Exception as e:
            self.logger.error(f"Error handling voice command: {e}")
            self._speak_response("I encountered an error processing your request.")
        finally:
            # Cleanup
            if audio_file and Path(audio_file).exists():
                try:
                    Path(audio_file).unlink()
                except Exception as e:
                    self.logger.warning(f"Could not delete {audio_file}: {e}")
    
    def _record_command(self, filename: str) -> bool:
        """Record audio command with multiple fallback methods"""
        recording_commands = [
            ['arecord', '-D', 'default', '-f', 'S16_LE', '-r', '16000', '-c', '1', 
             '-t', 'wav', '-d', str(self.audio_config.recording_duration), filename],
            ['arecord', '-D', 'plughw:1,0', '-f', 'S16_LE', '-r', '16000', '-c', '1', 
             '-t', 'wav', '-d', str(self.audio_config.recording_duration), filename],
            ['arecord', '-D', 'hw:1,0', '-f', 'S16_LE', '-r', '16000', '-c', '1', 
             '-t', 'wav', '-d', str(self.audio_config.recording_duration), filename]
        ]
        
        for i, cmd in enumerate(recording_commands, 1):
            try:
                self.logger.info(f"Recording attempt {i}/{len(recording_commands)}")
                
                # Temporarily pause the main audio stream to avoid conflicts
                if self.audio_stream:
                    self.audio_stream.stop_stream()
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=self.audio_config.recording_duration + 2
                )
                
                # Resume the main audio stream
                if self.audio_stream:
                    self.audio_stream.start_stream()
                
                if (result.returncode == 0 and 
                    Path(filename).exists() and 
                    Path(filename).stat().st_size > 1000):  # Minimum file size check
                    
                    self.logger.info("✓ Recording successful")
                    return True
                else:
                    self.logger.warning(f"Recording failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                self.logger.warning("Recording timed out")
                if self.audio_stream:
                    self.audio_stream.start_stream()
            except Exception as e:
                self.logger.warning(f"Recording error: {e}")
                if self.audio_stream:
                    self.audio_stream.start_stream()
        
        return False
    
    def _transcribe_audio(self, audio_file: str) -> Optional[str]:
        """Transcribe audio using Whisper with enhanced error handling"""
        try:
            if not Path(audio_file).exists():
                self.logger.error(f"Transcription error: Audio file not found: {audio_file}")
                raise TranscriptionError(f"Audio file not found: {audio_file}")
            
            cmd = [
                self.whisper_path,
                '-m', self.whisper_model,
                '-f', audio_file,
                '--output-txt',
                '--no-prints'
            ]
            
            self.logger.info(f"Executing Whisper command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=self.whisper_config.timeout
            )
            
            self.logger.info(f"Whisper stdout: {result.stdout}")
            if result.stderr:
                self.logger.error(f"Whisper stderr: {result.stderr}")
            
            if result.returncode == 0:
                txt_file = Path(audio_file).with_suffix('.txt')
                self.logger.info(f"Checking for transcription file: {txt_file}")
                if txt_file.exists():
                    try:
                        transcription = txt_file.read_text().strip()
                        self.logger.info(f"Transcription result: '{transcription}'")
                        if not transcription:
                            self.logger.error("Transcription error: Empty transcription returned")
                        txt_file.unlink()  # Cleanup
                        return transcription if transcription else None
                    except Exception as e:
                        self.logger.error(f"Transcription error: Failed to read transcription file {txt_file}: {e}")
                        raise TranscriptionError(f"Failed to read transcription file: {e}")
                else:
                    self.logger.error(f"Transcription error: Transcription file not found: {txt_file}")
                    raise TranscriptionError(f"Transcription file not found: {txt_file}")
            else:
                self.logger.error(f"Transcription error: Whisper command failed with exit code {result.returncode}")
                raise TranscriptionError(f"Whisper command failed with exit code {result.returncode}")
            
        except subprocess.TimeoutExpired:
            self.logger.error("Transcription error: Whisper command timed out")
            raise TranscriptionError("Whisper command timed out")
        except TranscriptionError as e:
            self.logger.error(f"Transcription error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Transcription error: Unexpected error during transcription: {e}")
            raise TranscriptionError(f"Unexpected error during transcription: {e}")
    
    def _get_llm_response(self, question: str) -> Optional[str]:
        """Get response from Ollama with error handling and retry logic"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                cmd = ['ollama', 'run', 'phi', question]
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
                    self.logger.warning(f"Ollama error (attempt {attempt + 1}): {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                self.logger.warning(f"Ollama timeout (attempt {attempt + 1})")
            except Exception as e:
                self.logger.error(f"LLM error (attempt {attempt + 1}): {e}")
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
        
        return None
    
    def _speak_response(self, text: str) -> None:
        """Text-to-speech with enhanced error handling and user feedback"""
        try:
            print("🔊 Speaking response...")
            self.logger.info("Attempting to speak response")
            
            # Pause microphone to avoid feedback
            if self.audio_stream and self.audio_stream.is_active():
                self.audio_stream.stop_stream()
            
            # Clean and limit text, handle Unicode
            clean_text = text.replace('\n', ' ').strip()[:500]
            clean_text = clean_text.encode('ascii', errors='ignore').decode('ascii')
            
            # Try espeak first
            cmd = ['espeak', '-s', '150', '-v', 'en', clean_text]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                self.logger.info("✓ espeak TTS successful")
                print(" Response spoken successfully")
            else:
                self.logger.warning(f"espeak failed: {result.stderr}")
                # Try espeak-ng as fallback
                cmd = ['espeak-ng', '-s', '150', '-v', 'en', clean_text]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    self.logger.info("✓ espeak-ng TTS successful")
                    print(" Response spoken successfully using espeak-ng")
                else:
                    self.logger.error(f"espeak-ng failed: {result.stderr}")
                    raise TTSError("Both espeak and espeak-ng failed")
            
        except subprocess.TimeoutExpired:
            self.logger.error("TTS timed out")
            print(" TTS timed out")
            self._fallback_text_output(text)
        except FileNotFoundError:
            self.logger.error("espeak or espeak-ng not found")
            print(" espeak/espeak-ng not installed. Install with: sudo apt-get install espeak espeak-ng")
            self._fallback_text_output(text)
        except Exception as e:
            self.logger.error(f"TTS error: {e}")
            print(f" TTS error: {e}")
            self._fallback_text_output(text)
        finally:
            # Resume microphone
            try:
                if self.audio_stream and not self.audio_stream.is_active():
                    self.audio_stream.start_stream()
                self.logger.info("Audio stream resumed")
            except Exception as e:
                self.logger.error(f"Error resuming audio stream: {e}")
                self._fallback_text_output(text)
    
    def _fallback_text_output(self, text: str) -> None:
        """Fallback text output when TTS fails"""
        print(f" Assistant: {text}")
    
    def signal_handler(self, sig, frame) -> None:
        """Handle shutdown signals gracefully"""
        self.logger.info("Shutdown signal received...")
        self.is_running = False
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self) -> None:
        """Comprehensive cleanup of resources"""
        self.logger.info("Cleaning up resources...")
        
        try:
            if self.audio_stream:
                if self.audio_stream.is_active():
                    self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None
        except Exception as e:
            self.logger.error(f"Error closing audio stream: {e}")
        
        try:
            if self.pa:
                self.pa.terminate()
                self.pa = None
        except Exception as e:
            self.logger.error(f"Error terminating PyAudio: {e}")
        
        try:
            if self.porcupine:
                self.porcupine.delete()
                self.porcupine = None
        except Exception as e:
            self.logger.error(f"Error deleting Porcupine: {e}")
        
        # Clean up temporary files
        temp_files = ['command.wav', 'command.txt']
        for temp_file in temp_files:
            try:
                if Path(temp_file).exists():
                    Path(temp_file).unlink()
            except Exception as e:
                self.logger.warning(f"Could not delete {temp_file}: {e}")
    
    def run(self) -> None:
        """Main execution loop with proper error handling"""
        print("=" * 60)
        print(" RASPBERRY PI OFFLINE VOICE ASSISTANT")
        print("=" * 60)
        print(f"Wake word: '{self.wake_word}'")
        print("Press Ctrl+C to exit")
        print("=" * 60)
        
        # Set up signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.is_running = True
        
        try:
            self.listen_for_hotword()
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"Fatal error in main loop: {e}")
        finally:
            self.cleanup()




def main():
    """Main function with comprehensive environment setup and validation"""
    print(" Raspberry Pi Voice Assistant Setup")
    print("=" * 50)
    
    try:
        # Initialize and run the voice assistant
        assistant = VoiceAssistant()
        assistant.run()
        
    except VoiceAssistantError as e:
        print(f"\n Voice Assistant Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check audio devices: arecord -l")
        print("2. Test microphone: arecord -d 3 test.wav && aplay test.wav")
        print("3. Verify Ollama: ollama list")
        print("4. Check Whisper installation")
        return 1
        
    except KeyboardInterrupt:
        print("\n Goodbye!")
        return 0
        
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        print("Check voice_assistant.log for detailed error information")
        return 1
    
    return 0




if __name__ == "__main__":
    sys.exit(main())

