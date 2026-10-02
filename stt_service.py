import torch
import numpy as np
from scipy.io import wavfile
import io
import os


class WhisperSTT:
    def __init__(self, device="cpu"):
        """Инициализируем OpenAI Whisper локально"""
        self.device = device

        try:
            import whisper
            self.whisper = whisper
            self.model = whisper.load_model("base", device=device)
            print(f"✓ OpenAI Whisper загружена на {device}")
            self.available = True
        except ImportError:
            print("⚠️ whisper не установлена, установи: pip install openai-whisper")
            self.available = False
        except Exception as e:
            print(f"⚠️ Ошибка загрузки Whisper: {e}")
            self.available = False

    def transcribe(self, audio_data, sample_rate=16000):
        """
        Транскрибирует аудио в текст локально через Whisper

        Args:
            audio_data: аудио данные (bytes или numpy array)
            sample_rate: частота дискретизации

        Returns:
            str: распознанный текст
        """
        if not self.available:
            return None

        if not audio_data:
            return None

        try:
            # Конвертируем в numpy array если нужно
            if isinstance(audio_data, bytes):
                if audio_data[:4] == b'RIFF':
                    # Это WAV файл, сохраняем и загружаем через librosa
                    temp_file = "/tmp/audio_for_whisper.wav"
                    with open(temp_file, 'wb') as f:
                        f.write(audio_data)
                else:
                    # Raw PCM данные
                    audio_data = np.frombuffer(audio_data, dtype=np.int16)
                    temp_file = None
            else:
                temp_file = None

            # Нормализуем аудио
            if temp_file:
                result = self.model.transcribe(temp_file, language="ru", fp16=False)
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            else:
                # Нормализуем амплитуду
                audio_float = audio_data.astype(np.float32) / 32768.0

                # Resample если нужно
                if sample_rate != 16000:
                    import librosa
                    audio_float = librosa.resample(audio_float, orig_sr=sample_rate, target_sr=16000)

                result = self.model.transcribe(audio_float, language="ru", fp16=False)

            text = result.get("text", "").strip()
            return text if text else None

        except Exception as e:
            print(f"Ошибка при транскрибировании: {e}")
            return None

    def get_supported_formats(self):
        """Возвращает поддерживаемые форматы"""
        return {
            "formats": ["wav", "mp3", "m4a", "ogg", "flac"],
            "languages": ["ru", "en", "ja", "zh", "de", "fr", "es"],
            "sample_rates": [16000, 22050, 44100, 48000]
        }


# Инициализируем глобальный экземпляр
stt_model = None


def init_stt(device="cpu"):
    """Инициализирует STT модель"""
    global stt_model

    print(f"Инициализирую OpenAI Whisper STT (локально) на {device}...")
    stt_model = WhisperSTT(device=device)
    return stt_model


def speech_to_text(audio_data, sample_rate=16000):
    """Транскрибирует аудио в текст"""
    global stt_model

    if stt_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        init_stt(device=device)

    if stt_model is None or not stt_model.available:
        return "STT не инициализирована"

    return stt_model.transcribe(audio_data, sample_rate=sample_rate)


if __name__ == "__main__":
    # Тестирование
    init_stt()

    # Создаём тестовое аудио
    sample_rate = 16000
    duration = 1
    t = np.linspace(0, duration, int(sample_rate * duration))
    test_audio = np.sin(2 * np.pi * 440 * t) * 0.1
    test_audio = np.int16(test_audio * 32768)

    # Сохраняем в WAV
    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, sample_rate, test_audio)
    audio_bytes = wav_buffer.getvalue()

    result = speech_to_text(audio_bytes, sample_rate=sample_rate)

    if result:
        print(f"✓ Распознано: {result}")
    else:
        print("✗ Ошибка при распознавании")
