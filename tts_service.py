import torch
import os
import io

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False


class GoogleTTS:
    def __init__(self, device="cpu"):
        """Инициализируем Google Text-to-Speech"""
        self.device = device

        if GTTS_AVAILABLE:
            print(f"✓ Google Text-to-Speech (gTTS) загружена")
        else:
            print("⚠️ gTTS не установлена, установи: pip install gtts")

    def synthesize(self, text, speaker="default", speed=1.0):
        """
        Синтезирует речь из текста через Google TTS

        Args:
            text: текст для синтеза
            speaker: голос (не используется)
            speed: скорость речи (не используется)

        Returns:
            bytes: MP3 аудио файл
        """
        if not text:
            return None

        if not GTTS_AVAILABLE:
            return None

        try:
            # Используем Google TTS для русского языка
            tts = gTTS(text=text, lang='ru', slow=False, tld='com')

            # Сохраняем в буфер
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)

            return audio_buffer.getvalue()

        except Exception as e:
            print(f"Ошибка gTTS: {e}")
            return None

    def get_speakers(self):
        """Возвращает список доступных голосов"""
        return {
            "default": "Google голос (натуральная речь)"
        }




# Инициализируем глобальный экземпляр
tts_model = None


def init_tts(device="cpu"):
    """Инициализирует TTS модель"""
    global tts_model
    print(f"Инициализирую TTS (Google gTTS) на {device}...")
    tts_model = GoogleTTS(device=device)
    return tts_model


def text_to_speech(text, speaker="default", speed=1.0):
    """Синтезирует речь из текста"""
    global tts_model

    if tts_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        init_tts(device=device)

    return tts_model.synthesize(text, speaker=speaker, speed=speed)


if __name__ == "__main__":
    # Тестирование
    init_tts()

    test_text = "Привет, это тестовое сообщение на русском языке"
    audio = text_to_speech(test_text, speaker="default")

    if audio:
        with open("test_output.mp3", "wb") as f:
            f.write(audio)
        print("✓ Файл test_output.mp3 создан")
    else:
        print("✗ Ошибка синтеза")
