# Voice Assistant Server

Полнофункциональный голосовой ассистент на Python.

**Архитектура:**
- **STT (Speech-to-Text)** — OpenAI Whisper (локально, офлайн)
- **LLM (Language Model)** — CometAPI (Qwen3.8-Omni-Flash)
- **TTS (Text-to-Speech)** — Google Text-to-Speech (gTTS) (натуральная речь)

## Требования

- Python 3.8+
- pip
- FFmpeg (для обработки аудио)

## Установка

### 1. Клонируй репозиторий

```bash
git clone https://github.com/Lox0rd/voice-pipeline
cd voice-pipeline
```

### 2. Создай виртуальное окружение

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# или
venv\Scripts\activate  # Windows
```

### 3. Установи зависимости

```bash
pip install -r requirements.txt
```

### 4. Установи FFmpeg

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Скачай с https://ffmpeg.org/download.html или через `choco install ffmpeg`

### 5. Получи API ключ

1. Перейди на https://www.cometapi.com/console/token
2. Скопируй свой API ключ

### 6. Настрой переменные окружения

Создай файл `.env` в корне проекта:

```env
COMETAPI_KEY=your_api_key_here
```

Или установи через переменные окружения:

```bash
export COMETAPI_KEY=your_api_key_here
```

## Использование

### Запуск сервера

```bash
python server.py
```

Сервер запустится на `http://127.0.0.1:5000`

### API Endpoints

#### 1. Голосовой ввод (STT → LLM → TTS)

**POST** `/api/voice`

Отправляет WAV аудио, получает озвученный ответ.

**Параметры:**
- `audio` (bytes) — WAV аудио файл

**Ответ:**
```json
{
  "user_text": "Привет, как дела?",
  "assistant_text": "Привет! У меня всё хорошо. Чем я могу помочь?",
  "audio_file": "/tmp/tts_output.wav"
}
```

**Пример (curl):**
```bash
curl -X POST http://127.0.0.1:5000/api/voice \
  -F "audio=@audio.wav"
```

#### 2. Текстовый ввод (LLM → TTS)

**POST** `/api/text`

Отправляет текст, получает озвученный ответ ИИ.

**Параметры:**
```json
{
  "message": "Что такое радуга?"
}
```

**Ответ:**
```json
{
  "message": "Радуга - это...",
  "audio_file": "/tmp/tts_output.wav"
}
```

**Пример (curl):**
```bash
curl -X POST http://127.0.0.1:5000/api/text \
  -H "Content-Type: application/json" \
  -d '{"message": "Привет"}'
```

## Структура проекта

```
Ver/
├── server.py           # Flask сервер
├── stt_service.py      # Speech-to-Text (Whisper)
├── tts_service.py      # Text-to-Speech (pyttsx3)
├── requirements.txt    # Зависимости
├── .env               # Конфиг с API ключами
├── .gitignore         # Git игнорирование
└── README.md          # Этот файл
```

## Поддерживаемые языки STT

Whisper поддерживает 99+ языков, включая:
- 🇷🇺 Русский
- 🇬🇧 Английский
- 🇯🇵 Японский
- 🇨🇳 Китайский
- 🇩🇪 Немецкий
- 🇫🇷 Французский
- 🇪🇸 Испанский

## Производительность

| Операция | Время | Место |
|----------|-------|-------|
| STT (30сек) | 5-10сек | Локально (CPU) |
| LLM (200 токенов) | 3-5сек | CometAPI |
| TTS (500 символов) | 6-11сек | gTTS |
| **Общая задержка** | **10-18сек** | - |

С GPU (CUDA) STT работает в 5-10 раз быстрее.

## Решение проблем

### Whisper не загружается
```bash
pip install --upgrade openai-whisper
```

### FFmpeg ошибка
```bash
# Linux
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

### COMETAPI_KEY не установлена
Проверь файл `.env`:
```bash
cat .env
```

Должно быть:
```
COMETAPI_KEY=sk-...
```

### Медленное распознавание
Установи CUDA для GPU ускорения:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Лицензия

Apache 2.0 (Kokoro TTS)
MIT (остальной код)
