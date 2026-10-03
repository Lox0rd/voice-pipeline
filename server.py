import os
import io
import time
from flask import Flask, request, send_file, jsonify
import openai
from tts_service import text_to_speech, init_tts
from stt_service import speech_to_text, init_stt

app = Flask(__name__)

COMET_API_KEY = os.getenv("COMETAPI_KEY")
COMET_API_URL = "https://api.cometapi.com/v1"

# Конфигурируем OpenAI для работы с CometAPI
openai.api_base = COMET_API_URL
openai.api_key = COMET_API_KEY

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024

conversation_history = []
system_prompt = "Ты полезный голосовой ассистент. Отвечай кратко и по существу, чтобы ответ можно было озвучить."

# Инициализируем TTS и STT при запуске сервера
print("Инициализация TTS и STT сервисов...")
device = "cuda" if os.getenv("USE_GPU") else "cpu"
init_tts(device=device)
init_stt(device=device)


@app.route('/api/voice', methods=['POST'])
def handle_voice():
    """Основной эндпоинт для обработки голоса"""
    try:
        # Получаем аудио данные
        audio_data = request.data

        if not audio_data:
            return jsonify({"error": "Аудио данные не получены"}), 400

        # Распознаём речь
        user_text = speech_to_text(audio_data)
        if not user_text:
            return jsonify({"error": "Не удалось распознать речь"}), 400

        print(f"Пользователь: {user_text}")

        # Добавляем сообщение в историю
        conversation_history.append({
            "role": "user",
            "content": user_text
        })

        # Получаем ответ от CometAPI через OpenAI SDK с повторными попытками
        max_retries = 3
        retry_delay = 2
        assistant_text = None

        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="qwen3.8-omni-flash",
                    max_tokens=1024,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *conversation_history
                    ]
                )
                assistant_text = response.choices[0].message.content
                break
            except Exception as e:
                error_msg = str(e)
                print(f"Попытка {attempt + 1}/{max_retries}. Ошибка CometAPI: {error_msg}")

                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Ожидание {wait_time} сек перед повторной попыткой...")
                    time.sleep(wait_time)
                else:
                    print(f"Все попытки исчерпаны")
                    return jsonify({"error": f"Ошибка при обращении к ИИ: {error_msg}"}), 500

        if not assistant_text:
            return jsonify({"error": "Не удалось получить ответ от ИИ"}), 500

        print(f"Ассистент: {assistant_text}")

        # Добавляем ответ в историю
        conversation_history.append({
            "role": "assistant",
            "content": assistant_text
        })

        # Преобразуем ответ в речь
        response_audio = text_to_speech(assistant_text)

        # Возвращаем аудио
        return send_file(
            io.BytesIO(response_audio),
            mimetype="audio/wav",
            as_attachment=True,
            download_name="response.wav"
        )

    except Exception as e:
        print(f"Ошибка сервера: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/text', methods=['POST'])
def handle_text():
    """Альтернативный эндпоинт для текстовых сообщений"""
    try:
        data = request.get_json()
        user_text = data.get('message', '')

        if not user_text:
            return jsonify({"error": "Сообщение не получено"}), 400

        print(f"Пользователь: {user_text}")

        conversation_history.append({
            "role": "user",
            "content": user_text
        })

        # Получаем ответ от CometAPI через OpenAI SDK с повторными попытками
        max_retries = 3
        retry_delay = 2
        assistant_text = None

        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="qwen3.8-omni-flash",
                    max_tokens=1024,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *conversation_history
                    ]
                )
                assistant_text = response.choices[0].message.content
                break
            except Exception as e:
                error_msg = str(e)
                print(f"Попытка {attempt + 1}/{max_retries}. Ошибка CometAPI: {error_msg}")

                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Ожидание {wait_time} сек перед повторной попыткой...")
                    time.sleep(wait_time)
                else:
                    print(f"Все попытки исчерпаны")
                    return jsonify({"error": f"Ошибка при обращении к ИИ: {error_msg}"}), 500

        if not assistant_text:
            return jsonify({"error": "Не удалось получить ответ от ИИ"}), 500

        print(f"Ассистент: {assistant_text}")

        conversation_history.append({
            "role": "assistant",
            "content": assistant_text
        })

        return jsonify({
            "response": assistant_text,
            "user_message": user_text
        })

    except Exception as e:
        print(f"Ошибка сервера: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/system-prompt', methods=['POST'])
def update_system_prompt():
    """Изменяет системный промпт для LLM"""
    global system_prompt

    try:
        data = request.get_json()

        if not data or 'prompt' not in data:
            return jsonify({
                "error": "Необходимо передать поле 'prompt'"
            }), 400

        new_prompt = data['prompt']

        if not isinstance(new_prompt, str):
            return jsonify({
                "error": "Поле 'prompt' должно быть строкой"
            }), 400

        if not new_prompt.strip():
            return jsonify({
                "error": "Системный промпт не может быть пустым"
            }), 400

        system_prompt = new_prompt.strip()

        print(f"Системный промпт изменён: {system_prompt}")

        return jsonify({
            "message": "Системный промпт успешно изменён",
            "system_prompt": system_prompt
        })

    except Exception as e:
        print(f"Ошибка изменения системного промпта: {e}")
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/api/reset', methods=['POST'])
def reset_conversation():
    """Очищаем историю разговора"""
    global conversation_history
    conversation_history = []
    return jsonify({"message": "История очищена"})


@app.route('/health', methods=['GET'])
def health():
    """Проверка здоровья сервера"""
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
