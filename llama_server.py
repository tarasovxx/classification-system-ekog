from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

OLLAMA_BASE_URL = "http://localhost:11434/api"


@app.route('/generate_completion', methods=['POST'])
def generate_completion():
    try:
        data = request.json
        model = data.get('model')
        prompt = data.get('prompt')
        stream = data.get('stream', True)

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }

        response = requests.post(f"{OLLAMA_BASE_URL}/generate", json=payload, stream=stream)

        if stream:
            response_text = ""
            for line in response.iter_lines():
                if line:
                    response_text += line.decode('utf-8')
            return jsonify({"response": response_text})
        else:
            return jsonify(response.json())

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate_chat_completion', methods=['POST'])
def generate_chat_completion():
    try:
        data = request.json
        model = data.get('model')
        messages = data.get('messages')
        stream = data.get('stream', True)

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }

        response = requests.post(f"{OLLAMA_BASE_URL}/chat", json=payload, stream=stream)

        if stream:
            response_text = ""
            for line in response.iter_lines():
                if line:
                    response_text += line.decode('utf-8')
            return jsonify({"response": response_text})
        else:
            return jsonify(response.json())

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate_embeddings', methods=['POST'])
def generate_embeddings():
    try:
        data = request.json
        model = data.get('model')
        input_text = data.get('input')

        payload = {
            "model": model,
            "input": input_text
        }

        response = requests.post(f"{OLLAMA_BASE_URL}/embed", json=payload)
        return jsonify(response.json())

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
