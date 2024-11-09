FROM python:3.11-slim

WORKDIR /app

COPY requirements_ollama.txt .
RUN pip install --no-cache-dir -r requirements_ollama.txt

COPY . .

EXPOSE 5000
CMD ["python", "llama_server.py"]
