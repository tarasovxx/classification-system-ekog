FROM python:3.11-slim

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV STREAMLIT_PORT=8501
EXPOSE $STREAMLIT_PORT

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
