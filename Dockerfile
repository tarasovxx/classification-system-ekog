# Используем базовый образ с Python
FROM python:3.11-slim

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt, если у вас есть, или устанавливаем библиотеки напрямую
COPY requirements.txt requirements.txt

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Если requirements.txt нет, можно установить зависимости напрямую:
# RUN pip install streamlit pyedflib matplotlib

# Копируем файлы приложения в контейнер
COPY . .

# Указываем переменную среды для запуска Streamlit в контейнере
#ENV STREAMLIT_PORT=8501
#EXPOSE $STREAMLIT_PORT

# Запускаем Streamlit приложение
CMD ["streamlit", "run", "main.py"]
