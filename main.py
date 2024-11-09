import streamlit as st
import pyedflib
import numpy as np
import tempfile
import os
from pykalman import KalmanFilter  # Для фильтра Калмана
from scipy.signal import butter, lfilter  # Для фильтра Баттерворта
import plotly.graph_objs as go

st.title("Анализ и визуализация ЭКоГ данных крыс WAG/Rij")

uploaded_file = st.file_uploader("Загрузите файл .edf", type="edf")
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    try:
        edf_reader = pyedflib.EdfReader(tmp_file_path)
        n_channels = edf_reader.signals_in_file
        signal_labels = edf_reader.getSignalLabels()

        st.write("**Информация о данных:**")
        st.write("Количество каналов:", n_channels)
        st.write("Названия каналов:", signal_labels)

        selected_channel = st.selectbox("Выберите канал для анализа", signal_labels)
        channel_index = signal_labels.index(selected_channel)

        filter_type = st.selectbox("Выберите тип фильтрации", ("Без фильтрации", "Фильтр Калмана", "Фильтр Баттерворта"))

        if filter_type == "Фильтр Баттерворта":
            lowcut = st.number_input("Нижняя частота среза (Гц)", min_value=0.1, value=0.5)
            highcut = st.number_input("Верхняя частота среза (Гц)", min_value=0.1, value=40.0)
            order = st.slider("Порядок фильтра", 1, 10, 5)

        data = edf_reader.readSignal(channel_index)
        sfreq = edf_reader.getSampleFrequency(channel_index)
        total_duration = len(data) / sfreq

        st.write("**Настройка временного окна:**")
        start_time = st.number_input("Начальное время (в секундах)", min_value=0.0, max_value=total_duration, value=0.0)
        duration = st.number_input("Длительность окна (в секундах)", min_value=0.1, max_value=60.0, value=10.0)
        end_time = start_time + duration

        if end_time > total_duration:
            st.warning("Конечное время превышает длительность записи. Корректируем конечное время.")
            end_time = total_duration
            duration = end_time - start_time

        start_idx = int(start_time * sfreq)
        end_idx = int(end_time * sfreq)

        signal_data = data[start_idx:end_idx]

        if filter_type == "Фильтр Калмана":
            kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
            signal_data_filtered, _ = kf.filter(signal_data)
            signal_data_filtered = signal_data_filtered.flatten()
        elif filter_type == "Фильтр Баттерворта":
            nyquist = 0.5 * sfreq
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype='band')
            signal_data_filtered = lfilter(b, a, signal_data)
        else:
            signal_data_filtered = signal_data

        time_array = np.linspace(start_time, end_time, len(signal_data))

        st.subheader(f"Сигнал канала: {selected_channel}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_array, y=signal_data, mode='lines', name='Исходный сигнал', line=dict(width=1, color='blue')))
        if filter_type != "Без фильтрации":
            fig.add_trace(go.Scatter(x=time_array, y=signal_data_filtered, mode='lines', name='Отфильтрованный сигнал', line=dict(width=1, color='red')))
        fig.update_layout(
            xaxis_title="Время (сек)",
            yaxis_title="Амплитуда",
            legend=dict(x=0, y=1),
            hovermode="x unified",
            height=400,
            xaxis=dict(rangeslider=dict(visible=True), type="linear")
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Частотный спектр сигнала")
        freqs = np.fft.rfftfreq(len(signal_data_filtered), 1/sfreq)
        psd = np.abs(np.fft.rfft(signal_data_filtered))**2

        fig_psd = go.Figure()
        fig_psd.add_trace(go.Scatter(x=freqs, y=10 * np.log10(psd), mode='lines', name='Плотность спектральной мощности'))
        fig_psd.update_layout(
            xaxis_title="Частота (Гц)",
            yaxis_title="Мощность (дБ)",
            hovermode="x unified",
            height=400,
            xaxis=dict(type="linear")
        )
        st.plotly_chart(fig_psd, use_container_width=True)

    finally:
        edf_reader.close()
        os.remove(tmp_file_path)
