import streamlit as st
import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os

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

        st.write("Информация о данных:")
        st.write("Количество каналов:", n_channels)
        st.write("Названия каналов:", signal_labels)

        start_time = st.slider("Выберите начальное время (в секундах)", 0, int(edf_reader.file_duration), 0)
        duration = st.slider("Выберите длительность окна (в секундах)", 1, 60, 10)

        st.subheader("ЭКоГ данные каналов:")
        for i, label in enumerate(signal_labels):
            data = edf_reader.readSignal(i)
            sfreq = edf_reader.getSampleFrequency(i)
            end_time = start_time + duration
            start_idx = int(start_time * sfreq)
            end_idx = int(end_time * sfreq)

            # Ensure indices are within bounds and valid
            if start_idx >= 0 and end_idx <= len(data) and end_idx > start_idx:
                fig, ax = plt.subplots()
                time_array = np.linspace(start_time, end_time, end_idx - start_idx)
                ax.plot(time_array, data[start_idx:end_idx])
                ax.set_title(f"Канал {label}")
                ax.set_xlabel("Время (сек)")
                ax.set_ylabel("Амплитуда")
                st.pyplot(fig)
            else:
                st.write(f"Некорректные индексы для канала {label}. Проверьте начальное время и длительность окна.")

        st.subheader("Частотный спектр данных:")
        for i, label in enumerate(signal_labels):
            data = edf_reader.readSignal(i)
            psds, freqs = plt.psd(data[start_idx:end_idx], NFFT=1024, Fs=sfreq)

            fig, ax = plt.subplots()
            ax.plot(freqs, 10 * np.log10(psds))
            ax.set_title(f"Частотный спектр канала {label}")
            ax.set_xlabel("Частота (Гц)")
            ax.set_ylabel("Мощность (дБ)")
            st.pyplot(fig)

    finally:
        edf_reader.close()
        os.remove(tmp_file_path)
