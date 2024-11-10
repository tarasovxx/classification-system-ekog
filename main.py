import streamlit as st
import pyedflib
import numpy as np
import pandas as pd
import tempfile
import os
from pykalman import KalmanFilter
from scipy.signal import butter, lfilter, spectrogram
import plotly.graph_objs as go
import zipfile
import re
import datetime

st.title("Анализ и визуализация ЭКоГ данных крыс WAG/Rij")

if 'temp_dir' not in st.session_state:
    st.session_state['temp_dir'] = tempfile.TemporaryDirectory()

temp_dir = st.session_state['temp_dir'].name

edf_file_paths = []
txt_file_paths = []
original_file_names = []

def process_uploaded_files(uploaded_files):
    for uploaded_file in uploaded_files:
        original_file_names.append(uploaded_file.name)
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.read())
        edf_file_paths.append(temp_file_path)

def process_uploaded_txt_files(uploaded_txt_files):
    for uploaded_file in uploaded_txt_files:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.read())
        txt_file_paths.append(temp_file_path)

def process_uploaded_zip(uploaded_zip):
    with zipfile.ZipFile(uploaded_zip) as zf:
        for member in zf.infolist():
            if member.filename.endswith('.edf') or member.filename.endswith('.txt'):
                member.filename = os.path.basename(member.filename)
                zf.extract(member, temp_dir)
                file_path = os.path.join(temp_dir, member.filename)
                if member.filename.endswith('.edf'):
                    edf_file_paths.append(file_path)
                    original_file_names.append(member.filename)
                elif member.filename.endswith('.txt'):
                    txt_file_paths.append(file_path)

upload_option = st.radio("Выберите способ загрузки файлов", ("Загрузить файлы (.edf и .txt)", "Загрузить архив .zip с файлами"))

if upload_option == "Загрузить файлы (.edf и .txt)":
    uploaded_files = st.file_uploader("Загрузите файлы .edf", type="edf", accept_multiple_files=True)
    uploaded_txt_files = st.file_uploader("Загрузите файлы .txt", type="txt", accept_multiple_files=True)
    if uploaded_files:
        process_uploaded_files(uploaded_files)
    if uploaded_txt_files:
        process_uploaded_txt_files(uploaded_txt_files)
elif upload_option == "Загрузить архив .zip с файлами":
    uploaded_zip = st.file_uploader("Загрузите архив .zip с файлами", type="zip")
    if uploaded_zip:
        process_uploaded_zip(uploaded_zip)

if edf_file_paths:
    st.sidebar.title("Выбор файла")
    selected_file = st.sidebar.selectbox("Выберите файл для анализа", original_file_names)
    selected_file_index = original_file_names.index(selected_file)
    selected_file_path = edf_file_paths[selected_file_index]

    # Поиск соответствующего TXT-файла
    base_filename = os.path.splitext(os.path.basename(selected_file))[0]
    base_filename = base_filename.split('.')[0]
    if base_filename.endswith("fully_marked"):
        base_filename = base_filename.split("_fully_marked")[0]
    matching_txt_file = None
    for txt_file in txt_file_paths:
        txt_base_filename = os.path.splitext(os.path.basename(txt_file))[0]
        if txt_base_filename == base_filename:
            matching_txt_file = txt_file
            break

    # Функция для преобразования времени в секунды
    def time_to_seconds(t):
        try:
            if pd.isnull(t) or t == '':
                return None
            parts = list(map(int, re.split('[:]', str(t))))
            if len(parts) == 3:
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
            elif len(parts) == 2:
                return parts[0] * 60 + parts[1]
            elif len(parts) == 1:
                return parts[0]
            else:
                return None
        except ValueError:
            return None

    # Функция для парсинга и поиска корректных интервалов
    def parse_intervals(file_path):
        intervals_df = pd.read_csv(file_path, sep='\t', header=None, names=['NN', 'Время', 'Маркер'])
        intervals_df['Начало'] = intervals_df['Время'].apply(time_to_seconds)

        # Удаляем строки с некорректным временем
        intervals_df['Начало'] = pd.to_numeric(intervals_df['Начало'], errors='coerce')
        intervals_df = intervals_df.dropna(subset=['Начало']).reset_index(drop=True)

        paired_intervals = []
        open_markers = {}

        for _, row in intervals_df.iterrows():
            marker_match = re.match(r"([a-zA-Z]+)", row['Маркер'])
            if marker_match:
                marker_base = marker_match.group(0)
                if marker_base in open_markers:
                    start_row = open_markers[marker_base]
                    # Проверяем, что время корректное
                    if isinstance(row['Начало'], (int, float)) and isinstance(start_row['Начало'], (int, float)):
                        if row['Начало'] > start_row['Начало']:
                            paired_intervals.append((start_row['Начало'], row['Начало'], marker_base))
                    del open_markers[marker_base]
                else:
                    open_markers[marker_base] = row

        # Создаем DataFrame с корректными парами
        if paired_intervals:
            valid_intervals_df = pd.DataFrame(
                paired_intervals,
                columns=['Начало', 'Конец', 'Маркер']
            )
        else:
            valid_intervals_df = pd.DataFrame(columns=['Начало', 'Конец', 'Маркер'])

        return valid_intervals_df

    # Парсинг TXT-файла с использованием новой логики
    intervals_df = None
    marker_colors = {}
    if matching_txt_file:
        intervals_df = parse_intervals(matching_txt_file)

        if not intervals_df.empty:
            # Преобразование времени в формат mm:ss - mm:ss
            # intervals_df['Время_формат'] = intervals_df['Начало'].apply(
            #     lambda x: str(datetime.timedelta(seconds=int(x)))[2:7] if pd.notnull(x) else 'Unknown'
            # )
            intervals_df['Время_формат'] = intervals_df.apply(lambda row:
            (str(datetime.timedelta(seconds=int(row['Начало'])))[2:7] if pd.notnull(row['Начало']) else 'Unknown') + " - " +
            (str(datetime.timedelta(seconds=int(row['Конец'])))[2:7] if pd.notnull(row['Конец']) else 'Unknown'), axis=1)

            # Сортировка интервалов по времени начала
            intervals_df = intervals_df.sort_values(by='Начало').reset_index(drop=True)

            # Словарь для расшифровки маркеров
            marker_explanations = {
                'swd': 'эпилептический разряд',
                'ds': 'фаза глубокого сна',
                'is': 'промежуточная фаза сна'
            }

            # Маркеры уже содержат базовые маркеры
            intervals_df['Базовый_маркер'] = intervals_df['Маркер']

            unique_base_markers = intervals_df['Базовый_маркер'].unique()

            # Добавляем выбор цвета для каждого базового маркера с расшифровкой
            marker_colors = {}
            color_options = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
            st.sidebar.subheader("Выбор цветов для маркеров")
            for i, base_marker in enumerate(unique_base_markers):
                explanation = marker_explanations.get(base_marker, base_marker)
                color = st.sidebar.selectbox(f"Цвет для {explanation} ({base_marker})", color_options,
                                             index=i % len(color_options), key=f"color_{base_marker}")
                marker_colors[base_marker] = color
            # for base_marker in unique_base_markers:
            #     explanation = marker_explanations.get(base_marker, base_marker)
            #     color = st.sidebar.selectbox(f"Цвет для {explanation} ({base_marker})", color_options, index=0, key=f"color_{base_marker}")
            #     marker_colors[base_marker] = color
        else:
            st.write("Нет корректных интервалов в TXT-файле.")

    try:
        # Используем контекстный менеджер для автоматического закрытия файла
        with pyedflib.EdfReader(selected_file_path) as edf_reader:
            n_channels = edf_reader.signals_in_file
            signal_labels = edf_reader.getSignalLabels()

            st.write("**Информация о данных:**")
            st.write("Выбранный файл:", selected_file)
            st.write("Количество каналов:", n_channels)
            st.write("Названия каналов:", signal_labels)

            selected_channel = st.selectbox("Выберите канал для анализа", signal_labels)
            channel_index = signal_labels.index(selected_channel)

            data = edf_reader.readSignal(channel_index)
            sfreq = edf_reader.getSampleFrequency(channel_index)
            nyquist = 0.5 * sfreq  # Определяем nyquist здесь
            total_duration = len(data) / sfreq

        st.write("**Настройка временного окна:**")
        start_time = st.number_input("Начальное время (в секундах)", min_value=0.0, max_value=total_duration, value=0.0)
        duration = st.number_input(
            "Длительность окна (в секундах)",
            min_value=0.1,
            max_value=total_duration - start_time,
            value=min(10.0, total_duration - start_time),
        )
        end_time = start_time + duration

        if end_time > total_duration:
            st.warning("Конечное время превышает длительность записи. Корректируем конечное время.")
            end_time = total_duration
            duration = end_time - start_time

        start_idx = int(start_time * sfreq)
        end_idx = int(end_time * sfreq)

        signal_data = data[start_idx:end_idx]

        if len(signal_data) == 0:
            st.error("Выбранное временное окно не содержит данных. Пожалуйста, выберите другой интервал.")
        else:
            filter_type = st.selectbox(
                "Выберите тип фильтрации",
                ("Без фильтрации", "Фильтр Калмана", "Фильтр Баттерворта"),
            )

            if filter_type == "Фильтр Калмана":
                kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
                signal_data_filtered, _ = kf.filter(signal_data)
                signal_data_filtered = signal_data_filtered.flatten()
            elif filter_type == "Фильтр Баттерворта":
                lowcut = st.number_input(
                    "Нижняя частота среза (Гц)", min_value=0.1, max_value=nyquist - 0.1, value=0.5
                )
                highcut = st.number_input(
                    "Верхняя частота среза (Гц)", min_value=lowcut + 0.1, max_value=nyquist - 0.1, value=40.0
                )
                order = st.slider("Порядок фильтра", 1, 10, 5)
                b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
                signal_data_filtered = lfilter(b, a, signal_data)
            else:
                signal_data_filtered = signal_data

            time_array = np.linspace(start_time, end_time, len(signal_data))

            # Преобразование time_array в datetime
            time_array_datetime = [datetime.datetime(1900, 1, 1) + datetime.timedelta(seconds=ts) for ts in time_array]

            # Функция для построения основного графика
            def plot_main_signal(highlighted_interval_idx=None):
                st.subheader(f"Сигнал канала: {selected_channel}")
                fig = go.Figure()

                # Отображаем исходный сигнал
                fig.add_trace(
                    go.Scatter(
                        x=time_array_datetime,
                        y=signal_data,
                        mode="lines",
                        name="Исходный сигнал",
                        line=dict(width=1, color="blue"),
                    )
                )

                # Отображаем фильтрованный сигнал, если выбран фильтр
                if filter_type != "Без фильтрации":
                    fig.add_trace(
                        go.Scatter(
                            x=time_array_datetime,
                            y=signal_data_filtered,
                            mode="lines",
                            name="Отфильтрованный сигнал",
                            line=dict(width=1, color="red"),
                        )
                    )

                # Добавление маркеров из TXT-файла с выбранными цветами
                if intervals_df is not None and not intervals_df.empty:
                    intervals_in_range = intervals_df[
                        (intervals_df['Начало'] <= end_time) & (intervals_df['Конец'] >= start_time)
                        ]
                    for idx, row in intervals_in_range.iterrows():
                        interval_start = max(row['Начало'], start_time)
                        interval_end = min(row['Конец'], end_time)
                        x0 = datetime.datetime(1900, 1, 1) + datetime.timedelta(seconds=interval_start)
                        x1 = datetime.datetime(1900, 1, 1) + datetime.timedelta(seconds=interval_end)
                        base_marker = row['Базовый_маркер']
                        color = marker_colors.get(base_marker, 'red')  # Получаем цвет базового маркера

                        # Проверяем, является ли текущий интервал выбранным
                        if highlighted_interval_idx is not None and idx == highlighted_interval_idx:
                            opacity = 0.5  # Более заметное выделение для выбранного интервала
                            line_width = 2  # Обводка для выделения
                        else:
                            opacity = 0.2
                            line_width = 0

                        fig.add_vrect(
                            x0=x0,
                            x1=x1,
                            fillcolor=color,
                            opacity=opacity,
                            layer="below",
                            line_width=line_width,
                            line_color='black',  # Цвет обводки
                            annotation_text=row['Маркер'],
                            annotation_position="top left"
                        )

                fig.update_layout(
                    xaxis_title="Время (мм:сс)",
                    yaxis_title="Амплитуда",
                    legend=dict(x=0, y=1),
                    hovermode="x unified",
                    height=400,
                    xaxis=dict(
                        type='date',
                        tickformat='%M:%S',
                        rangeslider=dict(visible=True),
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)


            # Боковая панель с интервалами
            if intervals_df is not None and not intervals_df.empty:
                st.sidebar.subheader("Интервалы из TXT-файла")
                # Добавляем опцию "Без выбранного интервала"
                interval_options = ['Без выбранного интервала'] + intervals_df.apply(
                    lambda row: f"{row['Маркер']} ({row['Время_формат']})", axis=1).tolist()
                selected_interval = st.sidebar.selectbox(
                    "Выберите интервал",
                    interval_options
                )
                show_interval = st.sidebar.button("Показать интервал на графике", key="show_interval_button")

                if selected_interval != 'Без выбранного интервала' and show_interval:
                    interval_idx = intervals_df.index[interval_options.index(selected_interval) - 1]
                    interval_start = intervals_df.loc[interval_idx, 'Начало']
                    interval_end = intervals_df.loc[interval_idx, 'Конец']
                    st.write(f"**Выбранный интервал:** {selected_interval}")
                    start_time_str = str(datetime.timedelta(seconds=int(interval_start)))[2:7]
                    end_time_str = str(datetime.timedelta(seconds=int(interval_end)))[2:7]
                    st.write(f"**Время:** {start_time_str} - {end_time_str}")

                    # Не изменяем временное окно, просто выделяем выбранный интервал на графике
                    plot_main_signal(highlighted_interval_idx=interval_idx)
                else:
                    # Если выбран "Без выбранного интервала" или кнопка не нажата
                    st.write("Отображается текущий сигнал с интервалами.")
                    plot_main_signal()
            else:
                # Если нет интервалов или TXT-файл не загружен
                st.write("Интервалы не найдены или TXT-файл не загружен.")
                plot_main_signal()

            # Спектрограмма
            st.subheader("Спектрограмма сигнала")
            f, t_spec, Sxx = spectrogram(signal_data_filtered, fs=sfreq)
            t_spec_datetime = [datetime.datetime(1900, 1, 1) + datetime.timedelta(seconds=ts + start_time) for ts in t_spec]
            fig_spectrogram = go.Figure(data=go.Heatmap(
                x=t_spec_datetime,
                y=f,
                z=10 * np.log10(Sxx),
                colorscale='Viridis'
            ))
            fig_spectrogram.update_layout(
                xaxis_title="Время (мм:сс)",
                yaxis_title="Частота (Гц)",
                height=400,
                xaxis=dict(
                    type='date',
                    tickformat='%M:%S',
                ),
            )
            st.plotly_chart(fig_spectrogram, use_container_width=True)

            # Разбиение по частотным диапазонам
            st.subheader("Разбиение по частотным диапазонам")
            bands = {
                'Delta (0.5-4 Hz)': (0.5, 4),
                'Theta (4-8 Hz)': (4, 8),
                'Alpha (8-13 Hz)': (8, 13),
                'Beta (13-30 Hz)': (13, 30),
                'Gamma (30-100 Hz)': (30, 100)
            }
            for band_name, (low, high) in bands.items():
                b, a = butter(4, [low / nyquist, high / nyquist], btype="band")
                band_data = lfilter(b, a, signal_data_filtered)
                fig_band = go.Figure()
                fig_band.add_trace(
                    go.Scatter(
                        x=time_array_datetime,
                        y=band_data,
                        mode="lines",
                        name=band_name,
                        line=dict(width=1),
                    )
                )
                fig_band.update_layout(
                    xaxis_title="Время (мм:сс)",
                    yaxis_title="Амплитуда",
                    hovermode="x unified",
                    height=300,
                    xaxis=dict(
                        type='date',
                        tickformat='%M:%S',
                    ),
                    title=band_name
                )
                st.plotly_chart(fig_band, use_container_width=True)

            # 3D Визуализация мозга
            st.subheader("3D Визуализация мозга")
            # Закомментируем код визуализации до тех пор, пока не будет найден корректный источник модели мозга
            st.write("3D визуализация мозга временно недоступна.")
            # try:
            #     from urllib.request import urlopen
            #     import json

            #     url = 'https://raw.githubusercontent.com/plotly/datasets/master/mesh3d_sample_data/brain_surface.obj'
            #     response = urlopen(url)
            #     brain_mesh_data = response.read().decode('utf-8')

            #     # Парсинг OBJ-файла или использование альтернативного способа визуализации

            #     st.write("Модель мозга успешно загружена.")
            # except Exception as e:
            #     st.error(f"Не удалось загрузить модель мозга: {e}")
            #     st.write("Попробуйте снова или обратитесь к администратору.")

    except Exception as e:
        st.error(f"Произошла ошибка: {e}")
