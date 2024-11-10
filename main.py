import streamlit as st
import pyedflib
import numpy as np
import pandas as pd
import tempfile
import os
from pykalman import KalmanFilter
from scipy.signal import butter, lfilter, spectrogram, decimate
import plotly.graph_objs as go
import zipfile
import re
import datetime
import streamlit.components.v1 as components

st.title("Анализ и визуализация ЭКоГ данных крыс WAG/Rij")

# Инициализация session_state для хранения интервалов и цветов маркеров
if 'temp_dir' not in st.session_state:
    st.session_state['temp_dir'] = tempfile.TemporaryDirectory()

if 'intervals_df' not in st.session_state:
    st.session_state['intervals_df'] = pd.DataFrame(columns=['Начало', 'Конец', 'Маркер'])

if 'marker_colors' not in st.session_state:
    st.session_state['marker_colors'] = {}

temp_dir = st.session_state['temp_dir'].name

edf_file_paths = []
txt_file_paths = []
original_file_names = []

# Функции для обработки загрузки файлов
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

# Выбор способа загрузки файлов
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
        intervals_df = pd.read_csv(file_path, sep='\t', header=0)  # Убедитесь, что заголовок есть
        required_columns = {'NN', 'время', 'маркер'}
        if not required_columns.issubset(intervals_df.columns):
            st.error(f"TXT файл должен содержать колонки: {required_columns}")
            return pd.DataFrame(columns=['Начало', 'Конец', 'Маркер'])

        # Сортировка по NN для правильного парирования
        intervals_df = intervals_df.sort_values(by='NN').reset_index(drop=True)

        # Парирование ds1/ds2, is1/is2 и т.д.
        paired_intervals = []
        open_markers = {}

        for _, row in intervals_df.iterrows():
            marker = row['маркер']
            base_marker_match = re.match(r"([a-zA-Z]+)", marker)
            if base_marker_match:
                base_marker = base_marker_match.group(1)
                if marker.endswith('1'):
                    # Начало интервала
                    open_markers[base_marker] = row['время']
                elif marker.endswith('2'):
                    # Конец интервала
                    if base_marker in open_markers:
                        start_time = time_to_seconds(open_markers[base_marker])
                        end_time = time_to_seconds(row['время'])
                        if start_time is not None and end_time is not None and end_time > start_time:
                            paired_intervals.append({'Начало': start_time, 'Конец': end_time, 'Маркер': base_marker})
                        del open_markers[base_marker]
                    else:
                        st.warning(f"Конец интервала {marker} найден без соответствующего начала.")
            else:
                st.warning(f"Неправильный формат маркера: {marker}")

        # Создаем DataFrame с корректными парами
        if paired_intervals:
            valid_intervals_df = pd.DataFrame(paired_intervals)
        else:
            valid_intervals_df = pd.DataFrame(columns=['Начало', 'Конец', 'Маркер'])

        return valid_intervals_df

    # Функция для форматирования времени в hh:mm:ss
    def seconds_to_hh_mm_ss(seconds):
        try:
            td = datetime.timedelta(seconds=float(seconds))
            total_seconds = int(td.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            secs = total_seconds % 60
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        except:
            return 'Unknown'

    # Функция для загрузки и обработки сигнала из EDF
    def load_edf_signal(edf_path, channel_name=None):
        with pyedflib.EdfReader(edf_path) as edf_reader:
            n_channels = edf_reader.signals_in_file
            signal_labels = edf_reader.getSignalLabels()
            if channel_name and channel_name in signal_labels:
                channel_index = signal_labels.index(channel_name)
            else:
                channel_index = 0  # По умолчанию первый канал
                channel_name = signal_labels[channel_index]
            data = edf_reader.readSignal(channel_index)
            sfreq = edf_reader.getSampleFrequency(channel_index)
            total_duration = len(data) / sfreq
            time_array = np.linspace(0, total_duration, len(data))
        return data, sfreq, time_array, channel_name

    # Функция для присваивания цветов маркерам (только базовые маркеры)
    def assign_marker_colors(intervals_df):
        unique_markers = intervals_df['Маркер'].unique()
        for i, marker in enumerate(unique_markers):
            if marker not in st.session_state['marker_colors']:
                color_options = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
                color = st.sidebar.selectbox(
                    f"Цвет для {marker}",
                    color_options,
                    index=i % len(color_options),
                    key=f"color_{marker}"
                )
                st.session_state['marker_colors'][marker] = color
        return st.session_state['marker_colors']

    # Функция для создания основного Plotly графика
    def plot_main_signal(signal_data, signal_data_filtered, time_array_datetime, intervals_df, marker_colors, selected_interval_idx=None, xrange=None):
        fig = go.Figure()

        # Отображаем исходный сигнал
        fig.add_trace(
            go.Scatter(
                x=time_array_datetime,
                y=signal_data,
                mode="lines",
                name="Исходный сигнал",
                line=dict(width=1, color="black"),
                hovertemplate='Время: %{x}<br>Амплитуда: %{y}<extra></extra>',
            )
        )

        # Отображаем фильтрованный сигнал, если выбран фильтр
        if signal_data_filtered is not None and not np.array_equal(signal_data, signal_data_filtered):
            fig.add_trace(
                go.Scatter(
                    x=time_array_datetime,
                    y=signal_data_filtered,
                    mode="lines",
                    name="Фильтрованный сигнал",
                    line=dict(width=1, color="red"),
                    hovertemplate='Время: %{x}<br>Амплитуда: %{y}<extra></extra>',
                )
            )

        # Добавляем прямоугольники для интервалов
        if intervals_df is not None and not intervals_df.empty:
            for idx, row in intervals_df.iterrows():
                interval_start = row['Начало']
                interval_end = row['Конец']
                marker = row['Маркер']
                color = marker_colors.get(marker, 'gray')

                # Преобразуем секунды в datetime, приводя к стандартному типу
                x0 = datetime.datetime(1900, 1, 1) + datetime.timedelta(seconds=float(interval_start))
                x1 = datetime.datetime(1900, 1, 1) + datetime.timedelta(seconds=float(interval_end))

                # Выделяем выбранный интервал
                if selected_interval_idx is not None and idx == selected_interval_idx:
                    opacity = 0.6
                    line_width = 2
                else:
                    opacity = 0.3
                    line_width = 0

                fig.add_vrect(
                    x0=x0,
                    x1=x1,
                    fillcolor=color,
                    opacity=opacity,
                    layer="below",
                    line_width=line_width,
                    line_color='black',
                )

        # Добавляем легенду для маркеров
        for marker, color in marker_colors.items():
            explanation = marker  # Можно добавить более понятные названия
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(color=color, size=10),
                    name=explanation,
                    showlegend=True
                )
            )

        # Обновляем макет графика
        layout = dict(
            title="Визуализация сигнала канала",
            xaxis_title="Время (hh:mm:ss)",
            yaxis_title="Амплитуда",
            legend=dict(x=0, y=1),
            hovermode="x unified",
            height=600,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                tickformat='%H:%M:%S',
            ),
        )

        if xrange:
            layout['xaxis']['range'] = xrange

        fig.update_layout(layout)

        st.plotly_chart(fig, use_container_width=True)

    # Функция для создания дополнительного графика всех интервалов
    def plot_all_intervals(intervals_df, marker_colors, total_duration):
        fig = go.Figure()

        for _, row in intervals_df.iterrows():
            interval_start = row['Начало']
            interval_end = row['Конец']
            marker = row['Маркер']
            color = marker_colors.get(marker, 'gray')

            # Преобразуем секунды в datetime, приводя к стандартному типу
            x0 = datetime.datetime(1900, 1, 1) + datetime.timedelta(seconds=float(interval_start))
            x1 = datetime.datetime(1900, 1, 1) + datetime.timedelta(seconds=float(interval_end))

            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=color,
                opacity=0.5,
                layer="below",
                line_width=0
            )

        # Добавляем легенду для маркеров
        for marker, color in marker_colors.items():
            explanation = marker  # Можно добавить более понятные названия
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(color=color, size=10),
                    name=explanation,
                    showlegend=True
                )
            )

        fig.update_layout(
            title="Все интервалы и их цвета",
            xaxis_title="Время (hh:mm:ss)",
            yaxis=dict(showticklabels=False),
            legend=dict(x=0, y=1),
            height=300,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                range=[datetime.datetime(1900, 1, 1), datetime.datetime(1900, 1, 1) + datetime.timedelta(seconds=total_duration)],
                tickformat='%H:%M:%S',
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

    # Функция для генерации интервалов программно
    def generate_intervals(total_duration, num_intervals=5, min_duration=10, max_duration=60):
        np.random.seed(42)  # Для воспроизводимости
        intervals = []
        available_segments = [(0, total_duration)]

        for _ in range(num_intervals):
            if not available_segments:
                st.warning("Недостаточно доступного времени для генерации всех интервалов.")
                break  # No more available time to allocate intervals

            # Выбираем случайный доступный сегмент
            segment_idx = np.random.randint(0, len(available_segments))
            segment_start, segment_end = available_segments.pop(segment_idx)

            # Определяем максимально возможную длительность в этом сегменте
            max_possible_duration = min(max_duration, segment_end - segment_start)
            if max_possible_duration < min_duration:
                continue  # Недостаточно места для генерации интервала

            # Генерируем длительность интервала
            duration = np.random.uniform(min_duration, max_possible_duration)
            # Генерируем начальное время интервала в пределах сегмента
            start = np.random.uniform(segment_start, segment_end - duration)
            end = start + duration
            marker = np.random.choice(['ds', 'is', 'swd'])  # Выбор случайного маркера
            intervals.append({'Начало': start, 'Конец': end, 'Маркер': marker})

            # Обновляем доступные сегменты
            if start > segment_start:
                available_segments.append((segment_start, start))
            if end < segment_end:
                available_segments.append((end, segment_end))

        # Сортируем интервалы по времени начала
        intervals = sorted(intervals, key=lambda x: x['Начало'])
        return pd.DataFrame(intervals)

    # Функция для фильтрации сигнала
    def filter_signal(signal_data, filter_type, lowcut, highcut, order, sfreq):
        if filter_type == "Фильтр Калмана":
            kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
            signal_data_filtered, _ = kf.filter(signal_data)
            return signal_data_filtered.flatten()
        elif filter_type == "Фильтр Баттерворта":
            b, a = butter(order, [lowcut / (0.5 * sfreq), highcut / (0.5 * sfreq)], btype="band")
            return lfilter(b, a, signal_data)
        else:
            return signal_data  # Без фильтрации

    try:
        # Загрузка сигнала из EDF
        data, sfreq, time_array, channel_name = load_edf_signal(selected_file_path)
        nyquist = 0.5 * sfreq  # Частота Найквиста
        total_duration = time_array[-1]

        # Парсинг интервалов из TXT
        if matching_txt_file:
            parsed_intervals = parse_intervals(matching_txt_file)
            if not parsed_intervals.empty:
                st.session_state['intervals_df'] = parsed_intervals
                # Преобразование времени в формат hh:mm:ss - hh:mm:ss
                st.session_state['intervals_df']['Время_формат'] = st.session_state['intervals_df'].apply(
                    lambda row: f"{seconds_to_hh_mm_ss(row['Начало'])} - {seconds_to_hh_mm_ss(row['Конец'])}", axis=1
                )
                # Присваиваем цвета маркерам через функцию
                st.session_state['marker_colors'] = assign_marker_colors(st.session_state['intervals_df'])
            else:
                st.write("Нет корректных интервалов в TXT-файле.")

        # Добавление кнопки для генерации интервалов
        st.sidebar.subheader("Генерация интервалов")
        generate_button = st.sidebar.button("Сгенерировать интервалы программно")

        if generate_button:
            # Генерация интервалов
            generated_intervals = generate_intervals(total_duration, num_intervals=15, min_duration=100, max_duration=1000)
            st.session_state['intervals_df'] = generated_intervals

            # Добавляем форматированное время
            st.session_state['intervals_df']['Время_формат'] = st.session_state['intervals_df'].apply(
                lambda row: f"{seconds_to_hh_mm_ss(row['Начало'])} - {seconds_to_hh_mm_ss(row['Конец'])}", axis=1
            )

            # Присваиваем цвета маркерам
            st.session_state['marker_colors'] = assign_marker_colors(st.session_state['intervals_df'])

            st.sidebar.success("Интервалы успешно сгенерированы!")

        intervals_df = st.session_state['intervals_df']
        marker_colors = st.session_state['marker_colors']

        # Настройка временного окна
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

        # Выбор фильтрации
        filter_type = st.selectbox(
            "Выберите тип фильтрации",
            ("Без фильтрации", "Фильтр Калмана", "Фильтр Баттерворта"),
        )

        if filter_type == "Фильтр Баттерворта":
            lowcut = st.number_input(
                "Нижняя частота среза (Гц)", min_value=0.1, max_value=nyquist - 0.1, value=0.5
            )
            highcut = st.number_input(
                "Верхняя частота среза (Гц)", min_value=lowcut + 0.1, max_value=nyquist - 0.1, value=40.0
            )
            order = st.slider("Порядок фильтра", 1, 10, 5)
        else:
            lowcut = highcut = order = None  # Задаем значения по умолчанию

        # Боковая панель с интервалами
        if intervals_df is not None and not intervals_df.empty:
            st.sidebar.subheader("Интервалы из TXT-файла или сгенерированные")
            # Добавляем опцию "Без выбранного интервала"
            interval_options = ['Без выбранного интервала'] + intervals_df.apply(
                lambda row: f"{row['Маркер']} ({row['Время_формат']})", axis=1).tolist()
            selected_interval = st.sidebar.selectbox(
                "Выберите интервал",
                interval_options
            )
            show_interval = st.sidebar.button("Показать интервал на графике", key="show_interval_button")
        else:
            selected_interval = 'Без выбранного интервала'
            show_interval = False

        # Обработка выбора интервала
        if show_interval:
            if selected_interval != 'Без выбранного интервала':
                # Получаем индекс и временные границы выбранного интервала
                interval_idx = interval_options.index(selected_interval) - 1
                interval_start = intervals_df.loc[interval_idx, 'Начало']
                interval_end = intervals_df.loc[interval_idx, 'Конец']
                start_time, end_time = interval_start, interval_end  # Устанавливаем временное окно для выбранного интервала
                st.write(f"**Выбранный интервал:** {selected_interval}")
                st.write(f"**Время:** {seconds_to_hh_mm_ss(interval_start)} - {seconds_to_hh_mm_ss(interval_end)}")

                # Обновляем данные сигнала для выбранного интервала
                start_idx = int(start_time * sfreq)
                end_idx = int(end_time * sfreq)

                signal_data = data[start_idx:end_idx]
                if filter_type != "Без фильтрации":
                    signal_data_filtered = filter_signal(signal_data, filter_type, lowcut, highcut, order, sfreq)
                else:
                    signal_data_filtered = signal_data  # Без фильтрации

                # Обновляем временные метки для отображения
                time_array_selected = np.linspace(start_time, end_time, len(signal_data))
                time_array_datetime = [datetime.datetime(1900, 1, 1) + datetime.timedelta(seconds=float(ts)) for ts in time_array_selected]

                # Устанавливаем диапазон оси X для зума
                xrange = [datetime.datetime(1900, 1, 1) + datetime.timedelta(seconds=float(start_time)),
                          datetime.datetime(1900, 1, 1) + datetime.timedelta(seconds=float(end_time))]

                # Отображаем график с выделенным интервалом и зумом
                plot_main_signal(signal_data, signal_data_filtered, time_array_datetime, intervals_df, marker_colors, selected_interval_idx=interval_idx, xrange=xrange)
            else:
                st.write("Отображается весь сигнал с интервалами.")
                signal_data = data[int(start_time * sfreq):int(end_time * sfreq)]
                if filter_type != "Без фильтрации":
                    signal_data_filtered = filter_signal(signal_data, filter_type, lowcut, highcut, order, sfreq)
                else:
                    signal_data_filtered = signal_data  # Без фильтрации

                time_array_selected = np.linspace(start_time, end_time, len(signal_data))
                time_array_datetime = [datetime.datetime(1900, 1, 1) + datetime.timedelta(seconds=float(ts)) for ts in time_array_selected]

                plot_main_signal(signal_data, signal_data_filtered, time_array_datetime, intervals_df, marker_colors)
        else:
            # Используем временное окно, заданное пользователем
            start_idx = int(start_time * sfreq)
            end_idx = int(end_time * sfreq)
            signal_data = data[start_idx:end_idx]
            if filter_type != "Без фильтрации":
                signal_data_filtered = filter_signal(signal_data, filter_type, lowcut, highcut, order, sfreq)
            else:
                signal_data_filtered = signal_data  # Без фильтрации

            time_array_selected = np.linspace(start_time, end_time, len(signal_data))
            time_array_datetime = [datetime.datetime(1900, 1, 1) + datetime.timedelta(seconds=float(ts)) for ts in time_array_selected]

            plot_main_signal(signal_data, signal_data_filtered, time_array_datetime, intervals_df, marker_colors)

        # Спектрограмма
        st.subheader("Спектрограмма сигнала")
        # Ограничиваем спектрограмму, если данные слишком большие
        max_duration_for_spectrogram = float('inf')  # Можно установить лимит, например, 120 секунд
        if duration > max_duration_for_spectrogram:
            st.warning(f"Спектрограмма не может быть построена для длительности более {max_duration_for_spectrogram} секунд.")
        else:
            f, t_spec, Sxx = spectrogram(signal_data_filtered, fs=sfreq)
            t_spec_datetime = [datetime.datetime(1900, 1, 1) + datetime.timedelta(seconds=float(ts) + start_time) for ts in t_spec]
            fig_spectrogram = go.Figure(data=go.Heatmap(
                x=t_spec_datetime,
                y=f,
                z=10 * np.log10(Sxx + 1e-10),  # Добавляем небольшой констант для избежания log(0)
                colorscale='Viridis'
            ))
            fig_spectrogram.update_layout(
                xaxis_title="Время",
                yaxis_title="Частота (Гц)",
                height=400,
                xaxis=dict(
                    tickformat='%H:%M:%S',
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

        max_points = 50000  # Максимальное количество точек для отображения
        if len(signal_data_filtered) > max_points:
            decimation_factor = int(len(signal_data_filtered) / max_points)
        else:
            decimation_factor = 1

        for band_name, (low, high) in bands.items():
            b, a = butter(4, [low / nyquist, high / nyquist], btype="band")
            band_data = lfilter(b, a, signal_data_filtered)
            # Уменьшаем количество точек для отображения, если необходимо
            if len(band_data) > max_points:
                band_data_to_plot = decimate(band_data, decimation_factor)
                time_array_to_plot = time_array_datetime[::decimation_factor]
            else:
                band_data_to_plot = band_data
                time_array_to_plot = time_array_datetime

            fig_band = go.Figure()
            fig_band.add_trace(
                go.Scatter(
                    x=time_array_to_plot,
                    y=band_data_to_plot,
                    mode="lines",
                    name=band_name,
                    line=dict(width=1),
                )
            )
            fig_band.update_layout(
                xaxis_title="Время",
                yaxis_title="Амплитуда",
                hovermode="x unified",
                height=300,
                xaxis=dict(
                    tickformat='%H:%M:%S',
                ),
                title=band_name
            )
            st.plotly_chart(fig_band, use_container_width=True)

        # 3D Визуализация мозга
        st.subheader("3D Визуализация мозга")
        if os.path.exists("Right Thalamus.html"):
            with open("Right Thalamus.html", 'r') as file:
                brain_html = file.read()

            # Display the HTML 3D brain in Streamlit
            components.html(brain_html, height=600)
        else:
            st.warning("Файл 3D визуализации мозга (Right Thalamus.html) не найден.")

    except Exception as e:
        st.error(f"Произошла ошибка: {e}")
