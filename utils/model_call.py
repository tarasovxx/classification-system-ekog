from functools import cache
import glob
import os

from utils.edf_to_txt_converter import generate_txt_from_edf
from utils.file_processor import process_file

@cache
def get_model():
    pass # TODO вставить чтение модели из пикл

def get_df_from_edf_file(file_path):
    # читаем едф файл

    # читаем тхт в датафрейм
    # подаем датафрейм на модель
    # возвращаем данные
    
    if not glob.glob(file_path):
        raise ValueError(f"File {file_path} is not found")

    df = process_file(file_path)
    X = df.drop(columns=['LABEL','ID', 'X_FROM', 'X_TO'])

    model = get_model()
    y = model.predict(X)
    return X, y

    
    
    