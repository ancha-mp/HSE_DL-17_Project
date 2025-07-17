import pandas as pd
import streamlit as st
from PIL import Image
from model import open_data, preprocess_data, split_data, load_model_and_predict

def process_main_page():
    show_main_page()
    process_side_bar_inputs()

def show_main_page():
    image = Image.open('data/music2.jpg')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Music Genre Prediction",
        page_icon=image,

    )

    st.write(
        """
        # Классификация музыкальных жанров произведения
        Определяем, музычку, если вы ее опишете.
        """
    )

    st.image(image)

def write_user_data(df):
    st.write("## Ваши данные")
    st.write(df)

#def write_prediction(prediction, prediction_probas):
#    st.write("## Предсказание")
#    st.write(prediction)

#    st.write("## Вероятность предсказания")
#    st.write(prediction_probas)

def process_side_bar_inputs():
    st.sidebar.header('Заданные пользователем параметры')
    user_input_df = sidebar_input_features()

    train_df = open_data()
    train_X_df, _ = split_data(train_df)
    full_X_df = pd.concat((user_input_df, train_X_df), axis=0)
    preprocessed_X_df = preprocess_data(full_X_df, test=False)

    user_X_df = preprocessed_X_df[:1]
    write_user_data(user_X_df)

    prediction, prediction_probas = load_model_and_predict(user_X_df)
    write_prediction(prediction, prediction_probas)


def sidebar_input_features():
    key = st.sidebar.selectbox("Тональность", ("A", "B", "C", "E", "F", "G", "A#", "C#", "G#"))
    mode = st.sidebar.selectbox("Mode", ("Minor", "Major"))
    
    acousticness = st.sidebar.slider("acousticness", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    danceability = st.sidebar.slider("danceability", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    energy = st.sidebar.slider("energy", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    instrumentalness = st.sidebar.slider("instrumentalness", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    liveness = st.sidebar.slider("liveness", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    loudness = st.sidebar.slider("loudness", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    tempo = st.sidebar.slider("tempo", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    speechiness = st.sidebar.slider("speechiness", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
        
    data = {
        "key": key,
        "mode": mode,
        "acousticness": acousticness,
        "danceability": danceability,
        "energy": energy,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "loudness": loudness,
        "tempo": tempo,
        "speechiness": speechiness
    }

    df = pd.DataFrame(data, index=[0])

    return df


if __name__ == "__main__":
    process_main_page()
