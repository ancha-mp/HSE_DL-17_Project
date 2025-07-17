st.set_page_config(
    layout="wide",
    initial_sidebar_state="auto",
    page_title="Music Genre Prediction",
    page_icon="🎵"
)

def show_main_page():
    image = Image.open('data/music2.jpg')
    st.title("Классификация музыкальных жанров произведения")
    st.write("Определяем музыку, если вы её опишете.")
    st.image(image, use_column_width=True)

def write_user_data(df):
    st.write("## Ваши данные")
    st.dataframe(df)

def write_prediction(prediction, prediction_probas):
    st.write("## Предсказание")
    st.write(prediction)

    st.write("## Вероятности по классам")
    for genre, prob in prediction_probas.items():
        st.write(f"{genre}: {prob:.3f}")

def sidebar_input_features():
    st.sidebar.header('Заданные пользователем параметры')
    key = st.sidebar.selectbox("Тональность", ("A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"))
    mode = st.sidebar.selectbox("Mode", ("Minor", "Major"))
    acousticness = st.sidebar.slider("acousticness", 0.0, 1.0, 0.2, 0.01)
    danceability = st.sidebar.slider("danceability", 0.0, 1.0, 0.2, 0.01)
    energy = st.sidebar.slider("energy", 0.0, 1.0, 0.2, 0.01)
    instrumentalness = st.sidebar.slider("instrumentalness", 0.0, 1.0, 0.2, 0.01)
    liveness = st.sidebar.slider("liveness", 0.0, 1.0, 0.2, 0.01)
    loudness = st.sidebar.slider("loudness", -100.0, 20.0, -10.0, 1.0)
    speechiness = st.sidebar.slider("speechiness", 0.0, 1.0, 0.2, 0.01)
    tempo = st.sidebar.slider("tempo", 0.0, 300.0, 120.0, 1.0)
    
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
    df = pd.DataFrame([data])
    return df

def main():
    show_main_page()
    user_input_df = sidebar_input_features()
    train_df = open_data()
    train_X_df, _ = split_data(train_df)

    # Конкатенируем, чтобы корректно обработать категориальные фичи
    full_X_df = pd.concat([user_input_df, train_X_df], axis=0, ignore_index=True)
    preprocessed_X_df = preprocess_data(full_X_df, test=False)

    user_X_df = preprocessed_X_df.iloc[[0]]
    write_user_data(user_X_df)

    prediction, prediction_probas = load_model_and_predict(user_X_df)
    write_prediction(prediction, prediction_probas)

if __name__ == "__main__":
    main()
