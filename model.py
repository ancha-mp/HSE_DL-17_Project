from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from pickle import dump, load
import pandas as pd


def split_data(df: pd.DataFrame):
    y = df["music_genre"]
    X = df[["key", "mode", "acousticness", "danceability", "energy", "instrumentalness", "liveness", "loudness", "tempo", "speechiness"]]

    return X, y


def open_data(path="data/music_genre_train.csv"):
    df = pd.read_csv(path)
    df = df[["key", "mode", "acousticness", "danceability", "energy", "instrumentalness", "liveness", "loudness", "tempo", "speechiness", "music_genre"]]

    return df


def preprocess_data(df: pd.DataFrame, test=True):
    df.dropna(inplace=True)

    if test:
        X_df, y_df = split_data(df)
        return X_df, y_df
    else:
        X_df = df
        return df
        
  
def fit_and_save_model(X_df, y_df, path="data/model_weights.cbm"):
    model = CatBoostClassifier(verbose=0)
    model.fit(X_df, y_df)

    test_prediction = model.predict(X_df)
    accuracy = accuracy_score(test_prediction, y_df)
    print(f"Model accuracy is {accuracy}")

    with open(path, "wb") as file:
        dump(model, file)

    print(f"Model was saved to {path}")


def load_model_and_predict(df, path="data/model_weights.cbm"):
    model = CatBoostClassifier()
    model.load_model(path)

    prediction = model.predict(df)[0]
    # prediction = np.squeeze(prediction)

    prediction_proba = model.predict_proba(df)[0]
    # prediction_proba = np.squeeze(prediction_proba)

    encode_prediction_proba = {
        0: "Вам не повезло с вероятностью",
        1: "Вы выживете с вероятностью"
    }

    encode_prediction = {
        0: "Сожалеем, вам не повезло",
        1: "Ура! Вы будете жить"
    }

    prediction_data = {}
    for key, value in encode_prediction_proba.items():
        prediction_data.update({value: prediction_proba[key]})

    prediction_df = pd.DataFrame(prediction_data, index=[0])
    prediction = encode_prediction[prediction]

    return prediction, prediction_df


if __name__ == "__main__":
    df = open_data()
    X_df, y_df = preprocess_data(df)
    fit_and_save_model(X_df, y_df)
