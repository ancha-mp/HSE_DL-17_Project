from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from pickle import dump, load
import pandas as pd

CATEGORICAL_FEATURES = ["key", "mode"]

def split_data(df: pd.DataFrame):
    y = df["music_genre"]
    X = df[["key", "mode", "acousticness", "danceability", "energy",
        "instrumentalness", "liveness", "loudness", "tempo", "speechiness"]]

    return X, y


def open_data(path="data/music_genre_train.csv"):
    df = pd.read_csv(path)
    cols = [
        "key", "mode", "acousticness", "danceability", "energy",
        "instrumentalness", "liveness", "loudness", "tempo", "speechiness", "music_genre"
    ]
    df = df[cols]
    df = df.dropna().reset_index(drop=True)
    
    return df

def preprocess_data(df: pd.DataFrame, test=True):
       # Привести категориальные к строкам
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)
    # Привести числовые к float
    for col in NUM_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Оставить только нужные столбцы В ТОЧНОМ ПОРЯДКЕ
    df = df[ALL_FEATURES]
    df = df.dropna().reset_index(drop=True)
    if test and "music_genre" in df.columns:
        X_df, y_df = split_data(df)
        return X_df, y_df
    else:
        return df

def fit_and_save_model(X_df, y_df, path="data/model_weights.cbm"):
    model = CatBoostClassifier(
        iterations=300,
        random_seed=42,
        verbose=0,
        cat_features=CATEGORICAL_FEATURES
    )
    model.fit(X_df, y_df)
    acc = accuracy_score(y_df, model.predict(X_df))
    print(f"Train accuracy: {acc:.3f}")
    model.save_model(path)
    print(f"Model saved to {path}")
    

def load_model_and_predict(df, path="data/model_weights.cbm"):
    model = CatBoostClassifier()
    model.load_model(path)
    prediction = model.predict(df)[0]
    probas = model.predict_proba(df)[0]
    classes = model.classes_
    prediction_probas = {cls: prob for cls, prob in zip(classes, probas)}
    return prediction, prediction_probas
    
# Скрипт для обучения!
if __name__ == "__main__":
    df = open_data()
    X, y = split_data(df)
    fit_and_save_model(X, y)
