import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# MLflow Tracking Server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Experiment
mlflow.set_experiment("Prediksi Keterlambatan Pengiriman")

# Load dataset hasil preprocessing
data = pd.read_csv("Dataset_olist_preprocessing.csv")

# Split dataset
X = data.drop("is_late", axis=1)
y = data["is_late"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Contoh input untuk model signature
input_example = X_train.iloc[:5]

# Aktifkan autolog
mlflow.autolog()

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=505,
        max_depth=37,
        random_state=42
    )

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    print(f"Accuracy: {accuracy}")

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )