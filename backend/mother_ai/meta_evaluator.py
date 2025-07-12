import os
import joblib
import pandas as pd
from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODEL_PATH = "backend/mother_ai/meta_model.pkl"

class MetaEvaluator:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self) -> Optional[RandomForestClassifier]:
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        return None

    def train(self, data: pd.DataFrame):
        if "score" not in data.columns or "outcome" not in data.columns:
            raise ValueError("Training data must include 'score' and 'outcome' columns")

        features = data.drop(columns=["outcome", "timestamp", "symbol", "signal", "price"])
        labels = data["outcome"]  # e.g., 1 for profitable, 0 for loss

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, max_depth=6)
        model.fit(X_train, y_train)
        joblib.dump(model, self.model_path)
        self.model = model

    def predict_refined_score(self, features: dict) -> float:
        if not self.model:
            return features.get("confidence", 0.5)  # fallback

        df = pd.DataFrame([features])
        score = self.model.predict_proba(df)[0][1]  # probability of success
        return float(score)
