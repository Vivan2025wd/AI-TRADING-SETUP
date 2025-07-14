import os
import joblib
import pandas as pd
from typing import Optional, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODEL_PATH = "backend/mother_ai/meta_model.pkl"

class MetaEvaluator:
    def __init__(self, model_path: str = MODEL_PATH, verbose: bool = False):
        self.model_path = model_path
        self.verbose = verbose
        self.model: Optional[RandomForestClassifier] = self._load_model()

    def _load_model(self) -> Optional[RandomForestClassifier]:
        if os.path.exists(self.model_path):
            if self.verbose:
                print(f"🔍 Loading meta model from {self.model_path}")
            return joblib.load(self.model_path)
        if self.verbose:
            print("⚠️ No trained meta model found.")
        return None

    def train(self, data: pd.DataFrame) -> None:
        if "outcome" not in data.columns:
            raise ValueError("❌ Training data must include 'outcome' column")

        drop_cols = [col for col in ["timestamp", "symbol", "signal", "price", "score"] if col in data.columns]
        features = data.drop(columns=["outcome"] + drop_cols)
        labels = data["outcome"]

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        model.fit(X_train, y_train)

        joblib.dump(model, self.model_path)
        self.model = model

        if self.verbose:
            acc = model.score(X_test, y_test)
            print(f"✅ Meta model trained. Accuracy on test set: {acc:.2%}")

    def predict_refined_score(self, features: Dict[str, float]) -> float:
        if not self.model:
            if self.verbose:
                print("⚠️ No model loaded. Returning raw confidence.")
            return features.get("confidence", 0.5)

        try:
            df = pd.DataFrame([features])
            score = self.model.predict_proba(df)[0][1]
            return float(score)
        except Exception as e:
            if self.verbose:
                print(f"❌ Error during prediction: {e}")
            return features.get("confidence", 0.5)
