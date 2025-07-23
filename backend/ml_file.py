import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

class MLTrader:
    def __init__(self, model_path='ml_model.pkl'):
        self.model_path = model_path
        self.model = self.load_model()

    def train_model(self, data):
        # Feature engineering
        data['returns'] = data['close'].pct_change()
        data['sma'] = data['close'].rolling(15).mean()
        data['lma'] = data['close'].rolling(50).mean()
        data.dropna(inplace=True)

        # Define features and target
        features = ['returns', 'sma', 'lma']
        target = 'signal'
        data[target] = (data['returns'] > 0).astype(int)  # 1 for up, 0 for down

        X = data[features]
        y = data[target]

        # Split data and train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate and save model
        y_pred = model.predict(X_test)
        print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
        joblib.dump(model, self.model_path)
        self.model = model

    def load_model(self):
        try:
            return joblib.load(self.model_path)
        except FileNotFoundError:
            return None

    def predict_signal(self, data):
        if self.model is None:
            raise Exception("Model not trained or loaded.")

        # Feature engineering
        data['returns'] = data['close'].pct_change()
        data['sma'] = data['close'].rolling(15).mean()
        data['lma'] = data['close'].rolling(50).mean()
        data.dropna(inplace=True)

        features = ['returns', 'sma', 'lma']
        return self.model.predict(data[features])
