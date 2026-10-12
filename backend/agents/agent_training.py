import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from backend.ml_engine.feature_extractor import extract_features
from backend.agents.generic_agent import GenericAgent  # or your agent class

def prepare_training_data(ohlcv: pd.DataFrame, labeled_actions: pd.Series) -> pd.DataFrame:
    """
    Extract features from OHLCV and align with labeled actions.

    Args:
        ohlcv (pd.DataFrame): Raw OHLCV data with datetime index.
        labeled_actions (pd.Series): Series indexed same as ohlcv, values are 'buy', 'sell', 'hold'.

    Returns:
        pd.DataFrame: Features + target column "action".
    """
    features = extract_features(ohlcv)

    # Align features and labels
    data = features.copy()
    data['action'] = labeled_actions

    # Drop rows where label is missing
    data = data.dropna(subset=['action'])
    return data

def train_agent_model(symbol: str, training_data: pd.DataFrame, model_dir="backend/agents/models"):
    """
    Train and save an ML model for an agent symbol.

    Args:
        symbol (str): Trading symbol like "BTCUSDT"
        training_data (pd.DataFrame): Dataframe with features + "action" column.
        model_dir (str): Directory to save models.
    """
    features = training_data.drop(columns=["action"])
    labels = training_data["action"]

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{symbol.lower()}_model.pkl")
    joblib.dump(model, model_path)

    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print(f"Trained model for {symbol}")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Model saved to: {model_path}")

    return model_path

if __name__ == "__main__":
    # Example usage:

    # 1. Load your OHLCV data and labels
    # Here you should load historical data & your known buy/sell/hold signals aligned to dates
    ohlcv = pd.read_csv("data/BTCUSDT_1h.csv", index_col=0, parse_dates=True)
    # labeled_actions should be a Series indexed by the same datetime as OHLCV
    # Example dummy: hold everywhere
    labeled_actions = pd.Series(["hold"] * len(ohlcv), index=ohlcv.index)

    # TODO: Replace with your labeled action data from backtest or manual labeling

    # 2. Prepare training dataset
    training_data = prepare_training_data(ohlcv, labeled_actions)

    # 3. Train and save model
    model_path = train_agent_model("BTCUSDT", training_data)
