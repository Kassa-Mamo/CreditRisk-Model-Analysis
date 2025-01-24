from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

MODEL_PATH = r"C:\Users\user\Desktop\10 Academy- Machine-Learning\10 Academy W6\models\credit_risk_model.pkl"

def train_model(X_train, y_train):
    """
    Train a Random Forest model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance.
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report

def save_model(model):
    """
    Save the trained model to disk.
    """
    joblib.dump(model, MODEL_PATH)
