import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def test_model_accuracy():
    
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    model = joblib.load("iris_model.pkl")


    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    assert accuracy >= 0.9, f"Model accuracy is too low: {accuracy}"