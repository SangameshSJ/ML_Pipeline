import pickle
import numpy as np

# Load the model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

def test_model_prediction():
    sample_input = np.array([0.5, -1.2, 3.1, 0.2, -0.7]).reshape(1, -1)
    prediction = model.predict(sample_input)
    assert prediction in [0, 1], "Prediction should be either 0 or 1"