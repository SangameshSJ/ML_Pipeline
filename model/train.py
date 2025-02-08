import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=100, n_features=5, random_state=42)

# Train a simple logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Save the model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")