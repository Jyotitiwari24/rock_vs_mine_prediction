# This file makes `src` a Python package

# Optional: import main modules so you can use them easily
from .model import model, X_train, X_test, Y_train, Y_test
from .predict import predict_sonar

# You can define package-level metadata
__version__ = "1.0.0"
__author__ = "Your Name"


# Optional: small helper function
def hello():
    print("Sonar Classifier Package Ready!")
