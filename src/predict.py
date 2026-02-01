import numpy as np
import joblib

# Load saved model
model = joblib.load('src/sonar_model.pkl')


def predict_sonar(input_data):
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    if prediction == 'R':
        return "The object is a rock"
    else:
        return "The object is a mine"


# Example usage
if __name__ == "__main__":
    input_data = (0.0210,0.0121,0.0203,0.1036,0.1675,0.0418,0.0723,0.0828,0.0494,
                  0.0686,0.1125,0.1741,0.2710,0.3087,0.3575,0.4998,0.6011,0.6470,
                  0.8067,0.9008,0.8906,0.9338,1.0000,0.9102,0.8496,0.7867,0.7688,
                  0.7718,0.6268,0.4301,0.2077,0.1198,0.1660,0.2618,0.3862,0.3958,
                  0.3248,0.2302,0.3250,0.4022,0.4344,0.4008,0.3370,0.2518,0.2101,
                  0.1181,0.1150,0.0550,0.0293,0.0183,0.0104,0.0117,0.0101,0.0061,
                  0.0031,0.0099,0.0080,0.0107,0.0161,0.0133)
    
    print(predict_sonar(input_data))
