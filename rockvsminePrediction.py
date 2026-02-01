import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.metrics import  classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler



#loading the dataset to a pandas Dataframe
sonar_data = pd.read_csv('/content/sonar.csv',header = None)

sonar_data.head()

# Number of rows and columns
sonar_data.shape

sonar_data.describe()  #desribe statistical measures of the data.

sonar_data[60].value_counts()

sonar_data.groupby(60).mean()

# seprating data and labels:
X = sonar_data.drop(columns= 60 , axis = 1)
Y = sonar_data[60]

print(X)
print(Y)

X_train, X_test,  Y_train ,Y_test  = train_test_split(X,Y, test_size = 0.1 , stratify = Y, random_state = 1)

print(X.shape, X_train.shape, X_test.shape)

print(X_train)
print(Y_train)

model = LogisticRegression()


# training the logistic regression model with training data
model.fit(X_train, Y_train)

# Acccuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction , Y_train)

print("Accuracy on training data :" , training_data_accuracy)

# Acccuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction , Y_test)

print("Accuracy on test data :" , test_data_accuracy)

#input_data = (0.0286,0.0453,0.0277,0.0174,0.0384,0.0990,0.1201,0.1833,0.2105,0.3039,0.2988,0.4250,0.6343,0.8198,1.0000,0.9988,0.9508,0.9025,0.7234,0.5122,0.2074,0.3985,0.5890,0.2872,0.2043,0.5782,0.5389,0.3750,0.3411,0.5067,0.5580,0.4778,0.3299,0.2198,0.1407,0.2856,0.3807,0.4158,0.4054,0.3296,0.2707,0.2650,0.0723,0.1238,0.1192,0.1089,0.0623,0.0494,0.0264,0.0081,0.0104,0.0045,0.0014,0.0038,0.0013,0.0089,0.0057,0.0027,0.0051,0.0062)
#input_data = (0.0201,0.0423,0.0554,0.0783,0.0620,0.0871,0.1201,0.2707,0.1206,0.0279,0.2251,0.2615,0.1770,0.3709,0.4533,0.5553,0.4616,0.3797,0.3450,0.2665,0.2395,0.1127,0.2556,0.5169,0.3779,0.4082,0.5353,0.5116,0.4544,0.4258,0.3869,0.3939,0.4661,0.3974,0.2194,0.1816,0.1023,0.2108,0.3253,0.3697,0.2912,0.3010,0.2563,0.1927,0.2062,0.1751,0.0841,0.1035,0.0641,0.0153,0.0081,0.0191,0.0182,0.0160,0.0290,0.0090,0.0242,0.0224,0.0190,0.0096)
input_data = (0.0210,0.0121,0.0203,0.1036,0.1675,0.0418,0.0723,0.0828,0.0494,0.0686,0.1125,0.1741,0.2710,0.3087,0.3575,0.4998,0.6011,0.6470,0.8067,0.9008,0.8906,0.9338,1.0000,0.9102,0.8496,0.7867,0.7688,0.7718,0.6268,0.4301,0.2077,0.1198,0.1660,0.2618,0.3862,0.3958,0.3248,0.2302,0.3250,0.4022,0.4344,0.4008,0.3370,0.2518,0.2101,0.1181,0.1150,0.0550,0.0293,0.0183,0.0104,0.0117,0.0101,0.0061,0.0031,0.0099,0.0080,0.0107,0.0161,0.0133)
# Changing the input_data to a numpy array

input_data_as_numpy_array = np.asarray(input_data)

# Reshape numpy array as we are predicting for one instance

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 'R'):
  print("The object is a rock")
else:
    print("The object is mine")

