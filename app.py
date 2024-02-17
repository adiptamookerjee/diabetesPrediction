import sys
import scipy as sc
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

#validating the library versions
print("\n---------------------Version Validation-------------------------")
print('Python: {}'.format(sys.version))
print('scipy: {}'.format(sc.__version__))
print('numpy: {}'.format(np.__version__))
print('pandas: {}'.format(pd.__version__))
print('sklearn: {}'.format(sk.__version__))
print("----------------------------------------------------------------\n")

#Loading the data
df = pd.read_csv('/home/adipta28/Workspace/KIIT/6th Semester/ML/Assignment1/diabetes.csv')

#top 10
result = df.head(10)
print("\n---------------------------------------------------TOP 10-------------------------------------------------------")
print(result)
print("----------------------------------------------------------------------------------------------------------------\n")

#bottom 10
result = df.tail(10)
print("\n\n---------------------------------------------------BOTTOM 10----------------------------------------------------")
print(result)
print("------------------------------------------------------------------------------------------------------------------\n\n")

#Data frame description
print("------------------Attribute--Null Count--Data Type---------------------")
print("Pregnancies\t",df['Pregnancies'].isnull().sum(),"\t",df['Pregnancies'].dtypes)
print("Glucose\t",df['Glucose'].isnull().sum(),"\t",df['Glucose'].dtypes)
print("BloodPressure\t",df['BloodPressure'].isnull().sum(),"\t",df['BloodPressure'].dtypes)
print("SkinThickness\t",df['SkinThickness'].isnull().sum(),"\t",df['SkinThickness'].dtypes)
print("Insulin\t",df['Insulin'].isnull().sum(),"\t",df['Insulin'].dtypes)
print("BMI\t",df['BMI'].isnull().sum(),"\t",df['BMI'].dtypes)
print("DiabetesPedigreeFunction\t",df['DiabetesPedigreeFunction'].isnull().sum(),"\t",df['DiabetesPedigreeFunction'].dtypes)
print("Age\t",df['Age'].isnull().sum(),"\t",df['Age'].dtypes)
print("-----------------------------------------------------------------------")

print("\n------------------------------------------------------Data Frame Description----------------------------------------------------------------")
print(df.describe())
print("----------------------------------------------------------------------------------------------------------------------------------------------\n")

#creating histogram image
df.hist(figsize=(40,20))
plt.savefig('histogram.png')

X = df.drop('Outcome',axis=1)
y = df['Outcome']
#Normalising the data
scaler = MinMaxScaler()
X_normal = scaler.fit_transform(X)
df_normalised=pd.DataFrame(X_normal,columns=X.columns)
#Normalizing dataset
print('\n-------------------------------------------------Normalised Data------------------------------------------------')
print(X_normal)
print('----------------------------------------------------------------------------------------------------------------')

#Splitting the dataset in 80:20
X_train, X_test, y_train, y_test = train_test_split(df_normalised, y, test_size=0.2, random_state=42)

#Importing the model and fitting the data
model = LogisticRegression()
model.fit(X_train,y_train)

#Printing Different metrics
print("\nAccuracy:", model.score(X_test, y_test))


# Predicting on test set
y_pred = model.predict(X_test)
# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Print confusion matrix
print("\nConfusion Matrix\n",conf_matrix)


# Importing classification report
from sklearn.metrics import classification_report

# Generate classification report
class_report = classification_report(y_test, y_pred)
# Print classification report
print("\n----------------Classification Report----------------")
print(class_report)
print("-----------------------------------------------------\n")

# New input data and its prediction
new_inputs = pd.read_csv('/home/adipta28/Workspace/KIIT/6th Semester/ML/Assignment1/testData.csv')

# Normalizing new input data
new_inputs_normalized = scaler.transform(new_inputs)

# Predicting on new input data
predictions = model.predict(new_inputs_normalized)

# Print predictions
print("\n---------------Predictions for New Input Data----------------")
for i, pred in enumerate(predictions):
    print(f"Patient {i+1}: {'Diabetic' if pred == 1 else 'Non-Diabetic'}")
print("------------------------------------------------------------\n")



