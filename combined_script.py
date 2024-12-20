import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


churn = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

churn.head()

churn['MultipleLines'].unique()

churn['InternetService'].unique()

churn['OnlineSecurity'].unique()

churn['DeviceProtection'].unique()

churn['TechSupport'].unique()

churn['StreamingTV'].unique()

churn['StreamingMovies'].unique()

churn['Contract'].unique()

churn['PaymentMethod'].unique()

churn.info()

churn.drop('customerID', axis=1, inplace=True)
churn.drop('gender', axis=1, inplace=True)
churn.drop('TotalCharges',axis=1,inplace=True)

churn.replace('No',0, inplace=True)
churn.replace('Yes',1, inplace=True)
churn.replace('No phone service',2, inplace=True)
churn.replace('DSL',1, inplace=True)
churn.replace('Fiber optic',2, inplace=True)
churn.replace('No internet service',2, inplace=True)
churn.replace('Month-to-month',0, inplace=True)
churn.replace('One year',1, inplace=True)
churn.replace('Two year',2, inplace=True)
churn.replace('Electronic check',0, inplace=True)
churn.replace('Mailed check',1, inplace=True)
churn.replace('Bank transfer (automatic)',2, inplace=True)
churn.replace('Credit card (automatic)',3, inplace=True)

churn.dropna(axis=0, inplace=True)

churn.head()

churn.info()

X,y = churn.drop('Churn', axis=1),churn['Churn'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=87)


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(max_iter=7400)
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

from flask import Flask, request, jsonify
app = Flask(__name__)

churn.drop('Churn', axis=1).iloc[0].to_json(orient="index")

@app.get("/churn/sample")
def get_sample():
    return churn.drop('Churn', axis=1).iloc[0].to_json(orient="index")

@app.get("/churn/explain")
def get_explanation():
    explanation = """
    This dataset is used to predict customer churn. Here is an explanation of each field:
    
    - SeniorCitizen: Indicates if the customer is a senior citizen. 0 = No, 1 = Yes.
    - Partner: Indicates if the customer has a partner. 0 = No, 1 = Yes.
    - Dependents: Indicates if the customer has dependents. 0 = No, 1 = Yes.
    - tenure: The number of months the customer has stayed with the company.
    - PhoneService: Indicates if the customer has phone service. 0 = No, 1 = Yes, 2 = No phone service.
    - MultipleLines: Indicates if the customer has multiple lines. 0 = No, 1 = Yes, 2 = No phone service.
    - InternetService: Indicates the type of internet service. 0 = No, 1 = DSL, 2 = Fiber optic, 3 = No internet service.
    - OnlineSecurity: Indicates if the customer has online security. 0 = No, 1 = Yes, 2 = No internet service.
    - OnlineBackup: Indicates if the customer has online backup. 0 = No, 1 = Yes, 2 = No internet service.
    - DeviceProtection: Indicates if the customer has device protection. 0 = No, 1 = Yes, 2 = No internet service.
    - TechSupport: Indicates if the customer has tech support. 0 = No, 1 = Yes, 2 = No internet service.
    - StreamingTV: Indicates if the customer has streaming TV. 0 = No, 1 = Yes, 2 = No internet service.
    - StreamingMovies: Indicates if the customer has streaming movies. 0 = No, 1 = Yes, 2 = No internet service.
    - Contract: Indicates the contract type. 0 = Month-to-month, 1 = One year, 2 = Two year.
    - PaperlessBilling: Indicates if the customer has paperless billing. 0 = No, 1 = Yes.
    - PaymentMethod: Indicates the payment method. 0 = Electronic check, 1 = Mailed check, 2 = Bank transfer (automatic), 3 = Credit card (automatic).
    - MonthlyCharges: The amount charged to the customer monthly.
    """
    return explanation

@app.post("/churn/evaluate")
def evaluate_sample():
    if request.is_json:
        data = request.get_json()
        df = pd.DataFrame([data])
        result = logModel.predict(df)
        result_str = "Churn" if result[0] == 1 else "No Churn"
        return jsonify(result_str)
    return jsonify({"error": "Request must be JSON"})

if __name__ == '__main__':
    app.run()

import requests, json


"""Test the /churn/sample endpoint"""
response = requests.get("http://127.0.0.1:5000/churn/sample")
print(response.json())

"""Test the /churn/explain endpoint"""
response = requests.get("http://127.0.0.1:5000/churn/explain")
explanation = response.text
print("Explanation:", explanation)

"""Test the /churn/evaluate endpoint with different inputs"""
data1= {"SeniorCitizen": 0, "Partner": 1, "Dependents": 0, "tenure": 1, "PhoneService": 1, "MultipleLines": 0, "InternetService": 1, "OnlineSecurity": 0, "OnlineBackup": 1, "DeviceProtection": 0, "TechSupport": 0, "StreamingTV": 0, "StreamingMovies": 0, "Contract": 0, "PaperlessBilling": 1, "PaymentMethod": 0, "MonthlyCharges": 29.85}
data2= {"SeniorCitizen": 0, "Partner": 0, "Dependents": 0, "tenure": 34, "PhoneService": 1, "MultipleLines": 0, "InternetService": 1, "OnlineSecurity": 1, "OnlineBackup": 0, "DeviceProtection": 1, "TechSupport": 0, "StreamingTV": 0, "StreamingMovies": 0, "Contract": 1, "PaperlessBilling": 0, "PaymentMethod": 1, "MonthlyCharges": 56.95}
data3= {"SeniorCitizen": 0, "Partner": 0, "Dependents": 0, "tenure": 2, "PhoneService": 1, "MultipleLines": 0, "InternetService": 2, "OnlineSecurity": 1, "OnlineBackup": 1, "DeviceProtection": 0, "TechSupport": 0, "StreamingTV": 0, "StreamingMovies": 0, "Contract": 0, "PaperlessBilling": 1, "PaymentMethod": 1, "MonthlyCharges": 53.85}
data4= {"SeniorCitizen": 1, "Partner": 1, "Dependents": 1, "tenure": 45, "PhoneService": 1, "MultipleLines": 2, "InternetService": 1, "OnlineSecurity": 1, "OnlineBackup": 1, "DeviceProtection": 1, "TechSupport": 1, "StreamingTV": 1, "StreamingMovies": 2, "Contract": 1, "PaperlessBilling": 0, "PaymentMethod": 2, "MonthlyCharges": 42.30}
data5= {"SeniorCitizen": 0, "Partner": 0, "Dependents": 0, "tenure": 2, "PhoneService": 1, "MultipleLines": 0, "InternetService": 2, "OnlineSecurity": 0, "OnlineBackup": 0, "DeviceProtection": 0, "TechSupport": 0, "StreamingTV": 0, "StreamingMovies": 0, "Contract": 0, "PaperlessBilling": 1, "PaymentMethod": 0, "MonthlyCharges": 70.70}

response = requests.post("http://127.0.0.1:5000/churn/evaluate", json=data1)
result1 = response.json()
print(result1)

response = requests.post("http://127.0.0.1:5000/churn/evaluate", json=data2)
result2 = response.json()
print(result2)

response = requests.post("http://127.0.0.1:5000/churn/evaluate", json=data3)
result3 = response.json()
print(result3)

response = requests.post("http://127.0.0.1:5000/churn/evaluate", json=data4)
result4 = response.json()
print(result4)

response = requests.post("http://127.0.0.1:5000/churn/evaluate", json=data5)
result5 = response.json()
print(result5)
