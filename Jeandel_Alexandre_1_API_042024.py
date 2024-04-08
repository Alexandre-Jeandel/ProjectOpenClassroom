# Import packages
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


# Global variables definition

'''
data_folder = "C:/Data_science/Projet_7_Git/data/cleaned/"                                 # local API
model_folder = "C:/Data_science/Projet_7_Git/models/"                                      # local API
data_name = 'data_prod.csv'                                       # local API
'''

#'''
data_folder = '/home/AlexandreJeandel/'         # online API
model_folder = '/home/AlexandreJeandel/'              # online API
data_name = 'data_prod.csv'                                         # online API
#'''

model_name = 'model.pkl'
sep='\t'

# Scaler instance
scaler = StandardScaler()

# Data and model loading
data = pd.read_csv(data_folder + data_name, sep=sep)
data_scaled = data.copy()

# Convert string representations of boolean values to actual booleans
boolean_strings = ['True', 'False']
for col in data_scaled.columns:
    if data_scaled[col].dtype == object and \
       all(item in boolean_strings for item in data_scaled[col].unique()):
        data_scaled[col] = data_scaled[col].map({'True': True, 'False': False})

# Iterate over each column and check for boolean data type
for col in data_scaled.columns:
    if data_scaled[col].dtype == bool:
        # Convert boolean column to integer (True to 1 and False to 0)
        data_scaled[col] = data_scaled[col].astype(int)

for col in data_scaled.columns:
    data_scaled[col] = pd.to_numeric(data_scaled[col], errors='coerce')


data_scaled.iloc[:,1:] = scaler.fit_transform(data_scaled.iloc[:,1:])

pipe = pickle.load(open(model_folder + model_name , 'rb'))

# API welcome function
@app.route("/")
def welcome():
    res = "Hello world! Welcome to the Default Predictor API!"
    return jsonify(res)

# Function returning all client IDs
@app.route("/client_list", methods=["GET"])
def load_client_id_list():
    id_list = data["SK_ID_CURR"].tolist()
    return jsonify(id_list)

# Function returning personal informaton of a given client (age, annuity amount, credit amount, total income amount)
@app.route("/client", methods=["GET"])
def load_client():

    client_id = int(request.args.get("id"))
    client = data[data["SK_ID_CURR"] == int(client_id)]

    if(client.size>0):

        DAYS_BIRTH = client['DAYS_BIRTH']
        AMT_INCOME_TOTAL = client['AMT_INCOME_TOTAL']
        AMT_CREDIT = client['AMT_CREDIT']
        AMT_ANNUITY = client['AMT_ANNUITY']

        return jsonify(DAYS_BIRTH=float(DAYS_BIRTH), AMT_INCOME_TOTAL=float(AMT_INCOME_TOTAL), AMT_CREDIT=float(AMT_CREDIT), AMT_ANNUITY=float(AMT_ANNUITY))

# Function returning, for a given informaton, the values of all clients (age, annuity amount, credit amount, total income amount)
@app.route("/data", methods=["GET"])
def load_data():
    col = request.args.get("col")
    data_col = data[col]
    data_list = data_col.tolist()
    return jsonify(data_list)

# Function returning, for a given client, the default probabiliy (in terms of percentage of default/no default)
@app.route("/predict_default", methods=["GET"])
def predict_default():

    id_client = int(request.args.get("id_client"))

    client = data_scaled.loc[data_scaled["SK_ID_CURR"] == id_client]
    client = client.iloc[:,1:]
    if(client.shape[0]==1):
        X = client.to_numpy()
        proba = pipe.predict_proba(X)[0]
        proba_0 = proba[0]  # No default
        proba_1 = proba[1]  # Default

        res = dict({'proba_0':proba_0, 'proba_1':proba_1})

    return jsonify(res)

if __name__ == "__main__":
    #app.run(host="localhost", port="5000", debug=True)        # local API
    app.run(debug=True)                                         # online API