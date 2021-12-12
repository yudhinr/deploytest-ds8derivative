from flask import Flask
from flask import render_template
from flask import jsonify
from flask import request

from datetime import datetime
from flask_ngrok import run_with_ngrok
import pandas as pd
import numpy as np
import pickle
# import scikit-learn

app = Flask(__name__)

@app.route("/")
def home():
  return render_template("index.html")


@app.route('/prediction', methods=['POST'])
def prediction():
  df = pd.DataFrame({
    'User ID':request.form.get('user_id'),
    'Gender':request.form.get('gender'),
    'Age':request.form.get('age'),
    'EstimatedSalary':request.form.get('estimatedsalary'),
  }, index=[0])

  preds = data_processing(df)
  
  if preds[0] == 0:
    result = 'Customer Not Converted'
  else:
    result='Customer Converted'  

  # print(data)
  return result

def data_processing(df):
  with open('scaler.sav', 'rb') as file:
    scaler = pickle.load(file)

  df = df.drop(['User ID'], axis = 1)
  df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
  df.iloc[:,1:3] = scaler.transform(df.iloc[:,1:3])
  print(df)

  with open('model_knn.pkl', 'rb') as file:
    model_knn = pickle.load(file)

  prediction_knn = model_knn.predict(df)

  return prediction_knn


if __name__ == '__main__':
  app.run()