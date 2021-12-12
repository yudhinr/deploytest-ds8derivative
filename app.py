from flask import Flask
from flask import jsonify
from flask import request

from datetime import datetime
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)

@app.route('/', methods = ['GET'])
def index():
  with open('scaler.sav', 'rb') as file:
    scaler = pickle.load(file)

  data = request.json
  df = pd.DataFrame(data, index = [0])

  df = df.drop(['User ID'], axis = 1)
  df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
  df.iloc[:,1:3] = scaler.transform(df.iloc[:,1:3])
  print(df)

  with open('model_knn.pkl', 'rb') as file:
    model_knn = pickle.load(file)

  with open('model_svm.pkl', 'rb') as file:
    model_svm = pickle.load(file)

  prediction_knn = model_knn.predict(df)
  prediction_svm = model_svm.predict(df)

  data['Prediction KNN'] = prediction_knn[0]
  data['Prediction SVM'] = prediction_svm[0]
  data['Datetime'] = datetime.now().strftime('%Y-%m-%d')

  with open('data_collection.txt', 'a') as file:
    file.write("%s\n" % data)

  return jsonify({'Status': 'Berhasil', 'Prediction KNN':str(prediction_knn[0]), 'Prediction SVM':str(prediction_svm[0])})

if __name__ == '__main__':
  app.run()