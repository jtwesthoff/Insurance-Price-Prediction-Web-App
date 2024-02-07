import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import pickle


app = Flask(__name__)
model_path = "C:\\Users\\Joseph Westhoff\\AI2023\\Final Project\\model.pkl"

def init():
    global model
    model = pickle.load(open(model_path, "rb"))

def convert_to_float(value):
    try:
        return float(value)
    except ValueError:
        return value

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/error')
def get_error_page():
    return render_template('error.html')

@app.route('/predict', methods=['POST'])
def result():
    
    if request.method == 'POST':
        try:
            to_predict_list = request.form.to_dict()
            to_predict_list = list(to_predict_list.values())
            to_predict = np.array(to_predict_list).reshape(1, 6)
            columnNames = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
            df = pd.DataFrame(to_predict, columns=columnNames)
            df = df.astype({'age':int, 'bmi':float, 'children':int})
            prediction = model.predict(df)
            return render_template("predict.html", prediction = prediction)
            
        except Exception as e:
            return get_error_page()

if __name__ == "__main__":
    init()
    app.run(debug=True)