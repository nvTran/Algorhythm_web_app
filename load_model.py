# Serve model as a flask application
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import pickle
import numpy as np
from flask import Flask, request
from flask_cors import CORS



model = None
app = Flask(__name__)

CORS(app)

def load_model():
    global model
    # model variable refers to the global variable
    with open('finalized_model.pkl', 'rb') as f:
        model = pickle.load(f)
    global tv


@app.route('/', methods=['GET'])
def classify():
    # if request.method == 'GET':
    return render_template('home_page.html')




@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    print(request)
    if request.method == 'POST':
        data = request.get_json()  # Get data posted as a json
        from data_process import process_data
        from sklearn.ensemble import RandomForestClassifier





        # with open('new_finalized_model.pkl', 'rb') as f:
        #     model = pickle.load(f)
        with open('tv.pkl', 'rb') as f:
            tv = pickle.load(f)


        processed_data = tv.transform(str(data['lyrics']))
        prediction = model.predict(processed_data)  # runs globally loaded model on the data
    
        
        if prediction.tolist() == [0]:
            print('Male')
            return jsonify({"message": "Male"}), 201
        if prediction.tolist() == [1]:
            print('Female')
            return jsonify({"message": "Female"}), 201
    return jsonify({"message": "Error"}), 400
    


if __name__ == '__main__':
    # load_model()  # load model at the beginning once only
    app.run(port=5000, debug=True)