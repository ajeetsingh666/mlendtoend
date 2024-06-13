import json
import pickle
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

model = pickle.load(open('airline_passenger_satisfaction.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("index.html")


@app.route('/predict', methods=["POST"])
def predict_api():
    Online_boarding = float(request.form.get('Online_boarding'))
    Inflight_wifi_service = float(request.form.get('Inflight_wifi_service'))
    Type_of_Travel = float(request.form.get('Type_of_Travel'))
    Flight_Class = float(request.form.get('Flight_Class'))
    Inflight_entertainment = float(request.form.get('Inflight_entertainment'))
    Seat_comfort = float(request.form.get('Seat_comfort'))
    Leg_room_service = float(request.form.get('Leg_room_service'))

    # Make prediction
    features = [[Online_boarding, Inflight_wifi_service, Type_of_Travel, Flight_Class,
                Inflight_entertainment, Seat_comfort, Leg_room_service]]


    prediction = model.predict(features)

    # Replace class_index_to_name dictionary with your own mapping if needed
    # class_index_to_name = {'neutral or dissatisfied': 0, 'satisfied': 1}

    if prediction == 0:
        predicted_class_name = 'Dissatisfied'
    elif prediction == 1:
        predicted_class_name = 'Satisfied'
    else:
        predicted_class_name = 'Unknown'

    return render_template('index.html', prediction_text=predicted_class_name)

if __name__ == "__main__":
    app.run(debug=True)
