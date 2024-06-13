import json
import pickle
import numpy as np
import pandas as pd
# from tensorflow.keras.models import load_model

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

model = pickle.load(open('airline_passenger_satisfaction.pkl', 'rb'))

# model = load_model('airline_passenger_satisfaction.h5')

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
    # features = [[sepal_length, sepal_width, petal_length, petal_width]]

    features = [[Online_boarding, Inflight_wifi_service, Type_of_Travel, Flight_Class,
                Inflight_entertainment, Seat_comfort, Leg_room_service]]

    # prediction = model.predict(features)
    # predicted_class = prediction[0]

    prediction = model.predict(features)
    # predicted_class = prediction[0]

    # Map predicted class index to class name (if applicable)
    # Replace class_index_to_name dictionary with your own mapping if needed
    class_index_to_name = {'neutral or dissatisfied': 0, 'satisfied': 1}

    if prediction == 0:
        predicted_class_name = 'Dissatisfied'
    elif prediction == 1:
        predicted_class_name = 'Satisfied'
    else:
        predicted_class_name = 'Unknown'

    # predicted_class_name = class_index_to_name.get(prediction, 'Unknown')

    # Return prediction result

    return render_template('index.html', prediction_text=predicted_class_name)

    # return jsonify({"predicted_class": predicted_class_name})


if __name__ == "__main__":
    app.run(debug=True)
