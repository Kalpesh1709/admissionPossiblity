from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Enabling CORS

# Load the model once at startup to avoid reloading on every request
MODEL_FILENAME = 'admission_model.pkl'
try:
    with open(MODEL_FILENAME, 'rb') as model_file:
        model = pickle.load(model_file)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/', methods=['GET'])
def home_page():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Read input data from form
            gre_score = float(request.form['gre_score'])
            toefl_score = float(request.form['toefl_score'])
            university_rating = float(request.form['university_rating'])
            sop = float(request.form['sop'])
            lor = float(request.form['lor'])
            cgpa = float(request.form['cgpa'])
            research = 1 if request.form['research'].lower() == 'yes' else 0
            
            # Ensure the model is loaded
            if model is None:
                return jsonify({"error": "Model could not be loaded"})
            
            # Make prediction
            input_data = np.array([[gre_score, toefl_score, university_rating, sop, lor, cgpa, research]])
            prediction = model.predict(input_data)
            predicted_value = round(100 * prediction[0])
            
            return render_template('results.html', prediction=predicted_value)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({"error": "Something went wrong during prediction"})

if __name__ == "__main__":
    app.run(debug=True)