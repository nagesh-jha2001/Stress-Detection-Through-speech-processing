from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

app = Flask(__name__, static_url_path='/static')

# Loading the trained model and the scaler
model = joblib.load('model1.pkl')

scaler = joblib.load('scaler1.pkl')

# Defining a function to extract features from the input audio file
def extract_features(file_path):
    X, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
    zcr = np.mean(librosa.feature.zero_crossing_rate(X, frame_length=2048, hop_length=512))
    f0 = np.nanmean(librosa.pyin(X, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')))
    mfcc = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=20).T, axis=0) 
    energy = np.mean(librosa.feature.rms(y=X))
    return np.concatenate((np.array([zcr, f0, energy]), mfcc))


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            new_feature = extract_features(file)
            new_feature_concat = np.concatenate((new_feature, np.zeros(24-len(new_feature))))
            new_feature = scaler.transform(new_feature_concat.reshape(1,-1))
            prediction = model.predict(new_feature.reshape(1, -1))[0]
            return render_template('result.html', prediction=prediction)
        else:
            return render_template('nhome.html', error='Please select an audio file.')
    else:
        return render_template('nhome.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/stre.html')
def stress_relief():
    return render_template('stre.html')


if __name__ == '__main__':
    app.run(debug=True)


