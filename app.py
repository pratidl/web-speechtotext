from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import asyncio
import edge_tts
import joblib
import librosa
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/output_audio"
MODEL_FOLDER = "model"

MODEL_PATH = os.path.join(MODEL_FOLDER, "model_asr_svm.pkl")
SCALER_PATH = os.path.join(MODEL_FOLDER, "scaler.pkl")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

model = None
scaler = None


def load_asr_model():
    global model, scaler

    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        print("Model ASR berhasil dimuat")
        print("Kelas kota:", list(model.classes_))

    except Exception as e:
        print("Gagal load model ASR:", e)
        model = None
        scaler = None


load_asr_model()


@app.route("/")
def landing():
    return render_template("landing.html")


async def generate_tts(text, voice, rate, filepath):
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate
    )
    await communicate.save(filepath)


@app.route("/text-to-speech", methods=["GET", "POST"])
def text_to_speech():
    audio_file = None

    if request.method == "POST":
        text = request.form.get("text")
        gender = request.form.get("gender")
        speed = request.form.get("speed")

        if text:
            filename = f"hasil_tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            filepath = os.path.join(OUTPUT_FOLDER, filename)

            if gender == "male":
                voice = "id-ID-ArdiNeural"
            else:
                voice = "id-ID-GadisNeural"

            if speed == "slow":
                rate = "-30%"
            elif speed == "fast":
                rate = "+30%"
            else:
                rate = "+0%"

            asyncio.run(generate_tts(text, voice, rate, filepath))

            audio_file = filename

    return render_template("text_to_speech.html", audio_file=audio_file)


def extract_features_from_file(audio_path, sr=16000, n_mfcc=13, target_duration=1.5):
    try:
        audio_data, sr = librosa.load(audio_path, sr=sr)

        audio_trimmed, _ = librosa.effects.trim(
            audio_data,
            top_db=25
        )

        target_samples = int(target_duration * sr)

        if len(audio_trimmed) < target_samples:
            audio_trimmed = np.pad(
                audio_trimmed,
                (0, target_samples - len(audio_trimmed))
            )

        audio_fixed = audio_trimmed[:target_samples]

        if np.max(np.abs(audio_fixed)) < 0.01:
            return None

        if np.max(np.abs(audio_fixed)) > 0:
            audio_fixed = audio_fixed / np.max(np.abs(audio_fixed))

        mfccs = librosa.feature.mfcc(
            y=audio_fixed,
            sr=sr,
            n_mfcc=n_mfcc
        )

        delta = librosa.feature.delta(mfccs)

        mfcc_mean = np.mean(mfccs.T, axis=0)
        mfcc_std = np.std(mfccs.T, axis=0)
        delta_mean = np.mean(delta.T, axis=0)
        delta_std = np.std(delta.T, axis=0)

        features = np.hstack(
            (
                mfcc_mean,
                mfcc_std,
                delta_mean,
                delta_std
            )
        )

        return features

    except Exception as e:
        print("Error ekstraksi fitur:", e)
        return None


def predict_city(audio_path):
    if model is None or scaler is None:
        return {
            "success": False,
            "message": "Model belum terbaca. Pastikan model_asr_svm.pkl dan scaler.pkl sudah ada di folder model."
        }

    features = extract_features_from_file(audio_path)

    if features is None:
        return {
            "success": False,
            "message": "Suara terlalu pelan atau audio gagal diproses. Coba rekam ulang dengan suara lebih jelas."
        }

    features_input = features.reshape(1, -1)
    features_scaled = scaler.transform(features_input)

    prediction = model.predict(features_scaled)[0]

    result = {
        "success": True,
        "prediction": prediction,
        "confidence": None,
        "top3": []
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities) * 100

        result["confidence"] = round(confidence, 2)

        sorted_index = np.argsort(probabilities)[::-1]

        for i in range(min(3, len(model.classes_))):
            index = sorted_index[i]

            result["top3"].append({
                "city": str(model.classes_[index]),
                "score": round(float(probabilities[index] * 100), 2)
            })

    return result


@app.route("/speech-to-text", methods=["GET", "POST"])
def speech_to_text():
    if request.method == "POST":
        audio = request.files.get("audio")

        if not audio:
            return jsonify({
                "success": False,
                "message": "Audio belum diterima."
            })

        filename = secure_filename(audio.filename)

        if filename == "":
            filename = "recorded_audio.wav"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_filename = f"{timestamp}_{filename}"
        audio_path = os.path.join(UPLOAD_FOLDER, saved_filename)

        audio.save(audio_path)

        result = predict_city(audio_path)

        return jsonify(result)

    return render_template("speech_to_text.html")


if __name__ == "__main__":
    app.run(debug=True)