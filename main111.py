import pickle
import librosa
import numpy as np

# Load trained pipeline
with open('model.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Load label encoder (IMPORTANT)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

SAMPLE_RATE = 1000
N_MFCC = 30
DURATION = 3      # 🔥 updated
MAX_LEN = 6       # 🔥 updated

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    # pad if shorter than 3 sec
    expected_len = SAMPLE_RATE * DURATION
    if len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)))

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=512   # 🔥 important fix
    )

    return mfcc


def pad_or_trim(mfcc, max_len):
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc



def predict_audio(file_path):

    # 1️⃣ Extract MFCC
    mfcc = extract_mfcc(file_path)

    # 2️⃣ Pad / Trim
    mfcc = pad_or_trim(mfcc, MAX_LEN)

    # 3️⃣ Flatten
    features = mfcc.flatten().reshape(1, -1)

    # 4️⃣ Predict
    pred = pipeline.predict(features)

    return pred[0]

def predict_label(file_path):
    pred = predict_audio(file_path)
    label = le.inverse_transform([pred])[0]
    return label


def predict_with_confidence(file_path):
    mfcc = extract_mfcc(file_path)
    mfcc = pad_or_trim(mfcc, MAX_LEN)
    features = mfcc.flatten().reshape(1, -1)

    probs = pipeline.predict_proba(features)
    pred = np.argmax(probs)

    label = le.inverse_transform([pred])[0]
    confidence = np.max(probs)

    return label, confidence

audio_path = "/content/drive/MyDrive/motor sound/IDMT-ISA-ELECTRIC-ENGINE/test/engine3_heavyload/atmo_low.wav"

result = predict_with_confidence(audio_path)

print("🎯 Prediction:", result)