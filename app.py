import streamlit as st
import pickle
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import soundfile as sf

# ─────────────────────────────────────────────
# 🎨 Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MotorSense AI",
    page_icon="⚙️",
    layout="wide"
)

# ─────────────────────────────────────────────
# 🎯 Load Model + Encoder
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        pipeline = pickle.load(f)

    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    return pipeline, le

pipeline, le = load_model()

# ─────────────────────────────────────────────
# ⚙️ Constants
# ─────────────────────────────────────────────
SAMPLE_RATE = 1000
N_MFCC = 30
DURATION = 3
MAX_LEN = 6

# ─────────────────────────────────────────────
# 🔊 Feature Extraction
# ─────────────────────────────────────────────
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    expected_len = SAMPLE_RATE * DURATION
    if len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)))

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=512
    )
    return mfcc, y


def pad_or_trim(mfcc, max_len):
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc


# ─────────────────────────────────────────────
# 🤖 Prediction Function
# ─────────────────────────────────────────────
def predict_with_confidence(file_path):
    mfcc, y = extract_mfcc(file_path)
    mfcc = pad_or_trim(mfcc, MAX_LEN)

    features = mfcc.flatten().reshape(1, -1)

    probs = pipeline.predict_proba(features)
    pred = np.argmax(probs)

    label = le.inverse_transform([pred])[0]
    confidence = float(np.max(probs))

    return label, confidence, y


# ─────────────────────────────────────────────
# 📊 Waveform Plot
# ─────────────────────────────────────────────
def plot_waveform(y):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(y)
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)


# ─────────────────────────────────────────────
# 🎨 UI Design
# ─────────────────────────────────────────────
st.markdown(
    """
    <h1 style='text-align: center; color: #00BFFF;'>
    ⚙️ Motor Condition Monitoring using Audio
    </h1>
    <p style='text-align: center;'>
    Upload motor sound and detect motor condition
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# Upload Section
uploaded_file = st.file_uploader(
    "📂 Upload Motor Audio (.wav)",
    type=["wav"]
)

if uploaded_file is not None:

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    st.success("✅ Audio Uploaded Successfully!")

    # Audio Player
    st.subheader("🔊 Audio Playback")
    st.audio(temp_path)

    # Prediction Button
    if st.button("🚀 Analyze Motor Sound"):

        with st.spinner("Analyzing..."):

            label, confidence, y = predict_with_confidence(temp_path)

        # ─────────────────────────────────────────────
        # 🎯 Result Section
        # ─────────────────────────────────────────────
        st.subheader("🎯 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Detected Condition", label)

        with col2:
            st.metric("Confidence", f"{confidence*100:.2f}%")

        # ─────────────────────────────────────────────
        # 📊 Waveform
        # ─────────────────────────────────────────────
        st.subheader("📊 Audio Waveform")
        plot_waveform(y)

        # ─────────────────────────────────────────────
        # 🎨 Confidence Bar
        # ─────────────────────────────────────────────
        st.progress(confidence)

        if confidence > 0.9:
            st.success("🟢 High Confidence Prediction")
        elif confidence > 0.7:
            st.warning("🟡 Moderate Confidence")
        else:
            st.error("🔴 Low Confidence")

else:
    st.info("👆 Upload a .wav file to start analysis")




#streamlit run app.py