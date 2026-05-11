# ⚙️ Motor Fault Detection using Audio

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56.0-FF4B4B?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 📋 Overview

**Motor Fault Detection using Audio** is a machine learning project that classifies motor conditions based on audio signals. The system detects three distinct motor states:
- **Engine 1 (Good)**: Healthy motor operation
- **Engine 2 (Broken)**: Motor with mechanical faults
- **Engine 3 (Heavy Load)**: Motor under heavy load conditions

Using MFCC (Mel-Frequency Cepstral Coefficients) feature extraction and a trained classifier, this system can reliably identify motor conditions with **high confidence predictions** through a user-friendly Streamlit web interface.

---

## ✨ Features

- 🎤 **Audio-Based Detection**: Analyzes motor sounds to predict operational status
- 🤖 **ML-Powered Predictions**: Uses pre-trained classification models
- 📊 **Confidence Scoring**: Returns prediction confidence for reliability assessment
- 🎨 **Interactive Web Interface**: Built with Streamlit for easy interaction
- 📈 **MFCC Feature Engineering**: Extracts acoustic features using librosa
- 🎯 **Multi-Class Classification**: Identifies 3 distinct motor conditions
- 📊 **Real-time Waveform Visualization**: Displays audio waveforms for analysis
- ⚡ **Fast Inference**: Quick predictions with cached model loading

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.8+ |
| **Web Framework** | Streamlit 1.56.0 |
| **ML Library** | scikit-learn 1.6.1 |
| **Audio Processing** | librosa 0.11.0 |
| **Audio I/O** | SoundFile 0.13.1 |
| **Data Science** | pandas, numpy |
| **Visualization** | matplotlib 3.10.8 |
| **Model Serialization** | pickle |
| **Gradient Boosting** | XGBoost 3.2.0 |

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/AdrshGoyal/Motor-fault-detection-using-Audio.git
cd Motor-fault-detection-using-Audio
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Running the Web Application

Launch the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

**Steps to use:**
1. Upload a WAV audio file (motor sound recording)
2. Click **"Analyze Motor Sound"** button
3. View results including:
   - Detected motor condition
   - Confidence percentage
   - Audio waveform visualization
   - Confidence indicator

### Running Predictions via Python Script

For batch predictions or integration:

```python
from main111 import predict_with_confidence

# Single prediction with confidence
audio_path = "path/to/motor_sound.wav"
label, confidence = predict_with_confidence(audio_path)

print(f"Motor Condition: {label}")
print(f"Confidence: {confidence * 100:.2f}%")
```

---

## 📁 Project Structure

```
Motor-fault-detection-using-Audio/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── app.py                            # Streamlit web application
├── main111.py                        # Inference script with API functions
├── model.pkl                         # Trained classification model
├── label_encoder.pkl                 # Label encoder for predictions
├── motor_mfcc_embeddings (1).csv     # Pre-computed MFCC features dataset
└── motor_Vibration_Fault_Detection (2).ipynb  # Jupyter notebook with analysis
```

### File Descriptions

| File | Purpose |
|------|---------|
| **app.py** | Streamlit-based web UI for interactive predictions |
| **main111.py** | Core prediction functions for batch processing |
| **model.pkl** | Pre-trained XGBoost classifier |
| **label_encoder.pkl** | Encodes/decodes class labels |
| **motor_mfcc_embeddings.csv** | Feature vectors for training/analysis |
| **notebook.ipynb** | Complete EDA, training pipeline, and evaluation |

---

## 🧠 Model Architecture & Algorithm

### Feature Extraction Pipeline

The system uses **MFCC (Mel-Frequency Cepstral Coefficients)** to convert raw audio into machine-readable features:

```
Raw Audio (.wav)
    ↓
Load with librosa (1000 Hz sample rate, 3 sec duration)
    ↓
Pad/Truncate to exactly 3000 samples
    ↓
Extract 30 MFCC coefficients (n_mfcc=30, n_fft=512)
    ↓
MFCC Matrix (30 × 6 time steps)
    ↓
Flatten to feature vector (180 features)
    ↓
Classification Model
    ↓
Predicted Label + Confidence Score
```

### Parameters

- **Sample Rate**: 1000 Hz (downsampled for efficiency)
- **Duration**: 3 seconds per audio clip
- **N_MFCC**: 30 coefficients
- **N_FFT**: 512 samples
- **Max Length**: 6 time steps (after padding/trimming)
- **Total Features**: 180 (30 × 6)

### Classification Model

- **Algorithm**: Gradient Boosting (XGBoost)
- **Training Data**: 507 audio samples across 3 classes
- **Output**: 3-class probability distribution
- **Prediction**: argmax(probabilities) for class label

### Model Performance

The model is trained to achieve high accuracy in distinguishing between:
- ✅ **Good Motor**: Normal operating conditions
- ⚠️ **Broken Motor**: Mechanical faults detected
- 🔧 **Heavy Load**: Motor under stress

---

## 📊 Dataset Information

### Dataset Composition

| Class | Samples | Type |
|-------|---------|------|
| engine1_good | 155 | Healthy motor sounds |
| engine2_broken | 174 | Faulty motor sounds |
| engine3_heavyload | 178 | Heavy load conditions |
| **Total** | **507** | **Audio files** |

### Audio Characteristics
- **Format**: WAV files
- **Duration**: 3+ seconds per file
- **Sample Rate**: Varies (normalized to 1000 Hz during processing)
- **Source**: IDMT-ISA-ELECTRIC-ENGINE dataset (motor sounds at different conditions)

### Data Preprocessing
1. Audio files loaded at 1000 Hz sampling rate
2. Padded/trimmed to exactly 3 seconds
3. MFCC features extracted for each file
4. Features standardized and stored in CSV format
5. Label encoded for classification

---

## 🔧 Configuration & Parameters

### Audio Processing Configuration

```python
SAMPLE_RATE = 1000      # Hz - downsampled from original
N_MFCC = 30             # Number of mel-frequency coefficients
DURATION = 3            # seconds - clip duration
MAX_LEN = 6             # time steps after feature extraction
N_FFT = 512             # FFT window size
```

### Streamlit UI Configuration

The application is configured with:
- **Page Title**: "MotorSense AI"
- **Icon**: ⚙️ (gear emoji)
- **Layout**: Wide layout for better visualization
- **Theme**: Default Streamlit theme

---

## 🌍 Environment Variables

No environment variables are required for basic operation. However, you can customize:

```bash
# Optional: Set Python path
export PYTHONPATH=$PYTHONPATH:./

# Optional: Set temporary directory for audio uploads
# export TMPDIR=/path/to/temp
```

---

## 📝 Jupyter Notebook

### `motor_Vibration_Fault_Detection (2).ipynb`

Comprehensive analysis notebook containing:

1. **Data Exploration**
   - Audio file discovery and inventory
   - Duration analysis across dataset
   - File structure exploration

2. **Feature Engineering**
   - MFCC extraction from raw audio
   - Feature matrix creation
   - Data normalization

3. **Model Training**
   - Train-test splitting
   - Model architecture selection
   - Hyperparameter tuning

4. **Evaluation**
   - Accuracy metrics
   - Confusion matrices
   - Cross-validation results

5. **Preprocessing Details**
   - Audio loading and resampling
   - Duration standardization
   - Feature scaling

**To run the notebook:**
```bash
jupyter notebook "motor_Vibration_Fault_Detection (2).ipynb"
```

---

## 💾 Pre-trained Models

### Model Files Included

1. **model.pkl** (252 KB)
   - Trained XGBoost classifier
   - Ready for inference
   - Expects 180 features (30 MFCC × 6 timesteps)

2. **label_encoder.pkl** (272 bytes)
   - Encoding: {0: 'engine1_good', 1: 'engine2_broken', 2: 'engine3_heavyload'}
   - Used for human-readable predictions

### Loading Models

```python
import pickle

# Load classifier
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
```

---

## 📈 Results & Evaluation Metrics

### Prediction Output

The model returns:
1. **Predicted Label**: One of ['engine1_good', 'engine2_broken', 'engine3_heavyload']
2. **Confidence Score**: Probability value (0.0 to 1.0)
3. **Audio Waveform**: Visual representation of input signal

### Confidence Interpretation

- 🟢 **High Confidence** (>90%): Very reliable prediction
- 🟡 **Moderate Confidence** (70-90%): Moderately reliable
- 🔴 **Low Confidence** (<70%): Low reliability, may need verification

### Sample Output

```
🎯 Prediction Result
├── Detected Condition: engine1_good
└── Confidence: 95.47%
    └── Status: 🟢 High Confidence Prediction
```

---

## 🔮 Future Improvements

- [ ] **Real-time Monitoring**: Live motor sound stream processing
- [ ] **Multi-frequency Analysis**: Support for various audio sample rates
- [ ] **Explainability**: SHAP values for prediction interpretability
- [ ] **Model Ensemble**: Combine multiple models for better accuracy
- [ ] **Anomaly Detection**: Identify unusual motor patterns beyond 3 classes
- [ ] **Transfer Learning**: Fine-tune pre-trained audio models
- [ ] **Data Augmentation**: Synthetic audio generation for robustness
- [ ] **Database Integration**: Store prediction history
- [ ] **Mobile Support**: React Native/Flutter mobile application
- [ ] **API Deployment**: REST API with Docker containerization
- [ ] **Edge Deployment**: TensorFlow Lite for edge devices
- [ ] **Multilingual UI**: Support for multiple languages

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Steps to Contribute

1. **Fork the Repository**
   ```bash
   git clone https://github.com/yourusername/Motor-fault-detection-using-Audio.git
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Add improvements, fix bugs, or enhance documentation
   - Follow PEP 8 style guidelines
   - Add comments and docstrings

4. **Commit Changes**
   ```bash
   git commit -m "Add: description of your feature"
   ```

5. **Push to Branch**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**
   - Describe changes clearly
   - Reference related issues

### Contribution Areas

- 🐛 **Bug Reports**: Report issues you find
- ✨ **Feature Requests**: Suggest new functionality
- 📝 **Documentation**: Improve README and comments
- 🧪 **Testing**: Add unit tests
- 🔧 **Performance**: Optimize code

---

## 📄 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

```
MIT License - Free to use, modify, and distribute with attribution
```

---

## 👤 Author & Contact

**Adrsh Goyal**

- 🔗 GitHub: [@AdrshGoyal](https://github.com/AdrshGoyal)
- 💼 Project: Motor Fault Detection using Audio
- 📅 Created: April 2026
- 🔄 Last Updated: May 11, 2026

---

## 📚 References & Resources

### Libraries Used
- [Librosa](https://librosa.org/) - Audio processing and MFCC extraction
- [scikit-learn](https://scikit-learn.org/) - Machine learning algorithms
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting classifier
- [Streamlit](https://streamlit.io/) - Web application framework
- [SoundFile](https://soundfile.readthedocs.io/) - Audio file I/O

### Concepts & Papers
- MFCC Features: [Popular audio feature extraction technique](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
- Condition Monitoring: Machine learning for predictive maintenance
- Audio Classification: Deep learning applications in acoustic analysis

### Dataset
- IDMT-ISA-ELECTRIC-ENGINE: Industrial motor sound dataset with labeled conditions

---

## ⭐ Show Your Support

If you find this project helpful:
- ⭐ **Star the repository**
- 🍴 **Fork for your own projects**
- 💬 **Provide feedback and suggestions**
- 📢 **Share with others interested in ML/Audio processing**

---

**Happy Motor Monitoring!** 🚀⚙️
