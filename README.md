# 🎬 IMDB Movie Review Sentiment Analysis

This project is an **end-to-end Deep Learning application** that predicts the sentiment (Positive/Negative) of movie reviews from the IMDB dataset using a Simple RNN (Recurrent Neural Network). It features a user-friendly web interface built with **Streamlit** for interactive sentiment classification.

---

## 🚀 Features

- **Deep Learning Model**: Uses TensorFlow and Keras with a Simple RNN for text classification.
- **IMDB Dataset**: Trained on 25,000 labeled movie reviews from the official IMDB dataset.
- **Custom Preprocessing**: Handles word-to-index mapping, padding, and text vectorization.
- **Early Stopping**: Prevents overfitting during model training.
- **Interactive Web App**: Classify your own movie reviews in real time with a Streamlit UI.
- **Word Count Validation**: Requires minimum 25 words for robust predictions.
- **Live Feedback**: Shows word count, sentiment label, and prediction score.
- **Model Persistence**: Trained model is saved and loaded for use in the web app.

---

## 🏗️ Project Structure

```
.
├── imdb_rnn_model.keras           # Trained Keras model
├── app.py / main.py / <your script>   # Main Streamlit app (see below for example)
├── README.md
├── requirements.txt
└── ... (other files)
```

---

## 🧑‍💻 How It Works

### 1. Data Loading & Preprocessing

- Loads IMDB reviews with TensorFlow's `imdb` dataset.
- Limits vocabulary size to 10,000 most common words.
- Pads each review to 500 tokens for uniformity.
- Provides utilities to decode/encode reviews for interpretability.

### 2. Model Architecture

- **Embedding Layer**: Converts word indices to dense vectors.
- **SimpleRNN Layer**: Processes sequences for sentiment features.
- **Dense Output**: Predicts a probability (sigmoid) for positive sentiment.

### 3. Training

- Trained with `EarlyStopping` based on validation loss.
- Achieves strong accuracy on validation set.
- Model is saved as `imdb_rnn_model.keras`.

### 4. Streamlit Web App

- Loads the trained model.
- Accepts user input (minimum 25 words required).
- Preprocesses input, encodes and pads the text.
- Predicts sentiment and displays the result with score.
- Provides clear, classify, and feedback features.

---

## 🖥️ Example Usage

### Training the Model (Python)

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Hyperparameters
VOCAB_SIZE = 10000
MAX_LEN = 500

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
X_train = sequence.pad_sequences(X_train, maxlen=MAX_LEN)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_LEN)

# Build the model
model = Sequential([
    Embedding(VOCAB_SIZE, 128),
    SimpleRNN(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.build(input_shape=(None, MAX_LEN))

# Train with EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    callbacks=[early_stopping]
)

# Save the trained model
model.save('imdb_rnn_model.keras')
```

---

### Running the Streamlit App

#### Example `app.py`:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Constants
VOCAB_SIZE = 10000
MAX_LEN = 500

# Load word index
word_index = imdb.get_word_index()
model = load_model('imdb_rnn_model.keras', compile=False)

def preprocess_text(text, vocab_size=VOCAB_SIZE):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    encoded_review = [i for i in encoded_review if i < vocab_size]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=MAX_LEN)
    return padded_review

for key in ["sentiment", "score", "review_input"]:
    if key not in st.session_state:
        st.session_state[key] = ""

def clear_input():
    st.session_state.review_input = ""
    st.session_state.sentiment = ""
    st.session_state.score = None

st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review with **at least 25 words** to classify it as Positive or Negative.")

st.text_area("Movie Review", key="review_input", height=150, placeholder="Type your movie review here...")
word_count = len(st.session_state.review_input.strip().split())
st.write(f"**Word count:** {word_count} (minimum 25 required)")

col1, col2 = st.columns(2)
with col1:
    if st.button("Classify"):
        review = st.session_state.review_input.strip()
        if word_count < 25:
            st.warning("🚫 Please enter at least 25 words to classify.")
        else:
            try:
                preprocessed_input = preprocess_text(review)
                prediction = model.predict(preprocessed_input)
                st.session_state.score = float(prediction[0][0])
                st.session_state.sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")

with col2:
    st.button("Clear", on_click=clear_input)

if st.session_state.sentiment:
    st.success(f"**Sentiment:** {st.session_state.sentiment}")
    st.write(f"**Prediction Score:** {st.session_state.score:.4f}")
elif not st.session_state.review_input.strip():
    st.info("✍️ Please enter a movie review.")
```

---

## 📦 Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/saksham3232/IMDB-Movie-Review-Sentiment-Analysis.git
   cd IMDB-Movie-Review-Sentiment-Analysis
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:
   ```
   tensorflow==2.16.1
   keras==3.10.0
   streamlit
   numpy
   ```

3. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

---

## 📝 Notes

- The model expects reviews with at least 25 words for meaningful prediction.
- The IMDB dataset is automatically downloaded by TensorFlow on first run.
- For best results, run the app in a Python environment with the specified versions of TensorFlow and Keras.

