# ============================================================
# IMDB SENTIMENT ANALYSIS – Comparing Deep Learning Models
# ============================================================
# Models used:
# - Bag-of-Words + Naive Bayes
# - Bag-of-Words + MLP (Multilayer Perceptron)
# - GRU (Gated Recurrent Unit)
# - BERT Transformer
# ============================================================

import random
import time
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, TextClassificationPipeline
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Bidirectional, Dense, Embedding
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.neural_network import MLPClassifier

# Set random seed for reproducibility
random.seed(25032005)

# ============================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================

# Load IMDB dataset from Keras
# It contains 25,000 training and 25,000 testing movie reviews (binary sentiment: 0=negative, 1=positive)
max_features = 100000   # Limit vocabulary size (top 100,000 words)
maxlen = 350            # Cut or pad reviews to a maximum length of 350 words

(x_data, y_data), _ = imdb.load_data(num_words=max_features)

# Split data into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=25032005, stratify=y_data
)

# Pad/truncate sequences so that all have the same length
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# Get mapping from words to indices and build reverse dictionary
word_index = imdb.get_word_index()
index_word = {index + 3: word for word, index in word_index.items()}
index_word[0], index_word[1], index_word[2] = "<PAD>", "<START>", "<UNK>"

def decode_review(encoded_review):
    """Decode numeric sequence back into a readable text review."""
    return " ".join(index_word.get(i, "<UNK>") for i in encoded_review)

# Convert numeric reviews to text form (for CountVectorizer and BERT)
X_train_text = [decode_review(review) for review in X_train]
X_test_text = [decode_review(review) for review in X_test]

# ============================================================
# 2. BAG-OF-WORDS (BoW) + NAIVE BAYES
# ============================================================

print("******** BOW **********")

# Convert text reviews into BoW vectors
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train_text)
X_test_bow = vectorizer.transform(X_test_text)

# Train Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_bow, y_train)

# Evaluate performance
y_pred = classifier.predict(X_test_bow)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Utility function: find index of first or N-th incorrect prediction
def first_index_not_equal(list1, list2, ommited_nr=1):
    _count = 0
    for i, (x, y) in enumerate(zip(list1, list2)):
        if (x != y) and (x == 1):
            _count += 1
            if (_count == ommited_nr):
                return i
    return -1

# Example of incorrect prediction inspection
z = first_index_not_equal(y_test, y_pred, 2)
print(z)
if z != -1:
    print(X_test_text[z])
    print("actual: " + ("positive" if y_test[z] else "negative"))
    print("predicted: " + ("positive" if y_pred[z] else "negative"))

# ============================================================
# 3. MLP CLASSIFIER ON BoW FEATURES
# ============================================================

print("******** MLP **********")

# Recreate BoW features with limited vocabulary and stopword removal
vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
X_train_bow = vectorizer.fit_transform(X_train_text)
X_test_bow = vectorizer.transform(X_test_text)

print("Building the MLP model...")

# Define a simple MLP architecture
mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),  # Two hidden layers with 128 and 64 neurons
    activation="relu",             # ReLU activation
    solver="adam",                 # Adam optimizer
    max_iter=10,                   # Number of epochs
    random_state=25032005,
    verbose=True
)

print("Training the MLP model...")
mlp_model.fit(X_train_bow, y_train)

print("Evaluating the model...")
y_pred = mlp_model.predict(X_test_bow)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# ============================================================
# 4. GRU (GATED RECURRENT UNIT) MODEL
# ============================================================

print("******** GRU **********")

# Estimate vocabulary size (add 1000 for unseen words)
vocab_size = max(max(x_data)) + 1000
max_words = 400       # Max length per review
embd_len = 64         # Embedding dimension size

# Pad sequences for GRU input
x_train = sequence.pad_sequences(X_train, maxlen=max_words)
x_test = sequence.pad_sequences(X_test, maxlen=max_words)

# Create validation split
x_valid, y_valid = x_train[:64], y_train[:64]
x_train_, y_train_ = x_train[64:], y_train[64:]

# Define GRU-based sequential model
gru_model = Sequential(name="GRU_Model")
gru_model.add(Embedding(vocab_size, embd_len))          # Embedding layer
gru_model.add(GRU(128, activation='tanh'))              # GRU recurrent layer
gru_model.add(Dense(1, activation='sigmoid'))           # Output layer (binary classification)

# Display model architecture summary
print(gru_model.summary())

# Compile the GRU model
gru_model.compile(
    loss="binary_crossentropy",
    optimizer='adam',
    metrics=['accuracy']
)

# Convert labels to NumPy arrays
y_train_ = np.array(y_train_)
y_valid = np.array(y_valid)

# Train GRU model
history2 = gru_model.fit(
    x_train_, y_train_,
    batch_size=64,
    epochs=10,
    verbose=1,
    validation_data=(x_valid, y_valid)
)

# Predict test data
y_pred = gru_model.predict(x_test)
y_pred = [1 if y > 0.5 else 0 for y in y_pred]

# Evaluate GRU performance
print()
print("GRU model Score---> ", gru_model.evaluate(x_test, np.array(y_test), verbose=1))
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# ============================================================
# 5. BERT TRANSFORMER-BASED SENTIMENT ANALYSIS
# ============================================================

print("******** BERT **********")

# Initialize tokenizer and pre-trained model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('gritli/bert-sentiment-analyses-imdb')
model.eval()

# Create text classification pipeline
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=None)

y_pred = []
start_time = time.time()

# Run prediction for each test sample (can take several minutes)
for i, text in enumerate(X_test_text):
    print(f"**** {i}//{len(X_test_text)}  -- {time.time() - start_time:.1f} seconds ****")
    text = " ".join(text.split()[:360])  # Limit to 360 tokens to prevent truncation
    predicted_class = int(pipe(text)[0][0]["label"].split("_")[-1])
    y_pred += [predicted_class]

# Evaluate BERT predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# ============================================================
# END OF PIPELINE
# ============================================================

# The script compares classical and deep learning sentiment classifiers:
# - Bag-of-Words (MultinomialNB, MLP)
# - Recurrent (GRU)
# - Transformer (BERT)
# Accuracy and interpretability vary by model complexity and feature representation.
