import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Softmax
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import librosa
from scripts.preprocessing import load_data, preprocess_data, augment_data
from scripts.acoustic_sensing import AcousticSensing
from transformers import TFAutoModel, AutoTokenizer
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping
from jiwer import wer
 def create_model(input_shape, output_classes):
    # Load pre-trained transformer model
    transformer_model = TFAutoModel.from_pretrained("tbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("tbert-base-uncased")
     # Create input layers
    input_ids = Input(shape=input_shape, dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=input_shape, dtype=tf.int32, name="attention_mask")
     # Apply transformer model to input sequence
    transformer_output = transformer_model({'input_ids': input_ids, 'attention_mask': attention_mask})[0]
    x = Bidirectional(LSTM(128, return_sequences=True))(transformer_output)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(output_classes)(x)
    output = Softmax()(x)
     # Create and compile model
    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    optimizer = AdamW(learning_rate=2e-5, epsilon=1e-8)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_accuracy'])
    return model
 def train_model(data_path, model_output_path, batch_size=16, epochs=10, test_split=0.2, validation_split=0.1):
    # Load and preprocess data
    data = load_data(data_path)
    X_data, y_data = preprocess_data(data)
     # Augment data
    X_data, y_data = augment_data(X_data, y_data)
     # Encode labels to categorical
    le = LabelEncoder()
    y_data = le.fit_transform(y_data)
    y_data = to_categorical(y_data)
     # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_split, random_state=42)
     # Tokenize sequences
    tokenizer = AutoTokenizer.from_pretrained("tbert-base-uncased")
    X_train = tokenizer(X_train.tolist(), padding=True, truncation=True, max_length=512, return_tensors="tf")
    X_test = tokenizer(X_test.tolist(), padding=True, truncation=True, max_length=512, return_tensors="tf")
     # Create model
    input_shape = X_train['input_ids'].shape[1:]
    output_classes = y_train.shape[1]
    model = create_model(input_shape=input_shape, output_classes=output_classes)
     # Define callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
     # Train model
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=[early_stop])
     # Evaluate model
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    test_loss, test_accuracy, test_sparse_categorical_accuracy = model.evaluate(X_test, y_test)
     # Calculate word error rate
    y_pred_transcript = le.inverse_transform(y_pred)
    y_test_transcript = le.inverse_transform(y_test)
    wer_score = wer(y_test_transcript, y_pred_transcript)
     # Save model
    model.save(os.path.join(model_output_path, "echospeech_model.h5"))
     # Save label encoder
    with open(os.path.join(model_output_path, "label_encoder.npy"), 'wb') as f:
        np.save(f, le.classes_)
     return history, test_loss, test_accuracy, test_sparse_categorical_accuracy, wer_score
 if __name__ == "__main__":
    data_path = os.path.join("data", "audio")
    model_output_path = os.path.join("models", "audio")
     history, test_loss, test_accuracy, test_sparse_categorical_accuracy, wer_score = train_model(data_path, model_output_path)
     print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")
    print(f"Test sparse categorical accuracy: {test_sparse_categorical_accuracy:.4f}")
    print(f"Word error rate: {wer_score:.4f}")
