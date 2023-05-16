import numpy as np 
import librosa 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import KFold 
from keras.utils import np_utils 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Conv1D, MaxPooling1D 
from keras.callbacks import TensorBoard, ModelCheckpoint 
from keras.optimizers import Adam 
 
class FacialMovement: 
    def __init__(self, audio_data, labels): 
        self.audio_data = audio_data 
        self.labels = labels 
        self.label_encoder = LabelEncoder() 
        self.model = None 
 
    def augment_data(self, audio_data, sample_rate, n_augmentations=5): 
        augmented_data = [] 
        for audio in audio_data: 
            for _ in range(n_augmentations): 
                augmented_audio = librosa.effects.time_stretch(audio, np.random.uniform(0.8, 1.2)) 
                augmented_audio = librosa.effects.pitch_shift(augmented_audio, sample_rate, np.random.randint(-5, 5)) 
                augmented_data.append(augmented_audio) 
        return np.array(augmented_data) 
 
    def cross_validate_model(self, normalized_audio, one_hot_labels, n_splits=5): 
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42) 
        cv_scores = [] 
        for train_indices, test_indices in kfold.split(normalized_audio, one_hot_labels): 
            X_train, X_test = normalized_audio[train_indices], normalized_audio[test_indices] 
            y_train, y_test = one_hot_labels[train_indices], one_hot_labels[test_indices] 
            input_shape = (X_train.shape[1], 1) 
            self.build_model(input_shape) 
            self.train_model(X_train, y_train) 
            scores = self.evaluate_model(X_test, y_test) 
            cv_scores.append(scores[1]) 
        return cv_scores 
 
    def preprocess_data(self, sample_rate, n_augmentations): 
        self.label_encoder.fit(self.labels) 
        encoded_labels = self.label_encoder.transform(self.labels) 
        one_hot_labels = np_utils.to_categorical(encoded_labels) 
        augmented_audio_data = [] 
        for audio_path, label in zip(self.audio_data, self.labels): 
            audio, _ = librosa.load(audio_path, sr=sample_rate) 
            augmented_audio = self.augment_data([audio], sample_rate, n_augmentations) 
            augmented_audio_data.extend(augmented_audio) 
            one_hot_labels = np.vstack([one_hot_labels] * n_augmentations) 
        features = np.array([self.extract_features(x, sample_rate) for x in augmented_audio_data]) 
        normalized_audio = np.array([librosa.util.normalize(x) for x in features]) 
        return normalized_audio, one_hot_labels 
 
    def prepare_train_test_data(self, normalized_audio, one_hot_labels, test_size=0.2): 
        X_train, X_test, y_train, y_test = train_test_split(normalized_audio, one_hot_labels,  
                                                            test_size=test_size, random_state=42) 
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1) 
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1) 
        return X_train, X_test, y_train, y_test 
 
    def build_model(self, input_shape): 
        self.model = Sequential() 
        self.model.add(Conv1D(32, 2, activation='relu', input_shape=input_shape)) 
        self.model.add(MaxPooling1D(pool_size=2)) 
        self.model.add(Dropout(0.2)) 
        self.model.add(Bidirectional(LSTM(units=100, return_sequences=True))) 
        self.model.add(Dropout(0.2)) 
        self.model.add(Bidirectional(LSTM(units=100))) 
        self.model.add(Dense(units=len(self.label_encoder.classes_), activation='softmax')) 
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy']) 
        self.model.summary() 
 
    def train_model(self, X_train, y_train, batch_size=32, epochs=50, tensorboard_log_dir='logs'): 
        tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir, write_graph=True) 
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[tensorboard_callback]) 
 
    def evaluate_model(self, X_test, y_test): 
        scores = self.model.evaluate(X_test, y_test, verbose=1) 
        return scores 
 
    def predict(self, audio_data, sample_rate): 
        features = np.array([self.extract_features(x, sample_rate) for x in audio_data]) 
        normalized_audio = np.array([librosa.util.normalize(x) for x in features]) 
        normalized_audio = normalized_audio.reshape(normalized_audio.shape[0], normalized_audio.shape[1], 1) 
        predictions = self.model.predict(normalized_audio) 
        predicted_labels = self.label_encoder.inverse_transform(np.argmax(predictions, axis=1)) 
        return predicted_labels 
 
    def extract_features(self, audio, sample_rate, n_mfcc=20): 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc) 
        return np.mean(mfccs, axis=1) 
