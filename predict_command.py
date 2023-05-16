import os
import sys
import numpy as np
import argparse
from keras.models import load_model, Model
from keras.layers import Input, concatenate, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from preprocessing import preprocess_data, augment_data
from fmri import load_fmri_data
from train_fmri import train_fmri_model
from predict_fmri import predict_fmri_command
 def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/', help='Directory containing the data files.')
    parser.add_argument('--model_name', type=str, default='AlterEcho_silent_speech_model.h5', help='Name of the trained model to be loaded.')
    parser.add_argument('--output_dir', type=str, default='output/', help='Directory to save the results.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training.')
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs to wait before early stopping.')
    parser.add_argument('--fmri_epochs', type=int, default=10, help='Number of epochs for training fMRI model.')
    parser.add_argument('--use_augmentation', type=bool, default=True, help='Whether to use data augmentation or not.')
    return parser.parse_args(argv)
 def create_cnn_model(input_shape):
    # Transfer Learning with VGG16
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
     # Freezing base layers
    for layer in base_model.layers:
        layer.trainable = False
     # Modifying top layers
    x = base_model.output
    x = Conv2D(256, (3,3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(10, activation='softmax')(x)
     # Creating model
    model = Model(inputs=base_model.input, outputs=predictions)
    return model
 def main(args):
    # Setting up directories and creating output directory
    data_dir = args.data_dir
    model_name = args.model_name
    output_dir = args.output_dir
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr
    patience = args.patience
    fmri_epochs = args.fmri_epochs
    use_augmentation = args.use_augmentation
     if not os.path.exists(output_dir):
        os.makedirs(output_dir)
     # Load and preprocess data
    data, labels = preprocess_data(data_dir)
    num_classes = len(set(labels))
    input_shape = data.shape[1:]
     # Splitting data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
     # Data Augmentation
    if use_augmentation:
        train_data, train_labels = augment_data(train_data, train_labels)
     # Creating CNN model
    model = create_cnn_model(input_shape)
     # Compiling model
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
     # Callbacks
    earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=patience)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
    checkpoint = ModelCheckpoint(model_name, monitor='val_loss', mode='min', save_best_only=True)
     # Training model
    history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=epochs, batch_size=batch_size, callbacks=[earlystop, reduce_lr, checkpoint])
     # Load fMRI data
    fmri_data = load_fmri_data(data_dir)
     # Train fMRI model
    fmri_model = train_fmri_model(fmri_data, labels, num_classes, epochs=fmri_epochs)
     # Predict the command using the combined deep learning and fMRI model
    command_prediction = predict_fmri_command(model, fmri_model, data)
     # Evaluate the results
    accuracy = accuracy_score(labels, command_prediction)
    print(f"Accuracy: {accuracy}")
     # Save the results to output directory
    np.savetxt(os.path.join(output_dir, 'predicted_commands.txt'), command_prediction, fmt='%d')
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
