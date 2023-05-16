import numpy as np  
import pandas as pd  
import os  
import time  
from sklearn.metrics import accuracy_score, confusion_matrix  
  
def evaluate_fmri(pred_file_path, true_file_path):  
    """  
    Evaluate fMRI prediction using the preprocessed data and the true labels.  
  
    Parameters:  
    pred_file_path (str): Path to the predicted labels file.  
    true_file_path (str): Path to the true labels file.  
  
    Returns:  
    accuracy (float): Accuracy of the prediction.  
    confusion_matrix (array): Confusion matrix of the prediction.  
    """  
  
    # Load predicted and true labels  
    pred_labels = np.loadtxt(pred_file_path)  
    true_labels = np.loadtxt(true_file_path)  
  
    # Compute accuracy and confusion matrix  
    accuracy = accuracy_score(true_labels, pred_labels)  
    confusion_matrix = confusion_matrix(true_labels, pred_labels)  
  
    return accuracy, confusion_matrix  
  
def evaluate_eeg(pred_file_path, true_file_path):  
    """  
    Evaluate EEG prediction using the preprocessed data and the true labels.  
  
    Parameters:  
    pred_file_path (str): Path to the predicted labels file.  
    true_file_path (str): Path to the true labels file.  
  
    Returns:  
    accuracy (float): Accuracy of the prediction.  
    confusion_matrix (array): Confusion matrix of the prediction.  
    """  
  
    # Load predicted and true labels  
    pred_labels = np.loadtxt(pred_file_path)  
    true_labels = np.loadtxt(true_file_path)  
  
    # Compute accuracy and confusion matrix  
    accuracy = accuracy_score(true_labels, pred_labels)  
    confusion_matrix = confusion_matrix(true_labels, pred_labels)  
  
    return accuracy, confusion_matrix  
  
def evaluate_command(pred_file_path, true_file_path):  
    """  
    Evaluate command prediction using the preprocessed data and the true labels.  
  
    Parameters:  
    pred_file_path (str): Path to the predicted labels file.  
    true_file_path (str): Path to the true labels file.  
  
    Returns:  
    accuracy (float): Accuracy of the prediction.  
    confusion_matrix (array): Confusion matrix of the prediction.  
    """  
  
    # Load predicted and true labels  
    pred_labels = np.loadtxt(pred_file_path)  
    true_labels = np.loadtxt(true_file_path)  
  
    # Compute accuracy and confusion matrix  
    accuracy = accuracy_score(true_labels, pred_labels)  
    confusion_matrix = confusion_matrix(true_labels, pred_labels)  
  
    return accuracy, confusion_matrix  
  
def evaluate_speech(pred_file_path, true_file_path):  
    """  
    Evaluate speech prediction using the preprocessed data and the true labels.  
  
    Parameters:  
    pred_file_path (str): Path to the predicted labels file.  
    true_file_path (str): Path to the true labels file.  
  
    Returns:  
    accuracy (float): Accuracy of the prediction.  
    confusion_matrix (array): Confusion matrix of the prediction.  
    """  
  
    # Load predicted and true labels  
    pred_labels = np.loadtxt(pred_file_path)  
    true_labels = np.loadtxt(true_file_path)  
  
    # Compute accuracy and confusion matrix  
    accuracy = accuracy_score(true_labels, pred_labels)  
    confusion_matrix = confusion_matrix(true_labels, pred_labels)  
  
    return accuracy, confusion_matrix  
  
def evaluate_image(pred_file_path, true_file_path):  
    """  
    Evaluate image prediction using the preprocessed data and the true labels.  
  
    Parameters:  
    pred_file_path (str): Path to the predicted labels file.  
    true_file_path (str): Path to the true labels file.  
  
    Returns:  
    accuracy (float): Accuracy of the prediction.  
    confusion_matrix (array): Confusion matrix of the prediction.  
    """  
  
    # Load predicted and true labels  
    pred_labels = np.loadtxt(pred_file_path)  
    true_labels = np.loadtxt(true_file_path)  
  
    # Compute accuracy and confusion matrix  
    accuracy = accuracy_score(true_labels, pred_labels)  
    confusion_matrix = confusion_matrix(true_labels, pred_labels)  
  
    return accuracy, confusion_matrix  
  
def evaluate_video(pred_file_path, true_file_path):  
    """  
    Evaluate video prediction using the preprocessed data and the true labels.  
  
    Parameters:  
    pred_file_path (str): Path to the predicted labels file.  
    true_file_path (str): Path to the true labels file.  
  
    Returns:  
    accuracy (float): Accuracy of the prediction.  
    confusion_matrix (array): Confusion matrix of the prediction.  
    """  
  
    # Load predicted and true labels  
    pred_labels = np.loadtxt(pred_file_path)  
    true_labels = np.loadtxt(true_file_path)  
  
    # Compute accuracy and confusion matrix  
    accuracy = accuracy_score(true_labels, pred_labels)  
    confusion_matrix = confusion_matrix(true_labels, pred_labels)  
  
    return accuracy, confusion_matrix  
  
def evaluate_fusion(pred_file_path, true_file_path):  
    """  
    Evaluate fusion prediction using the preprocessed data and the true labels.  
  
    Parameters:  
    pred_file_path (str): Path to the predicted labels file.  
    true_file_path (str): Path to the true labels file.  
  
    Returns:  
    accuracy (float): Accuracy of the prediction.  
    confusion_matrix (array): Confusion matrix of the prediction.  
    """  
  
    # Load predicted and true labels  
    pred_labels = np.loadtxt(pred_file_path)  
    true_labels = np.loadtxt(true_file_path)  
  
    # Compute accuracy and confusion matrix  
    accuracy = accuracy_score(true_labels, pred_labels)  
    confusion_matrix = confusion_matrix(true_labels, pred_labels)  
  
    return accuracy, confusion_matrix
