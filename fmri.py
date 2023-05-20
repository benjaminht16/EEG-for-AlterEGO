import numpy as np 
import pandas as pd 
import nibabel as nib 
from nilearn import image, signal 
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import Ridge 
from sklearn.pipeline import make_pipeline 
from sklearn.model_selection import KFold, cross_val_score 
 
 
def load_fmri_data(fmri_file, mask_file=None): 
    """ 
    Load fMRI data from a NIfTI file and a corresponding mask file (optional). 
 
    Params: 
    - fmri_file (str): path to the fMRI data file 
    - mask_file (str or None): path to the mask data file. If None, the full brain is used. 
 
    Returns: 
    - fmri_data (numpy array): fMRI data array with shape (n_samples, n_voxels) 
    - confounds (pandas dataframe): dataframe of confounds with shape (n_samples, n_confounds) 
    - mask_data (numpy array or None): mask data array with shape (n_voxels,) if mask_file is not None, else None 
    """ 
    fmri_img = nib.load(fmri_file) 
    fmri_data = fmri_img.get_fdata() 
    fmri_data = signal.clean(fmri_data, sessions=None, detrend=True, standardize=True) 
 
    if mask_file is not None: 
        mask_img = nib.load(mask_file) 
        mask_data = mask_img.get_fdata() 
        fmri_data = fmri_data[:, mask_data.astype(bool)] 
    else: 
        mask_data = None 
 
    confounds_file = fmri_file.replace('bold.nii.gz', 'confounds.tsv') 
    confounds = pd.read_csv(confounds_file, sep='\t') 
 
    return fmri_data, confounds, mask_data 
 
 
def preprocess_fmri_data(fmri_data, confounds): 
    """ 
    Preprocess fMRI data by regressing out the confounds and performing voxel-wise normalization. 
 
    Params: 
    - fmri_data (numpy array): fMRI data array with shape (n_samples, n_voxels) 
    - confounds (pandas dataframe): dataframe of confounds with shape (n_samples, n_confounds) 
 
    Returns: 
    - fmri_data_processed (numpy array): preprocessed fMRI data array with shape (n_samples, n_voxels) 
    """ 
    # Regress out confounds 
    regressors = ['global_signal', 'csf', 'white_matter', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'] 
    confounds_regress = confounds[regressors] 
    scaler = StandardScaler() 
    fmri_data_scaled = scaler.fit_transform(fmri_data) 
    model = make_pipeline(StandardScaler(), Ridge(alpha=1)) 
    for i in range(fmri_data.shape[1]): 
        model.fit(confounds_regress, fmri_data_scaled[:, i]) 
        fmri_data_scaled[:, i] = fmri_data_scaled[:, i] - model.predict(confounds_regress) 
    fmri_data_processed = scaler.inverse_transform(fmri_data_scaled) 
 
    # Normalize voxel-wise 
    pca = PCA(n_components=1, svd_solver='full') 
    fmri_data_normalized = np.zeros_like(fmri_data_processed) 
    for i in range(fmri_data_processed.shape[1]): 
        pca.fit(fmri_data_processed[:, i, np.newaxis]) 
        fmri_data_normalized[:, i] = pca.components_.reshape(-1) * pca.singular_values_ 
 
    return fmri_data_normalized 
 
 
def train_fmri_model(fmri_data, labels, cv=5): 
    """ 
    Train an fMRI predictive model using voxel-wise cross-validation. 
 
    Params: 
    - fmri_data (numpy array): fMRI data array with shape (n_samples, n_voxels) 
    - labels (numpy array): array of labels with shape (n_samples,) 
    - cv (int): number of cross-validation folds 
 
    Returns: 
    - model (sklearn estimator): trained predictive model 
    """ 
    model = Ridge(alpha=1) 
    cv_scores = cross_val_score(model, fmri_data, labels, cv=KFold(n_splits=cv)) 
    print(f'Cross-validation accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}') 
 
    model.fit(fmri_data, labels) 
 
    return model 
 
 
def predict_fmri_labels(fmri_data, model):
    """ 
    Predict labels using a trained fMRI predictive model. 
 
    Params: 
    - fmri_data (numpy array): fMRI data array with shape (n_samples, n_voxels) 
    - model (sklearn estimator): trained predictive model 
 
    Returns: 
    - labels_predicted (numpy array): array of predicted labels with shape (n_samples,) 
    """ 
    return model.predict(fmri_data)
