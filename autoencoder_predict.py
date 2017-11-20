"""
module to predict class using autoencoder
"""
import pandas as pd
import numpy as np
from autoencoder_model import Autoencoder_model, scale_data, get_model
from sklearn.metrics import confusion_matrix, f1_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns



def load_model():
    autoencoder = get_model()
    model = autoencoder.build_autoencoder()
    #load the trained weights
    model.load_weights('weights/autoencoder_{}layer_{}_{}embed.h5'.format(autoencoder.num_layers, autoencoder.mid_activation, autoencoder.encoding_dim))
    return model

def desc_normal_error(test=False):
    """
    describes the reconstruction error on normal data
    """
    if test == False:
        return desc_error('data/normal_data.csv')
    else:
        return desc_error('data/test_normal_data.csv')

def desc_full_error(test=False):
    """
    describes the reconstruction error on anomaly data
    """
    if test==False:
        return desc_error('data/full_data.csv')
    else:
        return desc_error('data/test_full_data.csv')

def desc_anomaly_error(test=False):
    if test==False:
        return desc_error('data/anomaly_data.csv')
    else:
        return desc_error('data/test_anomaly_data.csv')

def desc_error(filename):
    """
    describes the reconstruction error on anomaly data
    """
    #read the data
    data = pd.read_csv(filename)
    #the first key is the index so we discard it
    df = pd.DataFrame(data[data.keys()[1:-1]].values)
    #normalize the data
    min_max_scalar = preprocessing.MinMaxScaler()
    df_norm = min_max_scalar.fit_transform(df.values)
    features = df_norm
    # features = df_norm.values
    #use the default autoencoder with 8 inputs and 12 embedings
    model = load_model()

    #predict the 8 input values for each row
    preds = model.predict(features)
    #calcualte the error
    mse = np.mean(np.power(features - preds, 2), axis=1)

    error_df = pd.DataFrame({'reconstruction_error':mse, 'true_class':data['impact_class'].values})
    print(filename)
    print(error_df.describe())

    error_dict = {'df':error_df,
                  'min':error_df.describe()['reconstruction_error']['min'],
                  '25%':error_df.describe()['reconstruction_error']['25%'],
                  '50%':error_df.describe()['reconstruction_error']['50%'],
                  '75%':error_df.describe()['reconstruction_error']['75%'],
                  'max':error_df.describe()['reconstruction_error']['max']}
    return error_dict
    # return error_df,  error_df.describe()['reconstruction_error']['min'], error_df.describe()['reconstruction_error']['50%'], error_df.describe()['reconstruction_error']['75%']

def get_pred_idx(test=False, min_threshold='25%', max_threshold='75%'):
    """
    returns the indices for values above threshold
    Args:
        test: use test dataset
        min_threshold: min threshold in string, can be min or 25%
        max_threshold: max threshold in string, can be 50% or 75% or max
    """
    if test is False:
        full_error_dict= desc_full_error()
        anomaly_error_dict = desc_anomaly_error()
        # anomaly_error_dict = desc_normal_error()
    else:
        full_error_dict = desc_full_error(test=True)
        anomaly_error_dict = desc_anomaly_error(test=True)
        # anomaly_error_dict = desc_normal_error(test=True)
    impact_idx = list()
    confirm_idx = list()
    #use default threshold if not in list
    if min_threshold not in ['min', '25%']:
        min_threshold = 'min'
    if max_threshold not in ['50%', '75%', 'max']:
        max_threshold = '50%'
    for idx, val in enumerate(full_error_dict['df'].reconstruction_error.values):
        if val > anomaly_error_dict[min_threshold] and val < anomaly_error_dict[max_threshold]:
            impact_idx.append(idx)
        elif val > anomaly_error_dict[max_threshold]:
            confirm_idx.append(idx)
    #for testing print length of array
    # print(len(impact_idx), len(confirm_idx))
    return impact_idx, confirm_idx

def get_pred_idx_max(test=False,  max_threshold='75%'):
    """
    returns the indices for values above threshold
    Args:
        test: use test dataset
        min_threshold: min threshold in string, can be min or 25%
        max_threshold: max threshold in string, can be 50% or 75% or max
    """
    if test is False:
        full_error_dict= desc_full_error()
        anomaly_error_dict = desc_anomaly_error()
        # anomaly_error_dict = desc_normal_error()
    else:
        full_error_dict = desc_full_error(test=True)
        anomaly_error_dict = desc_anomaly_error(test=True)
        # anomaly_error_dict = desc_normal_error(test=True)
    impact_idx = list()
    confirm_idx = list()
    #use default threshold if not in list
    # if min_threshold not in ['min', '25%']:
    #     min_threshold = 'min'
    if max_threshold not in ['25%', '50%', '75%', 'max']:
        max_threshold = '75%'
    for idx, val in enumerate(full_error_dict['df'].reconstruction_error.values):
        if val > anomaly_error_dict[max_threshold]:
            impact_idx.append(idx)
    #for testing print length of array
    # print(len(impact_idx), len(confirm_idx))
    return impact_idx, confirm_idx


def get_pred_data(test=False, min_threshold='min', max_threshold='50%'):
    """
    gets the data to be fed into the post-autoencoder classifier
        Returns:
            data[keys]: the data between min and 50th percentile reconstruction error
            labels: the labels between min and 50th percentile reconstruction error
            confirm_labels: partially classified results
            true_labels: all of the labels for the data
    """

    if test == False:
        mid_idx, idx = get_pred_idx_max(test=False, max_threshold=max_threshold)
        data = pd.read_csv('data/full_data.csv')
        true_labels = data[data.keys()[-1]]
    else:
        mid_idx, idx = get_pred_idx_max(test=True, max_threshold=max_threshold)
        data = pd.read_csv('data/test_full_data.csv')
        true_labels = data[data.keys()[-1]]
    #for testing print number of idx
    # print(len(mid_idx))
    # print(len(idx))
    confirm_labels= np.zeros(shape=len(true_labels))
    confirm_labels[idx] = 1
    data = data.ix[mid_idx]
    keys = data.keys()[1:-1]
    mapping = {'X':0, 'I':1}
    labels = data[data.keys()[-1]].map(mapping)

    return data[keys], labels, confirm_labels, true_labels

def plot_pred_errors(test=False):
    if test is False:
        full_error_dict = desc_full_error()
        normal_error_dict = desc_normal_error()
        anomaly_error_dict = desc_anomaly_error()
    else:
        full_error_dict = desc_full_error(True)
        normal_error_dict = desc_normal_error(True)
        anomaly_error_dict = desc_anomaly_error(True)

    normal_errors = normal_error_dict['df']['reconstruction_error'].values
    anomaly_errors = anomaly_error_dict['df']['reconstruction_error'].values
    fig = plt.figure()
    #1x2 plot, plot 1
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

def main():
    #read the data
    data = pd.read_csv('data/test_full_data.csv')
    #the first key is the index and the last key is the real class so we discard both
    feature_keys = data.keys()[1:-1]
    #use the default autoencoder with 8 inputs and 12 embedings
    model = load_model()
    df = data[feature_keys]
    #normalize the data
    df_norm = (df - df.mean()) / (df.max() - df.min())
    features = df_norm.values
    #predict the 8 input values for each row
    preds = model.predict(features,batch_size=128)
    #calcualte the error
    mse = np.mean(np.power(features - preds, 2), axis=1)

    error_df = pd.DataFrame({'reconstruction_error':mse, 'true_class':data['impact_class'].values})
    #predict impact/non_impact using an error threshold
    anomaly_error_dict = desc_anomaly_error(test=True)
    #values confirmed to be anomaly
    y_pred = ['I' if e > anomaly_error_dict['75%'] else 'X' for e in error_df.reconstruction_error.values]
    #values to be fed to SVM
    # y_pred_check = ['I' if e > anomaly_error_dict['min'] and e < anomaly_error_dict['50%'] else 'X' for e in error_df.reconstruction_error.values]
    #get all the impact indices
    true_labels = preprocessing.label_binarize(data['impact_class'].values,classes=["X","I"])
    pred_labels = preprocessing.label_binarize(y_pred,classes=["X","I"])
    f1 = f1_score(true_labels, pred_labels)
    print("F1 score {}".format(f1))
    conf_matrix = confusion_matrix(error_df.true_class, y_pred)
    print(conf_matrix)
    LABELS = ['Impact', 'Not Impact']
    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()

if __name__ == '__main__':
    main()
