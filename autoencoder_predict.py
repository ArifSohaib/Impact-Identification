"""
module to predict class using autoencoder
"""
import pandas as pd
import numpy as np
from autoencoder_model import Autoencoder_model, scale_data
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn import preprocessing


def load_model():
    autoencoder = Autoencoder_model(input_dim=9, encoding_dim=4, mid_acivation='elu')
    model = autoencoder.build_autoencoder()
    #load the trained weights
    model.load_weights('weights/autoencoder_{}layer_{}_{}embed.h5'.format(autoencoder.num_layers, autoencoder.mid_activation, autoencoder.encoding_dim))
    return model

def desc_normal_error():
    """
    describes the reconstruction error on normal data
    """
    #read the data
    data = pd.read_csv('data/normal_data.csv')
    #the first key is the index so we discard it
    feature_keys = data.keys()[1:-1]
    features = scale_data(data[feature_keys].values)
    #use the default autoencoder with 8 inputs and 12 embedings
    model = load_model()

    #predict the 8 input values for each row
    preds = model.predict(features)
    #calcualte the error
    mse = np.mean(np.power(features - preds, 2), axis=1)
    
    error_df = pd.DataFrame({'reconstruction_error':mse})
    print(error_df.describe())

def desc_anomaly_error():
    """
    describes the reconstruction error on anomaly data
    """
    #read the data
    data = pd.read_csv('data/anomaly_data.csv')
    #the first key is the index so we discard it
    feature_keys = data.keys()[1:-1]
    features = scale_data(data[feature_keys].values)
    #use the default autoencoder with 8 inputs and 12 embedings
    model = load_model()

    #predict the 8 input values for each row 
    preds = model.predict(features)
    #calcualte the error
    mse = np.mean(np.power(features - preds, 2), axis=1)
    
    error_df = pd.DataFrame({'reconstruction_error':mse})
    print(error_df.describe())
    print(mse)

def main():
    #read the data
    data = pd.read_csv('data/full_data.csv')
    #the first key is the index and the last key is the real class so we discard both
    feature_keys = data.keys()[1:-1]
    #use the default autoencoder with 8 inputs and 12 embedings
    model = load_model()
    features = scale_data(data[feature_keys].values)
    #predict the 8 input values for each row 
    preds = model.predict(features,batch_size=1)
    #calcualte the error
    mse = np.mean(np.power(features - preds, 2), axis=1)
    
    error_df = pd.DataFrame({'reconstruction_error':mse, 'true_class':data['impact_class'].values})
    #predict impact/non_impact using an error threshold
    threshold = 0.012
    y_pred = ['I' if e > threshold else 'X' for e in error_df.reconstruction_error.values]
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