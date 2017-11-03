"""
Uses data returned after autoencoder to do prediction
"""
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras import regularizers
from keras.models import Model
from autoencoder_model import scale_data
from autoencoder_predict import get_pred_data
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class Post_Autoencoder:
    """
    class uses outputs from autoencoder model in a classifier
    """
    def build_model(self):
        input_layer = Input(shape=(8,))
        x = BatchNormalization()(input_layer)
        x = Dense(16, activation='elu')(x)
        # x = BatchNormalization()(x)
        x = Dense(8, activation='elu')(x)
        # x = BatchNormalization()(x)
        x = Dense(4, activation='elu')(x)
        x = Dense(1, activation='relu')(x)
        return Model(inputs=input_layer, outputs=x)

def main():
    model = Post_Autoencoder()
    model = model.build_model()
    model.compile(loss='binary_crossentropy',
                  optimizer='adadelta', metrics=['accuracy'])
    class_weights = {1:3, 0:1}
    features, labels, partial_labels, true_labels = get_pred_data(test=False, min_threshold='min', max_threshold='75%')
    test_features, test_labels, test_partial_labels, test_true_labels = get_pred_data(
        test=True, min_threshold='min', max_threshold='75%')
    #normalize the test features
    test_features = (test_features - test_features.mean()) /(test_features.max() - test_features.min())

    #normalize the train features
    features = (features - features.mean()) / (features.max() - features.min())
    if len(features) < 2000:
        n_epochs = 1000
    else:
        n_epochs = 3000
    # features = scale_data(data.values)
    model.fit(features.values, labels.values, class_weight=class_weights, epochs=n_epochs, shuffle=False)
    preds = model.predict(features.values)

    test_preds = model.predict(test_features.values)
    preds = [1 if p > 0.5 else 0 for p in preds]
    idx = [i for i, j in enumerate(preds) if j == 1]
    partial_labels[idx] = 1
    #binarize the pridictions
    test_preds = [1 if p > 0.5 else 0 for p in test_preds]
    test_idx = [i for i, j in enumerate(test_preds) if j == 1]
    test_partial_labels[test_idx] = 1
    #calcualte the F1 score
    mapping = {'X': 0., 'x': 0., 'i': 1., 'I': 1.}
    test_true_labels = test_true_labels.map(mapping).values
    true_labels = true_labels.map(mapping).values
    print("true labels:{}".format(test_true_labels))
    print("pred labels:{}".format(test_partial_labels))
    f1_test = f1_score(test_true_labels, test_partial_labels)
    f1_train = f1_score(true_labels, partial_labels)
    print("test F1 Score: {}".format(f1_test))
    print("train F1 score: {}".format(f1_train))
    conf_matrix_train = confusion_matrix(true_labels, partial_labels)
    conf_matrix = confusion_matrix(test_true_labels, test_partial_labels)
    print(conf_matrix_train)
    #plot the confusion matrix
    LABELS = ['Not Impact', 'Impact']
    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=LABELS,
            yticklabels=LABELS, annot=True, fmt="d")
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
if __name__ == '__main__':
    main()
