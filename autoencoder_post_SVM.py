import numpy as np
from autoencoder_predict import get_pred_data
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

features, labels, partial_labels, true_labels = get_pred_data(test=False, min_threshold='25%', max_threshold='75%')
test_features, test_labels, test_partial_labels, test_true_labels = get_pred_data(test=True, min_threshold='25%', max_threshold='75%')
class_weights = {1: 7, 0: 1}

clf = SVC(C=20.0, decision_function_shape='ovr', kernel='rbf', gamma='auto', class_weight=class_weights)
test_features = (test_features - test_features.mean()) / (test_features.max() - test_features.min())
features = (features - features.mean()) / (features.max() - features.min())
clf.fit(features.values,labels.values) #train it on old dataset
preds = clf.predict(features.values)
test_preds = clf.predict(test_features.values)
preds = [1 if p > 0.5 else 0 for p in preds]
idx = [i for i, j in enumerate(preds) if j == 1]
partial_labels[idx] = 1
#binarize the pridictions
test_preds = [1 if p > 0.5 else 0 for p in test_preds]
test_idx = [i for i, j in enumerate(test_preds) if j == 1]
test_partial_labels[test_idx] = 1
#calcualte the F1 score
mapping = {'X':0., 'x':0., 'i': 1., 'I':1.}
test_true_labels = test_true_labels.map(mapping).values
true_labels = true_labels.map(mapping).values
print("true labels:{}".format(test_true_labels))
print("pred labels:{}".format(test_partial_labels))
f1_test = f1_score(test_true_labels, test_partial_labels)

#get the indices of predicted colissions to a file
predicted_idx = [i for i, j in enumerate(test_partial_labels) if j == 1]
#save the indices to a file, use later to check what kind of colissions were predicted
np.save('./data/pred_colission_idx.npy',predicted_idx)

f1_train = f1_score(true_labels, partial_labels)
print("test F1 Score: {}".format(f1_test))
print("train F1 score: {}".format(f1_train))
conf_matrix_train = confusion_matrix(true_labels, partial_labels)
conf_matrix = confusion_matrix(test_true_labels, test_partial_labels)
print(conf_matrix_train)
#plot the confusion matrix
LABELS = ['Not Impact', 'Impact']
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
