import numpy as np
from autoencoder_predict import get_pred_data
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

features, labels, conf_features, conf_labels = get_pred_data(test=False)
test_features, test_labels, test_conf_features, test_conf_labels = get_pred_data(test=True)
clf = SVC(C=2.5, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf')
clf.fit(features.values,labels.values) #train it on old dataset
preds = clf.predict(features.values)
test_preds = clf.predict(test_features.values)
preds = [1 if p > 0.5 else 0 for p in preds]
# preds.append()
#binarize the pridictions
test_preds = [1 if p > 0.5 else 0 for p in test_preds]
#calcualte the F1 score
# print("test_F1 score: {}".format(f1_score(test_preds, test_labels.values)))
f1_test = f1_score(np.append(test_preds, test_conf_labels.values), np.append(test_labels.values, test_conf_labels.values))
f1_train = f1_score(np.append(preds, conf_labels.values), np.append(labels.values, conf_labels.values))
print("test F1 Score: {}".format(f1_test))
print("train F1 score: {}".format(f1_train))
conf_matrix_train = confusion_matrix(np.append(preds, conf_labels.values), np.append(labels.values, conf_labels.values))
conf_matrix = confusion_matrix(np.append(test_preds, test_conf_labels.values), np.append(test_labels.values, test_conf_labels.values))
print(conf_matrix_train)
#plot the confusion matrix
LABELS = ['Not Impact', 'Impact']
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
