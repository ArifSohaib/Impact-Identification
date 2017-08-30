import numpy as np
from autoencoder_predict import get_pred_data
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

features, labels = get_pred_data()
print(features.values.shape)
clf = SVC()
clf.fit(features.values,labels.values)
preds = clf.predict(features.values)
#binarize the pridictions
preds = [1 if p > 0.5 else 0 for p in preds]
#calcualte the F1 score
print(f1_score(preds, labels.values))
conf_matrix = confusion_matrix(preds, labels.values)
#plot the confusion matrix
LABELS = ['Not Impact', 'Impact']
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
