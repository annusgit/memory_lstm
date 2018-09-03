

import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import *
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                           normalize=False,
                           title='Confusion matrix',
                           cmap=plt.cm.Blues):
     """
     This function prints and plots the confusion matrix.
     Normalization can be applied by setting `normalize=True`.
     """
     if normalize:
         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
         print("Normalized confusion matrix")
     else:
         print('Confusion matrix, without normalization')

     print(cm)

     plt.imshow(cm, interpolation='nearest', cmap=cmap)
     plt.title(title)
     plt.colorbar()
     tick_marks = np.arange(len(classes))
     plt.xticks(tick_marks, classes, rotation=45)
     plt.yticks(tick_marks, classes)

     fmt = '.2f' if normalize else 'd'
     thresh = cm.max() / 2.
     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
         plt.text(j, i, format(cm[i, j], fmt),
                  horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")

     plt.tight_layout()
     plt.ylabel('True label')
     plt.xlabel('Predicted label')
     return plt

def data_load(inputpath):
     with open(inputpath, 'rb') as f:
         data = pickle.load(f)
     return np.asarray(data)

# test_file = "../data-set/test.pkl"
train_data_file = "train_data_1.pkl"
train_labels_file = "train_labels_1.pkl"
test_data_file = "test_data_1.pkl"
test_labels_file = "test_labels_1.pkl"
X_train = data_load(train_data_file)
train_y = data_load(train_labels_file)
X_test = data_load(test_data_file)
test_y = data_load(test_labels_file)

# X_test, test_y = data_load(test_file)

train_X = preprocessing.scale(X_train)
test_X = preprocessing.scale(X_test)
print(train_X[8,:])
print(train_X.shape, test_X.shape)

clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
clf.fit(train_X, train_y)
y_predict = clf.predict(test_X)
accuracy=accuracy_score(test_y, y_predict)
cm = confusion_matrix(test_y, y_predict)

print accuracy
print cm
classes = ['page_hit', "page_miss"]
plot_confusion_matrix(cm, classes,normalize=True)
plt.show()

