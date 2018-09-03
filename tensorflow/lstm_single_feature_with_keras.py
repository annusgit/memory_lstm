

from __future__ import print_function
from __future__ import division

import keras as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop
from sklearn import preprocessing
from sklearn.metrics import*
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle


class get_better_data(object):
    """
        Returns data in the following form
            [current_address, previous_address, difference_of_the_two]
    """
    def __init__(self):

        pass

    def get(self, file_name, vector_length=10):
        string_addresses, labels = [], []
        integer_addresses = []
        still_addresses = []
        separate_int_addresses = []
        print('Getting your data now...')
        file_name = '../train_and_test_data/{}'.format(file_name)
        with open(file_name+'.txt') as train_data:
            text = train_data.readlines()
            for idx, line in enumerate(text):
                address, label = line.split(',')
                while len(address) < vector_length:
                    address = '0' + address
                # get target length string address
                string_addresses.append(address), labels.append(label)
                separate_int_addresses.append([int(x) for x in address])
                # get int addresses
                integer_addresses.append(int(address))
                # get list ints for address
        " Features will be of the form "
        features = []
        for k in range(0, len(integer_addresses)-1): # leave the first and last address
            features.append((integer_addresses[k],
                             integer_addresses[k+1],
                             integer_addresses[k+1]-integer_addresses[k]))
            still_addresses.append(([integer_addresses[k]],
                                    [integer_addresses[k + 1]],
                                    [integer_addresses[k + 1] - integer_addresses[k]]))
        labels = map(int, labels)
        feature_labels = map(int, labels[0:len(labels)-1]) # leave the first and last address
        integer_addresses = np.asarray(integer_addresses)
        labels_vector = np.zeros([len(feature_labels), 2])
        labels_vector[range(labels_vector.shape[0]), feature_labels] = 1
        single_address_labels = np.zeros([len(labels), 2])
        single_address_labels[range(single_address_labels.shape[0]), labels] = 1

        with open('{}_data_3.pickle'.format(file_name), 'wb') as handle:
            pickle.dump(np.asarray(features), handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('{}_labels_3.pickle'.format(file_name), 'wb') as handle:
            pickle.dump(np.asarray(feature_labels), handle, protocol=pickle.HIGHEST_PROTOCOL)

        return string_addresses, np.asarray(separate_int_addresses), np.asarray(features), \
               np.asarray(still_addresses), np.asarray(feature_labels), labels, labels_vector, single_address_labels


class Network(object):

    def MLP(self, train_features, train_labels, test_features, test_labels, iterations, batch_size):
        model = Sequential()
        # print(train_features[0].shape)
        model.add(Dense(4, input_shape=(3,), activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        optimizer = Adam(lr=0.01)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_features, train_labels, epochs=iterations, batch_size=batch_size)
        loss, accuracy = model.evaluate(test_features, test_labels)
        predictions = model.predict(test_features)
        return predictions, loss, accuracy

    def LSTM(self, train_features, train_labels, test_features, test_labels, iterations, batch_size):
        # print(train_features.shape)
        model = Sequential()
        # model.add(Embedding(10, 100, input_length=3))
        model.add(LSTM(16, input_shape=(1, 10)))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        optimizer = Adam(lr=0.01)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_features, train_labels, epochs=iterations, batch_size=batch_size)
        loss, accuracy = model.evaluate(test_features, test_labels)
        predictions = model.predict(test_features)
        return predictions, loss, accuracy


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


def Main():

    data = get_better_data()
    _, int_train_features, train_features, still_train_features, _, _, _, train_labels = data.get('train', vector_length=10)
    _, int_test_features, test_features, still_test_features, _, _, _, test_labels = data.get('test', vector_length=10)
    print('train data', int_train_features.shape, train_labels.shape)
    print('test data', int_test_features.shape, test_labels.shape)
    print(test_features[79,:], test_labels[79])

    train_features = preprocessing.scale(int_train_features)
    test_features = preprocessing.scale(int_test_features)

    still_train_features = train_features.reshape(-1, 1, 10)
    still_test_features = test_features.reshape(-1, 1, 10)

    net = Network()
    # predictions, loss, acc = net.MLP(train_features, train_labels, test_features, test_labels,
    #                                  iterations=10, batch_size=1024)
    # print('MLP: test loss = {:.2f}, test acc = {:.2f}%'.format(loss, 100*acc))
    predictions, loss, acc = net.LSTM(still_train_features, train_labels, still_test_features, test_labels,
                                      iterations=10, batch_size=1024)
    print('LSTM: test loss = {:.2f}, test acc = {:.2f}%'.format(loss, 100*acc))
    conf_matrix = confusion_matrix(np.argmax(test_labels, axis=1), np.argmax(predictions, axis=1))

    classes = ['page_hit', "page_miss"]
    plot_confusion_matrix(conf_matrix, classes, normalize=True)
    plt.show()

    pass


if __name__ == '__main__':
    Main()
















