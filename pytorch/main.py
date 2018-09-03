

"""
    A simple script to train a small lstm network on memory addresses against their cache hit and miss
    Basically each address is read from left to right using an lstm network, and based on the final state,
    we predict whether it will be a cache hit or miss!
"""

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optimizers
import os
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import*
from sklearn import preprocessing


class NN(nn.Module):
    """
        Our class defines MLP and LSTM models to test our data learning
    """
    def __init__(self, input_dims):
        super(NN, self).__init__()
        self.input_dims = input_dims
        # define MLP
        self.MLP = nn.Sequential(
            nn.Linear(input_dims, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            # nn.ReLU(),
            # nn.Linear(4, 2),
            nn.LogSoftmax()
        )

        class LSTM(nn.Module):
            """
                This is our lstm class
            """
            def __init__(self, input_size, seq_length, hidden_size):
                super(LSTM, self).__init__()
                self.sequence_len = seq_length
                self.hidden_size = hidden_size
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
                self.fc_1 = nn.Linear(hidden_size, 2)

            def forward(self, input):
                # input = input.view(input.size()[0], 1, -1)
                h0 = torch.zeros(1, input.size()[0], self.hidden_size)  # (num_of_lstm_layers, batch_size, input_size)
                c0 = torch.zeros(1, input.size()[0], self.hidden_size)
                output, hidden = self.lstm(input, (h0, c0))
                output = self.fc_1(output).view(-1, 2)
                return output

        # instantiate lstm model with one layer of 8 hidden units
        self.my_lstm = LSTM(input_size=3, seq_length=1, hidden_size=8)


    def train_net(self, model, train_data, train_labels,
                  test_data, test_labels, epochs,
                  batch_size, learn_rate):
        # set it in training mode, and do some data conversion for pytorch
        self.train()
        train_data = torch.Tensor(train_data).float()
        train_labels = torch.Tensor(train_labels).long()
        optimizer = optimizers.Adam(self.parameters(), lr=learn_rate)
        criterion = nn.CrossEntropyLoss()
        # training loop for #N epochs
        for e in range(1, epochs+1):
            for i in range(train_data.size()[0]//batch_size):
                # pick random *batch_size* # of examples to train on...
                indices = np.random.choice(train_data.size()[0], batch_size, replace=False)
                batch_data = train_data[indices,:]
                batch_labels = train_labels[indices]

                output = model(batch_data)
                prediction = output.max(dim=1, keepdim=True)[1]
                batch_train_acc = batch_labels.eq(prediction.view_as(batch_labels)).sum().item()
                loss = criterion(output, batch_labels.view(-1))
                batch_loss = loss.item()
                loss.backward()
                optimizer.step()

                # zero out the gradients saved previously
                optimizer.zero_grad()
                self.zero_grad()

            print('epoch ({}/{}), batch_loss = {:.2f}, batch_acc = {:.2f}%'.format(e, epochs, batch_loss,
                                                                                   batch_train_acc*100.0/batch_size))
        # print('log: saving model now...')
        # torch.save(self.state_dict(), 'models/model-{}.ckpt'.format(e))
        print('\n testing now... \n')
        return self.test_model(model=model, test_examples=test_data, labels=test_labels)

    def test_model(self, model, test_examples, labels):
        # check performance on test set
        self.eval() # set in eval mode
        test_examples = torch.Tensor(test_examples).float()
        labels = torch.Tensor(labels).long()
        print('testing on {} examples...'.format(test_examples.size()[0]))
        output = model(test_examples)
        prediction = output.max(dim=1, keepdim=True)[1]
        accurate = prediction.eq(labels.view_as(prediction)).sum().item()
        print('Total test accuracy = {:.2f}%'.format(accurate*100/(test_examples.size()[0])))
        return prediction


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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


def main():
    # load data files
    data_dir = '../train_and_test_data'
    train_data_file = open(os.path.join(data_dir, 'train_data_1.pkl'), 'rb')
    train_labels_file = open(os.path.join(data_dir, 'train_labels_1.pkl'), 'rb')
    train_data = pickle.load(train_data_file)
    train_labels = pickle.load(train_labels_file)
    test_data_file = open(os.path.join(data_dir, 'test_data_1.pkl'), 'rb')
    test_labels_file = open(os.path.join(data_dir, 'test_labels_1.pkl'), 'rb')
    test_data = pickle.load(test_data_file)
    test_labels = pickle.load(test_labels_file)

    # preprocess them for easing the training...
    train_data = preprocessing.scale(train_data)
    test_data = preprocessing.scale(test_data)

    train_data_for_lstm = train_data.reshape((-1, 1, 3))
    test_data_for_lstm = test_data.reshape((-1, 1, 3))

    # reshape for one column
    train_labels = np.reshape(train_labels, newshape=(len(train_labels), 1))
    test_labels = np.reshape(test_labels, newshape=(len(test_labels), 1))

    print('3d training data, 1d labels: ', train_data.shape, train_labels.shape)
    print('3d test data, 1d labels: ', test_data.shape, test_labels.shape)

    # create model and train
    net = NN(input_dims=3)

    ###############################################################################################3
    # the first two lines are mlp implementation, the next two run the lstm
    # test mlp
    # predictions = net.train_net(net.MLP, train_data, train_labels, test_data, test_labels,
    #                             epochs=10, batch_size=1024, learn_rate=0.1)

    # test lstm
    predictions = net.train_net(net.my_lstm, train_data_for_lstm, train_labels, test_data_for_lstm,
                                test_labels, epochs=10, batch_size=1024, learn_rate=0.1)
    ##############################################################################################3

    conf_matrix = confusion_matrix(test_labels, predictions)
    classes = ['page_hit', "page_miss"]
    plot_confusion_matrix(conf_matrix, classes, normalize=True)
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    main()










