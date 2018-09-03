

from __future__ import print_function
from __future__ import division
import numpy as np
import tensorflow as tf


def get_dataset(file_name, one_hot=False):
    """
        Will return the dataset in a trainable format
    :param file_name: name of the file to extract the data from...
    :param one_hot: want the data as one hot or not?
    :return: the addresses along with their hit/miss status
    """
    def fit_to_length(addr, target_length):
        # step 1. make same size
        while len(addr) < target_length:
            addr = '0'+addr
        return addr

    with open(file_name) as this_file:
        examples = [x.split(',') for x in this_file.readlines()]
        dataset = np.asarray(examples)

    if not one_hot:
        return [fit_to_length(x, 10) for x in dataset[:, 0]], np.asarray([map(int, dataset[:, 1])])

    labels = np.asarray(map(int, dataset[:, 1]))
    one_hot_labels = np.zeros(shape=(labels.shape[0], 2))
    one_hot_labels[range(labels.shape[0]), labels] = 1
    return [fit_to_length(x, 10) for x in dataset[:, 0]], np.asarray(one_hot_labels)


class Network(object):
    """
        Defines the lstm network and trains it
    """
    def __init__(self, sequence_length, vector_size, hidden_size, num_classes):
        self.sequence_length = sequence_length
        self.vector_size = vector_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.x = tf.placeholder(dtype="float", shape=[None, sequence_length, vector_size])
        self.y = tf.placeholder(dtype="float", shape=[None, num_classes])
        self.istate = tf.placeholder(dtype="float", shape=[None, 2*hidden_size])

        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        self.initializer = tf.random_uniform_initializer(-1, 1)
        self.lstm = tf.nn.rnn_cell.LSTMCell(self.hidden_size, initializer=self.initializer)

        self.weights = {
            'hidden': weight_variable([vector_size, hidden_size]),
            'fc_1': weight_variable([hidden_size, 1024]),
            'fc_2': weight_variable([1024, num_classes]),
            'fc_3': weight_variable([512, 64]),
            'fc_4': weight_variable([64, num_classes])
        }
        self.biases = {
            'hidden': bias_variable([hidden_size]),
            'fc_1': bias_variable([1024]),
            'fc_2': bias_variable([num_classes]),
            'fc_3': bias_variable([64]),
            'fc_4': bias_variable([num_classes]),
        }

        pass

    def train_model(self, train_set, test_set, learning_rate, epochs, batch_size, log_after, test_after, save_after):
        # training
        def forward(lstm, sequence_length, x_batch, weights, biases):
            x_batch = tf.transpose(x_batch, [1, 0, 2])
            x_batch = tf.reshape(x_batch, [-1, self.vector_size])
            # x_batch = tf.nn.relu(tf.matmul(x_batch, weights['hidden']) + biases['hidden'])
            # x_batch = tf.nn.dropout(x_batch, keep_prob=0.9)
            # x_batch = tf.split(x_batch, sequence_length, 0)
            outputs, states = tf.nn.static_rnn(cell=lstm, inputs=x_batch, sequence_length=sequence_length,
                                                dtype=tf.float32)
            # print(outputs.shape)
            # outputs, states = tf.nn.dynamic_rnn(cell=lstm_2, inputs=outputs, sequence_length=sequence_length,
            #                                     dtype=tf.float32, scope='lstm_2')
            # print(outputs.shape)
            # indices = tf.stack([tf.range(batch_size), sequence_length - 1], axis=1)
            # outputs = tf.gather_nd(outputs, indices)

            fc_1 = tf.nn.relu(tf.matmul(outputs[-1], weights['fc_1']) + biases['fc_1'])
            # fc_1 = tf.nn.dropout(fc_1, keep_prob=0.9)
            fc_2 = tf.matmul(fc_1, weights['fc_2']) + biases['fc_2']
            # fc_3 = tf.nn.relu(tf.matmul(fc_2, weights['fc_3']) + biases['fc_3'])
            # fc_4 = tf.nn.relu(tf.matmul(fc_3, weights['fc_4']) + biases['fc_4'])
            fc_4 = tf.nn.softmax(fc_2, dim=1)
            return fc_4


        train_x, train_y = train_set
        x_test, y_test = test_set
        self.output = forward(self.lstm, self.sequence_length, self.x, self.weights, self.biases)
        cost = tf.losses.softmax_cross_entropy(logits=self.output, onehot_labels=self.y)  # Softmax loss
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)  # Adam Optimizer
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.output, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        NUM_THREADS = 5
        sess = tf.Session(
            config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS, inter_op_parallelism_threads=NUM_THREADS,
                                  log_device_placement=False))
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()

        # saver.restore(sess, 'models/model-11000')
        # print('restored saved model!')
        # training loop
        step = 1
        while step * batch_size < epochs:
            # [batch_size, 28 x 28], [batch_size, 10]
            # get random distinct indices between 0-num_examples
            indices = np.random.choice(train_x.shape[0], batch_size, replace=False)
            batch_xs, batch_ys = train_x[indices,:,:], train_y[indices,:]
            # [batch_size, 28 x 28] -> [batch_size, n_steps, n_input]
            batch_xs = batch_xs.reshape((batch_size, self.sequence_length, self.vector_size))
            # [batch_size, 2*128]
            c_istate = np.random.rand(batch_size, 2 * self.hidden_size)
            sess.run(optimizer, feed_dict={self.x: batch_xs, self.y: batch_ys, self.istate: c_istate})
            if step % log_after == 0:
                acc = sess.run(accuracy, feed_dict={self.x: batch_xs, self.y: batch_ys, self.istate: c_istate})
                loss = sess.run(cost, feed_dict={self.x: batch_xs, self.y: batch_ys, self.istate: c_istate})
                print("step : " + str(step * batch_size) + ", Minibatch Loss= " + "{:.2f}".format(
                    loss) + ", Training Accuracy= " + "{:.2f}%".format(acc*100))
            if step % save_after == 0:
                saver.save(sess, 'models/model-{}'.format(step))
                print('\n\t Saved model!')
            if step % test_after == 0:
                self.test_model(sess, accuracy, x_test, y_test, self.sequence_length, self.vector_size, self.hidden_size,
                                batch_size)
            step += 1

        pass

    def test_model(self, sess, accuracy, x_test, y_test, sequence_length, vector_size, hidden_size, batch_size):
        step = 1
        lower, upper = 0, batch_size
        net_acc = []
        while True:
            if upper > x_test.shape[0]:
                break
            # [batch_size, 28 x 28], [batch_size, 10]
            batch_xs, batch_ys = x_test[lower:upper,:,:], y_test[lower:upper,:]
            # [batch_size, 28 x 28] -> [batch_size, n_steps, n_input]
            batch_xs = batch_xs.reshape((batch_size, sequence_length, vector_size))
            # [batch_size, 2*128]
            c_istate = np.random.rand(batch_size, 2 * hidden_size)
            acc = sess.run(accuracy, feed_dict={self.x: batch_xs, self.y: batch_ys, self.istate: c_istate})
            net_acc.append(acc)
            lower += batch_size
            upper += batch_size
            step += 1
        print("Test Accuracy= {:.2f}%".format(np.asarray(net_acc).mean(axis=0)))


def main():
    # we receive the addresses as strings and their hit/miss labels as integers
    train_addresses, train_labels = get_dataset(file_name='train.txt', one_hot=True)
    test_addresses, test_labels = get_dataset(file_name='test.txt', one_hot=True)
    print('log: train data size: {} {}'.format(len(train_addresses), train_labels.shape))
    print('log: test data size: {} {}'.format(len(test_addresses), test_labels.shape))

    # next, we need a routine to convert each address into a sequence of vectors that we could feed into an lstm
    # and also we need to make them the same size as well
    def address_to_vector(addr, target_length):
        # step 1. make same size
        while len(addr) < target_length:
            addr = '0'+addr
        # step 2. turn into a matrix, basically target_length*10, target_length for the digits in the address,
        #         and 10 for the 10 possible digits at each spot, so... (and we assume a batch size = 1)
        vec = np.zeros(shape=(target_length, 1, 10)) # since there are 10 possible digits!
        for index, digit in enumerate(addr):
            vec[index, 0, int(digit)] = 1
        return vec

    embedded_train = np.asarray([address_to_vector(x, 10) for x in train_addresses]).reshape(-1, 10, 10)
    embedded_test = np.asarray([address_to_vector(x, 10) for x in test_addresses]).reshape(-1, 10, 10)
    print('log: new embedded train addresses size: {}'.format(embedded_train.shape))
    print('log: new embedded test addresses size: {}'.format(embedded_test.shape))
    # print(train_addresses[675], embedded_addresses[675, :, :])

    # initialize some variables
    learn_rate = 0.001
    hidden_size = 256
    sequence_length = 10
    each_vector_in_sequence_length = 10
    num_classes = 2
    batch_size = 16
    iterations = 1000000

    lstm = Network(sequence_length, each_vector_in_sequence_length, hidden_size, num_classes)
    lstm.train_model(train_set=(embedded_train, train_labels), test_set=(embedded_test, test_labels),
                     learning_rate=learn_rate, epochs=iterations, batch_size=batch_size, log_after=100,
                     test_after=5000, save_after=1000)
    pass


if __name__ == '__main__':
    main()



