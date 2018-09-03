

class data_manager(object):

    def __init__(self):
        pass

    def get_data(self, file_name, sequence_length, vector_length=10):
        """
        :param length: length of final address
        :return: return strings, int-list as well as one-hot-vectors of addresses
        """
        string_addresses, labels = [], []
        int_list_addresses = []
        integer_addresses = []
        vector_addresses = []
        print('Getting your data now...')
        with open(file_name) as train_data:
            text = train_data.readlines()
            for idx, line in enumerate(text):
                # if idx % 10000 == 0:
                #     print('on example {}'.format(idx))
                address, label = line.split(',')
                while len(address) < vector_length:
                    address = '0' + address
                # get target length string address
                string_addresses.append(address), labels.append(label)
                # get int addresses
                integer_addresses.append(int(address))
                # get list ints for address
                int_list_addresses.append([int(digit) for digit in address])
                # get vector of addresses
                zeros = np.zeros([sequence_length, vector_length])
                zeros[range(sequence_length), int_list_addresses[idx]] = 1
                vector_addresses.append(zeros)
            labels = map(int, labels)
            integer_addresses = np.asarray(integer_addresses)
            int_list_addresses = np.asarray(int_list_addresses)
            vector_addresses = np.asarray(vector_addresses)

            labels_vector = np.zeros([len(text), 2])
            labels_vector[range(labels_vector.shape[0]), labels]= 1
            return string_addresses, integer_addresses, int_list_addresses, vector_addresses, labels, labels_vector




def test_data():
    sequence_length = 10
    vector_length = 10
    manager = data_manager()
    train_save_dir = "Train_data"
    test_save_dir = "Test_data"
    if not os.path.exists(train_save_dir):
        print("starting anew to load and save data...")
        # get the training data
        data = manager.get_data(file_name='train.txt', sequence_length=sequence_length, vector_length=vector_length)
        train_strings, train_int_addresses, train_int_list, train_vectors, train_int_labels, train_onehot_labels = data
        os.mkdir(train_save_dir)
        with open(os.path.join(train_save_dir, 'string_addresses.pickle'), 'wb') as handle:
            pickle.dump(train_strings, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(train_save_dir, 'int_addresses.pickle'), 'wb') as handle:
            pickle.dump(train_int_addresses, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(train_save_dir, 'int_list.pickle'), 'wb') as handle:
            pickle.dump(train_int_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(train_save_dir, 'vector_addresses.pickle'), 'wb') as handle:
            pickle.dump(train_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(train_save_dir, 'int_labels.pickle'), 'wb') as handle:
            pickle.dump(train_int_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(train_save_dir, 'onehot_labels.pickle'), 'wb') as handle:
            pickle.dump(train_onehot_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('train_data.pickle', 'wb') as handle:
            pickle.dump(np.asarray(train_int_list), handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('train_labels.pickle', 'wb') as handle:
            pickle.dump(train_int_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else: # we must have saved data available
        print('loading saved data at {}'.format(test_save_dir))

        with open(os.path.join(train_save_dir, 'string_addresses.pickle'), 'rb') as handle:
            train_strings = pickle.load(handle)
        with open(os.path.join(train_save_dir, 'int_addresses.pickle'), 'rb') as handle:
            train_int_addresses = pickle.load(handle)
        with open(os.path.join(train_save_dir, 'int_list.pickle'), 'rb') as handle:
            train_int_list = pickle.load(handle)

        with open(os.path.join(train_save_dir, 'vector_addresses.pickle'), 'rb') as handle:
            train_vectors = pickle.load(handle)

        with open(os.path.join(train_save_dir, 'int_labels.pickle'), 'rb') as handle:
            train_int_labels = pickle.load(handle)
        with open(os.path.join(train_save_dir, 'onehot_labels.pickle'), 'rb') as handle:
            train_onehot_labels = pickle.load(handle)

        with open('train_data.pickle', 'wb') as handle:
            pickle.dump(np.asarray(np.asarray(train_int_list)), handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('train_labels.pickle', 'wb') as handle:
            pickle.dump(train_int_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # get the test data
    if not os.path.exists(test_save_dir):
        data = manager.get_data(file_name='test.txt', sequence_length=sequence_length, vector_length=vector_length)
        test_strings, test_int_addresses, test_int_list, test_vectors, test_int_labels, test_onehot_labels = data
        os.mkdir(test_save_dir)
        with open(os.path.join(test_save_dir, 'string_addresses.pickle'), 'wb') as handle:
            pickle.dump(test_strings, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(test_save_dir, 'int_addresses.pickle'), 'wb') as handle:
            pickle.dump(test_int_addresses, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(test_save_dir, 'int_list.pickle'), 'wb') as handle:
            pickle.dump(test_int_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(test_save_dir, 'vector_addresses.pickle'), 'wb') as handle:
            pickle.dump(test_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(test_save_dir, 'int_labels.pickle'), 'wb') as handle:
            pickle.dump(test_int_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(test_save_dir, 'onehot_labels.pickle'), 'wb') as handle:
            pickle.dump(test_onehot_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('test_data.pickle', 'wb') as handle:
            pickle.dump(np.asarray(test_int_list), handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('test_labels.pickle', 'wb') as handle:
            pickle.dump(test_int_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:  # we must have saved data available
        print('loading saved data at {}'.format(test_save_dir))

        with open(os.path.join(test_save_dir, 'string_addresses.pickle'), 'rb') as handle:
            test_strings = pickle.load(handle)
        with open(os.path.join(test_save_dir, 'int_addresses.pickle'), 'rb') as handle:
            test_int_addresses = pickle.load(handle)
        with open(os.path.join(test_save_dir, 'int_list.pickle'), 'rb') as handle:
            test_int_list = pickle.load(handle)

        with open(os.path.join(test_save_dir, 'vector_addresses.pickle'), 'rb') as handle:
            test_vectors = pickle.load(handle)

        with open(os.path.join(test_save_dir, 'int_labels.pickle'), 'rb') as handle:
            test_int_labels = pickle.load(handle)
        with open(os.path.join(test_save_dir, 'onehot_labels.pickle'), 'rb') as handle:
            test_onehot_labels = pickle.load(handle)

        with open('test_data.pickle', 'wb') as handle:
            pickle.dump(np.asarray(np.asarray(test_int_list)), handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('test_labels.pickle', 'wb') as handle:
            pickle.dump(test_int_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(test_save_dir, 'test.pickle'), 'wb') as handle:
            pickle.dump(test_onehot_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Train Addresses ==> ", len(train_strings), train_int_addresses.shape, train_vectors.shape)
    print("Train Labels ==> ", len(train_int_labels), train_onehot_labels.shape)
    print("Test Addresses ==> ", len(test_strings), test_int_addresses.shape, test_vectors.shape)
    print("Test Labels ==> ", len(test_int_labels), test_onehot_labels.shape)
    pass
