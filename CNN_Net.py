import load
import numpy as np
from dp_refined import Network


# def train_data_iterator(samples, labels, iteration_steps, chunkSize):
#     """
#     Iterator/Generator: get a batch of data
#     用于 for loop， just like range() function
#     """
#     if len(samples) != len(labels):
#         raise Exception('Length of samples and labels must equal')
#     stepStart = 0  # initial step
#     i = 0
#     while i < iteration_steps:
#         stepStart = (i * chunkSize) % (labels.shape[0] - chunkSize)
#         yield i, samples[stepStart:stepStart + chunkSize], labels[stepStart:stepStart + chunkSize]
#         i += 1

def train_data_iterator(samples, labels, iteration_steps, chunkSize):
    """
    Iterator/Generator: get a batch of data
    for loop， just like range() function
    """
    if len(samples) != len(labels):
        raise Exception('Length of samples and labels must equal')
    # reshuffle before each epoch to eliminate cyclic behavior
    from random import randint
    reshuffle = True
    stepStart = 0  # initial step
    lastStart = 0  # last step
    i = 0
    while i < iteration_steps:
        lastStart = stepStart
        stepStart = (i * chunkSize) % (labels.shape[0] - chunkSize)
        if reshuffle and stepStart < lastStart:
            for n in range(0 ,len(samples)):
                tmp = randint(n ,len(samples ) -1)
                tmp_image = samples[n]
                samples[n] = samples[tmp]
                samples[tmp] = tmp_image
                tmp_label = labels[n]
                labels[n] = labels[tmp]
                labels[tmp] = tmp_label
        yield i, samples[stepStart:stepStart + chunkSize], labels[stepStart:stepStart + chunkSize]
        i += 1

def test_data_iterator(samples, labels, chunkSize):
    """
    Iterator/Generator: get a batch of data
    for loop， just like range() function
    """
    if len(samples) != len(labels):
        raise Exception('Length of samples and labels must equal')
    stepStart = 0  # initial step
    i = 0
    while stepStart < len(samples):
        stepEnd = stepStart + chunkSize
        if stepEnd < len(samples):
            yield i, samples[stepStart:stepEnd], labels[stepStart:stepEnd]
            i += 1
        stepStart = stepEnd


def define_CNN():

    image_size = load.image_size
    num_labels = load.num_labels
    num_channels = load.num_channels
    #print(num_labels)
    net = Network(
        train_batch_size=64, test_batch_size=500, pooling_scale=2,
        dropout_rate=0.5,
        base_learning_rate=0.001, decay_rate=0.99)
    net.define_inputs(
        train_samples_shape=(64, image_size, image_size, num_channels),
        train_labels_shape=(64, num_labels),
        test_samples_shape=(500, image_size, image_size, num_channels),
    )
    #
    net.add_conv(patch_size=3, in_depth=num_channels, out_depth=32, activation='relu', pooling=False, name='conv1')
    net.add_conv(patch_size=3, in_depth=32, out_depth=32, activation='relu', pooling=True, name='conv2')
    net.add_conv(patch_size=3, in_depth=32, out_depth=32, activation='relu', pooling=False, name='conv3')
    net.add_conv(patch_size=3, in_depth=32, out_depth=32, activation='relu', pooling=True, name='conv4')

    # 4 = 2 pooling,  1/2
    # 32 = conv4 out_depth
    net.add_fc(in_num_nodes=(image_size // 4) * (image_size // 4) * 32, out_num_nodes=128, activation='relu',
               name='fc1')
    net.add_fc(in_num_nodes=128, out_num_nodes=4, activation=None, name='fc2')

    net.define_model()

    return net


if __name__ == '__main__':
    train_samples, train_labels = load.X_train, load.y_train
    test_samples, test_labels = load.X_test, load.y_test

    print('Training set', train_samples.shape, train_labels.shape)
    print('    Test set', test_samples.shape, test_labels.shape)

    cnn = define_CNN()
    # net.run(train_samples, train_labels, test_samples, test_labels, train_data_iterator=train_data_iterator,
    #         iteration_steps=3000, test_data_iterator=test_data_iterator)
    cnn.train(train_samples, train_labels, test_samples, test_labels, data_iterator_train=train_data_iterator, data_iterator_test=test_data_iterator, iteration_steps=1200)
    cnn.test(test_samples, test_labels, data_iterator=test_data_iterator)
    # predict_result = cnn.single_predict(load.test_target)
    # prediction = np.argmax(predict_result)
    # print("The predicted index is: {}".format(prediction))
