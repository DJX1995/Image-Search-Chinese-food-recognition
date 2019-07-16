from scipy.io import loadmat as load
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def reformat(samples, labels, max_label=10):
    # reshape image into (samples, height, width, channels)
    # new = np.transpose(samples, (3, 0, 1, 2)).astype(np.float32)
    new = samples
    # labels one-hot encoding, [2] -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    # digit 0 , represented as 10
    labels = np.array([x[0] for x in labels])  # slow code, whatever
    one_hot_labels = []
    for num in labels:
        one_hot = [0.0] * max_label
        if num == max_label:
            one_hot[0] = 1.0
        else:
            one_hot[num] = 1.0
        one_hot_labels.append(one_hot)
    labels = np.array(one_hot_labels).astype(np.float32)
    return new, labels


def normalize(samples):
    """
    RGB to grey scale
    0 ~ 255 to-1.0 ~ +1.0
    @samples: numpy array
    """
    return samples / 128.0 - 1.0


def distribution(labels, name):
    # keys:
    # 0
    # 1
    # 2
    # ...
    # 9
    count = { }
    for label in labels:
        key = 0 if label[0] == 10 else label[0]
        if key in count:
            count[key] += 1
        else:
            count[key] = 1
    x = []
    y = []
    for k, v in count.items():
        # print(k, v)
        x.append(k)
        y.append(v)

    y_pos = np.arange(len(x))
    plt.bar(y_pos, y, align='center', alpha=0.5)
    plt.xticks(y_pos, x)
    plt.ylabel('Count')
    plt.title(name + ' Label Distribution')
    plt.show()


def inspect(dataset, labels, i):
    # display
    if dataset.shape[3] == 1:
        shape = dataset.shape
        dataset = dataset.reshape(shape[0], shape[1], shape[2])
    print(labels[i])
    plt.imshow(dataset[i])
    plt.show()


# train = load('./data/train_32x32.mat')
# test = load('./data/test_32x32.mat')
data = load('./data/data.mat')

data_samples = data['image']
data_labels = data['label']

n_train_samples, _train_labels = reformat(data_samples, data_labels, 4)
print(n_train_samples.shape, _train_labels.shape)

_train_samples = normalize(n_train_samples)
print(_train_samples.shape, _train_labels.shape)

X_train, X_test, y_train, y_test = train_test_split(_train_samples, _train_labels, test_size=0.20, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

print("display label distribution in training set")
temp = np.argmax(y_train, axis=1)
result = [0] * 4
for i in temp:
    if i == 0:
        result[0] += 1
    elif i == 1:
        result[1] += 1
    elif i == 2:
        result[2] += 1
    else:
        result[3] += 1
print(result)
# print('Train Samples Shape:', train['X'].shape)
# print('Train  Labels Shape:', train['y'].shape)

# print('Train Samples Shape:', test['X'].shape)
# print('Train  Labels Shape:', test['y'].shape)


# train_samples = train['X']
# train_labels = train['y']
# test_samples = test['X']
# test_labels = test['y']


# n_train_samples, _train_labels = reformat(train_samples, train_labels)
# n_test_samples, _test_labels = reformat(test_samples, test_labels)

# _train_samples = normalize(n_train_samples)
# _test_samples = normalize(n_test_samples)

# test_target = _test_samples[1234]
# test_target = test_target.reshape((1,) + test_target.shape)

# print(test_target.shape)

num_labels = 4
image_size = 32
num_channels = 3



if __name__ == '__main__':
    pass
    #inspect(_train_samples, _train_labels, 1234)
# _train_samples = normalize(_train_samples)
# inspect(_train_samples, _train_labels, 1234)
# distribution(train_labels, 'Train Labels')
# distribution(test_labels, 'Test Labels')
