# Convert preprocessed images into .mat files

import numpy as np
import glob as gb
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import scipy.io as scio


def image_to_mat(path, label_index=1):
    img_path = gb.glob(path + "/*.jpg")
    print("path: " + path)
    # load the first image to initialize result
    print("load {}".format(img_path[0]))
    img = load_img(img_path[0])
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    result = x.copy()

    # process images starting from index 1
    for path in img_path[1:]:
        print("process image {}".format(path))
        img = load_img(path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        result = np.append(result, x, axis=0)

    print(result.shape)
    labels = np.array([label_index] * result.shape[0])
    labels = labels.reshape(labels.shape + (1,))
    print(labels.shape)

    return result, labels


def check_mat(path):
    data = scio.loadmat(path)
    print(data.keys())
    print(data['image'].shape)
    print(data['label'].shape)


if __name__ == '__main__':
    result0, label0 = image_to_mat("./0", 0)
    result1, label1 = image_to_mat("./1", 1)
    result2, label2 = image_to_mat("./2", 2)
    result3, label3 = image_to_mat("./3", 3)
    result = np.append(result0, result1, axis=0)
    result = np.append(result, result2, axis=0)
    result = np.append(result, result3, axis=0)
    label = np.append(label0, label1, axis=0)
    label = np.append(label, label2, axis=0)
    label = np.append(label, label3, axis=0)
    scio.savemat("./data.mat", {"image": result, "label": label})
    check_mat("data.mat")


