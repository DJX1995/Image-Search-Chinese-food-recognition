# Data Augmentaion using keras
import numpy as np
import glob as gb
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.1,
    fill_mode='constant')

img_path = gb.glob("./resultscang/*.jpg")

index = 0

# parameter "save_to_dir" is the target folder
for path in img_path:
    print("load {}".format(path))
    img = load_img(path)
    #reshape to 32 x 32
    img = img.resize((32, 32))

    try:
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)  # reshape to (1, channel, width, height)

        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir='results', save_prefix='MapoTofu{}'.format(index),
                                  save_format='jpg'):
            i += 1
            if i > 5:  # generate 5 images
                break
        index += 1
    except:
        pass

img_path2 = gb.glob("./results/*.jpg")
print(len(img_path2))










