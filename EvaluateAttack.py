import sys

import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import tensorflow.keras as keras
import random as python_random
import os
from PIL import Image

IMG_SIZE = 128
random_seed = 123456

isGood = True


def get_label(file_path):
    # parts = tf.strings.split(file_path, os.path.sep)
    the_good_part = tf.strings.substr(file_path, 0, 5)
    if the_good_part == 'goodw':
        return [0]
    else:
        return [1]


def get_image(path_img):
    f_path = "D:/DexRay/apks/model_eval/goodware/" + path_img.decode('ascii')
    # f_path = "D:/DexRay/apks/model_eval/malware/regular/" + path_img.decode('ascii')
    image = np.asarray(Image.open(f_path))
    image = tf.convert_to_tensor(image, dtype_hint=None, name=None)
    return image


def get_shape(image):
    return image.shape[0]


def decode_img(path_img):
    image = tf.numpy_function(get_image, [path_img], tf.uint8)
    shape = tf.numpy_function(get_shape, [image], tf.int32)
    image = tf.reshape(image, [shape, 1, 1])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMG_SIZE * IMG_SIZE, 1])
    return tf.reshape(image, [IMG_SIZE * IMG_SIZE, 1])


def process_path(file_path):
    label = get_label(file_path)
    img = decode_img(file_path)
    return img, label


def organizeBadData():
    # sanity check lets take 1 benign apk and test our model
    path_to_train_images = "D:/DexRay/apks/model_eval/malware/regular"
    train_images_as_list = os.listdir(path_to_train_images)
    print(train_images_as_list)
    # get the slices of an array in the form of objects
    bad_train_dataset = tf.data.Dataset.from_tensor_slices(train_images_as_list)
    # apply the function to every elecffment of the data set
    bad_train_dataset = bad_train_dataset.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    length_train = len(train_images_as_list)
    batch_train = length_train // 500
    bad_train_dataset = bad_train_dataset.cache()
    bad_train_dataset = bad_train_dataset.shuffle(buffer_size=length_train, seed=random_seed,
                                                  reshuffle_each_iteration=False)
    bad_train_dataset = bad_train_dataset.batch(1)
    bad_train_dataset = bad_train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return bad_train_dataset


def organizeGoodData():
    # sanity check lets take 1 benign apk and test our model
    path_to_train_images = "D:/DexRay/apks/model_eval/goodware/"
    train_images_as_list = os.listdir(path_to_train_images)
    print(train_images_as_list)
    # get the slices of an array in the form of objects
    good_train_dataset = tf.data.Dataset.from_tensor_slices(train_images_as_list)
    # apply the function to every element of the data set
    good_train_dataset = good_train_dataset.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    length_train = len(train_images_as_list)
    batch_train = length_train // 500
    good_train_dataset = good_train_dataset.cache()
    good_train_dataset = good_train_dataset.shuffle(buffer_size=length_train, seed=random_seed,
                                                    reshuffle_each_iteration=False)
    good_train_dataset = good_train_dataset.batch(1)
    good_train_dataset = good_train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return good_train_dataset


# first lets load the model
model = keras.models.load_model(filepath=os.path.join('results_dir', 'model3'),
                                custom_objects={'Precision': tf.keras.metrics.Precision(),
                                                'Recall': tf.keras.metrics.Recall(),
                                                'F1Score': tfa.metrics.F1Score(num_classes=2, average="micro",
                                                                               threshold=0.5)}, compile=False)
g_train_dataset = organizeGoodData()
# b_train_dataset = organizeBadData()

prediction = model.predict(
    g_train_dataset,
    batch_size=None,
    verbose=0,
    steps=None,
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False
)
# goodware = 0, malware = 1
for p in prediction.ravel():
    print(p*10)
    if p*10 < 0.5:
        print("Correct!")
# Now load an APK


path_to_good_eval_apks = "D:/DexRay/apks/model_eval/goodware"
path_to_regular_bad_eval_images = "D:/DexRay/apks/model_eval/malware/regular"
path_to_injected_bad_eval_images = "D:/DexRay/apks/model_eval/malware/code_injected"
