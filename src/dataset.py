from keras_preprocessing.text import Tokenizer
from src.preprocess import Preprocess
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import json
import pathlib
import random
from collections import defaultdict
from src.utils import load_image
from src.preprocess import image_features_extract_model
from tqdm import tqdm

class ImageCaptionDataset:
    def __init__(self, images_url, annotations_url) -> None:
        self.images_url = images_url
        self.annotations_url = annotations_url

    def _fetch_dataset(self) -> None:
        annotation_folder = "/annotations/"
        if not os.path.exists(os.path.abspath(".") + annotation_folder):
            annotation_zip = tf.keras.utils.get_file(
                "captions.zip",
                cache_subdir=os.path.abspath("."),
                origin=self.annotations_url,
                extract=True,
            )
            annotation_file = (
                os.path.dirname(annotation_zip) + "/annotations/captions_train2014.json"
            )
            os.remove(annotation_zip)
        else:
            annotation_file = (
                os.path.abspath(".") + "/annotations/captions_train2014.json"
            )
        # Download image files
        image_folder = "/train2014/"
        if not os.path.exists(os.path.abspath(".") + image_folder):
            image_zip = tf.keras.utils.get_file(
                "train2014.zip",
                cache_subdir=os.path.abspath("."),
                origin=self.images_url,
                extract=True,
            )
            image_path = os.path.dirname(image_zip) + image_folder
            os.remove(image_zip)
        else:
            image_path = os.path.abspath(".") + image_folder
        return annotation_file, image_path

    def load_dataset(self, annotation_file, base_image_path, limit_size=6000):
        with open(annotation_file, "r") as f:
            annotations = json.load(f)
        image_path_to_caption = defaultdict(list)
        for val in annotations["annotations"]:
            caption = f"<start> {val['caption']} <end>"
            image_path = (
                base_image_path + "COCO_train2014_" + "%012d.jpg" % (val["image_id"])
            )
            image_path_to_caption[image_path].append(caption)
        image_paths = list(image_path_to_caption.keys())
        random.shuffle(image_paths)
        train_image_paths = image_paths[:limit_size]
        train_captions = []
        img_name_vector = []

        for image_path in train_image_paths:
            caption_list = image_path_to_caption[image_path]
            train_captions.extend(caption_list)
            img_name_vector.extend([image_path] * len(caption_list))

        return train_captions, img_name_vector

    def split_dataset(self, img_name_vector, cap_vector):

        img_to_cap_vector = defaultdict(list)
        for img, cap in zip(img_name_vector, cap_vector):
            img_to_cap_vector[img].append(cap)

        # Create training and validation sets using an 80-20 split randomly.
        img_keys = list(img_to_cap_vector.keys())
        random.shuffle(img_keys)

        slice_index = int(len(img_keys) * 0.8)
        img_name_train_keys, img_name_val_keys = (
            img_keys[:slice_index],
            img_keys[slice_index:],
        )

        img_name_train = []
        cap_train = []
        for imgt in img_name_train_keys:
            capt_len = len(img_to_cap_vector[imgt])
            img_name_train.extend([imgt] * capt_len)
            cap_train.extend(img_to_cap_vector[imgt])

        img_name_val = []
        cap_val = []
        for imgv in img_name_val_keys:
            capv_len = len(img_to_cap_vector[imgv])
            img_name_val.extend([imgv] * capv_len)
            cap_val.extend(img_to_cap_vector[imgv])
        return img_name_train, cap_train, img_name_val, cap_val

    def create_dataset(self, img_train, cap_train, buffer_size=1000, batch_size=64):

        # Load the numpy files
        def map_func(img_name, cap):
            img_tensor = np.load(img_name.decode("utf-8") + ".npy")
            return img_tensor, cap

        dataset = tf.data.Dataset.from_tensor_slices((img_train, cap_train))

        # Use map to load the numpy files in parallel
        dataset = dataset.map(
            lambda item1, item2: tf.numpy_function(
                map_func, [item1, item2], [tf.float32, tf.int32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Shuffle and batch
        dataset = dataset.shuffle(buffer_size).batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def preprocess_features(self, img_name_vector):
        # Get unique images
        encode_train = sorted(set(img_name_vector))

        # Feel free to change batch_size according to your system configuration
        image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
        image_dataset = image_dataset.map(
        load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(16)

        for img, path in tqdm(image_dataset):
            batch_features = image_features_extract_model(img)
            batch_features = tf.reshape(batch_features,
                                        (batch_features.shape[0], -1, batch_features.shape[3]))

        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())

    def prepare_data(self, limit_size, buffer_size, batch_size):
        annotation_file, image_path = self._fetch_dataset()
        train_captions, img_name_vector = self.load_dataset(
            annotation_file, image_path, limit_size=limit_size
        )
        self.preprocess_features(img_name_vector)
        cap_vector, max_length, tokenizer = Preprocess.tokenize(train_captions)
        img_name_train, cap_train, img_name_val, cap_val = self.split_dataset(
            img_name_vector, cap_vector
        )
        train_dataset = self.create_dataset(
            img_name_train, cap_train, buffer_size, batch_size
        )
        val_dataset = self.create_dataset(
            img_name_val, cap_val, buffer_size, batch_size
        )
        return train_dataset, val_dataset, max_length, tokenizer
