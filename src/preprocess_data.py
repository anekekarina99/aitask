# Data preprocessing and augmentation script
# Fungsi untuk memproses dan melakukan augmentasi pada gambar
# Kode program klasifikasi sampah menggunakan TensorFlow dan ResNet
import tensorflow as tf  # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import datasets  # type: ignore
import wandb  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.decomposition import PCA # type: ignore
from tensorflow.keras import layers, models  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore # Untuk memproses gambar
import random
from PIL import Image, ImageFilter # type: ignore

def preprocess_and_augment_images(dataset, num_images=0, target_size=(224, 224), apply_blur=True):
    train_images = []
    train_labels = []

    for i in range(min(num_images, len(dataset['train']))):  # Loop sebanyak num_images atau jumlah gambar yang ada
        try:
            img = dataset['train'][i]['image'].convert('RGB').resize(target_size)
            label = dataset['train'][i]['label']
            
            img_array = np.array(img) / 255.0  # Normalisasi Min-Max

            # Deteksi blur
            if np.std(img_array) < 0.05:
                img_array = np.clip(img_array * 2, 0, 1)  # Tingkatkan kontras

            if apply_blur:
                img = img.filter(ImageFilter.GaussianBlur(radius=1))
                img_array = np.array(img) / 255.0

            augment_type = random.choice(['flip', 'rotate', 'brightness'])
            if augment_type == 'flip':
                img_array = np.fliplr(img_array)
            elif augment_type == 'rotate':
                img_array = np.rot90(img_array)
            elif augment_type == 'brightness':
                factor = random.uniform(0.5, 1.5)
                img_array = np.clip(img_array * factor, 0, 1)

            train_images.append(img_array)
            train_labels.append(label)
        except Exception as e:
            print(f"Error processing image {i + 1}: {e}")

    return train_images, train_labels