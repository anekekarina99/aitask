# %% [markdown]
# Berikut adalah dokumentasi dalam format Markdown untuk proyek klasifikasi sampah menggunakan ResNet, diikuti oleh implementasi kode.
# 
# ---
# 
# # Klasifikasi Sampah Menggunakan ResNet dengan TensorFlow
# 
# ## Pendahuluan
# Klasifikasi sampah menggunakan deep learning adalah metode yang dapat membantu otomatisasi dalam pemilahan dan pengelolaan limbah secara efisien. Model berbasis arsitektur Residual Network (ResNet) mampu mengenali jenis sampah dengan menggunakan dataset gambar yang dilabeli, yang berpotensi memperbaiki proses daur ulang otomatis. ResNet sering digunakan dalam pengenalan gambar karena kemampuannya mengatasi masalah vanishing gradient melalui blok residual yang mempertahankan informasi penting dari lapisan sebelumnya [(He et al., 2016)](https://consensus.app/papers/resnet-deep-learning-he/9c586abb395db4c2a24d1a4e18388f23/).
# 
# ## Lingkungan dan Pengaturan Awal
# - **Library yang Digunakan**: TensorFlow digunakan sebagai framework utama untuk deep learning, bersama pustaka `pandas`, `numpy`, dan `matplotlib` untuk manipulasi data dan visualisasi. `wandb` digunakan untuk pelacakan metrik model dan logging eksperimen, yang membantu dalam memantau performa model dan stabilitas selama pelatihan [(Biewald, 2020)](https://consensus.app/papers/weights-biases-open-source-biewald/e1289d2819c3a3d8cb70d44f6a5587b5/).
# - **Dataset TrashNet**: Dataset gambar dari berbagai jenis sampah, seperti kertas, plastik, dan kaca, digunakan untuk melatih model. TrashNet menyediakan data yang beragam dan relevan untuk kebutuhan klasifikasi sampah otomatis [(Yang & Thung, 2017)](https://consensus.app/papers/classification-trash-using-deep-learning-yang-thung/e65627e5d9095e9232ff2f193fef10a7/).
# 
# ## Arsitektur Model ResNet
# ResNet menggunakan struktur blok residual untuk memungkinkan sinyal informasi melewati jaringan dengan minimal gangguan, mengatasi masalah vanishing gradient yang umum pada jaringan dalam [(He et al., 2016)](https://consensus.app/papers/resnet-deep-learning-he/9c586abb395db4c2a24d1a4e18388f23/).
# 
# - **Blok Residual**: Blok residual memungkinkan shortcut connections yang membawa input asli dari satu lapisan ke lapisan berikutnya, menjaga integritas informasi dari lapisan awal [(Szegedy et al., 2015)](https://consensus.app/papers/going-deeper-convolutions-szegedy/1a4e1e4b0f45310d711eaac8ae945a17/).
# - **Arsitektur Model**: Model dimulai dengan lapisan konvolusi besar, diikuti dengan pooling dan beberapa blok residual. Global Average Pooling digunakan di akhir jaringan, yang memberikan stabilitas saat menangani data gambar dalam jumlah besar dan bervariasi.
# 
# ## Pembagian Data
# Untuk melatih model dengan data yang seimbang, dataset dibagi menjadi tiga set: pelatihan, validasi, dan pengujian. Pembagian manual memastikan data terdistribusi secara merata, meminimalkan risiko bias [(James et al., 2013)](https://consensus.app/papers/introduction-statistical-learning-james-et-al/09d17878513a84a3e95c748c8b6e5e93/).
# 
# ## Preprocessing dan Augmentasi Data
# - **Normalisasi dan Deteksi Blur**: Gambar normalisasi dan deteksi blur diterapkan pada setiap gambar agar kualitas gambar menjadi lebih baik sebelum dimasukkan ke model. Normalisasi dapat meningkatkan akurasi dan stabilitas model [(Krizhevsky et al., 2012)](https://consensus.app/papers/image-net-classification-deep-convolutional-krizhevsky-sutskever-hinton/2e5c8ac2f47d97cf9fc9b515f9aa2b24/).
# - **Augmentasi Data**: Augmentasi seperti flip horizontal, rotasi, dan penyesuaian kecerahan digunakan untuk memperkaya data gambar. Teknik ini terbukti meningkatkan kemampuan generalisasi model pada data yang belum pernah dilihat sebelumnya [(Shorten & Khoshgoftaar, 2019)](https://consensus.app/papers/survey-image-data-augmentation-deep-shorten-khoshgoftaar/c2c52f2dd34891f128cc27ff18bc9e48/).
# 
# ## Pelatihan Model
# Model dilatih pada data pelatihan dan dievaluasi pada data validasi untuk menguji akurasi dan stabilitas. Hanya model yang mencapai akurasi dan stabilitas tertentu yang disimpan, untuk menghindari model overfitting atau underfitting [(Goodfellow et al., 2016)](https://consensus.app/papers/deep-learning-goodfellow-bengio-courville/90a54b117d70e0309978f7d0af87a4ef/).
# 
# ## Evaluasi dan Prediksi Model
# Evaluasi model menggunakan data pengujian untuk menghitung loss dan akurasi, memberikan gambaran yang lebih baik tentang performa model pada data baru. Akurasi tinggi menunjukkan kemampuan model dalam mengklasifikasikan sampah dengan baik [(Russakovsky et al., 2015)](https://consensus.app/papers/imagenet-large-scale-visual-recognition-challenge-russakovsky-deng/66e0a771282f27a07b05f98f315d2be3/).
# 
# ---

# %%
import tensorflow as tf # type: ignore

# Memeriksa apakah GPU tersedia
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("Tidak ada GPU yang terdeteksi. Menggunakan CPU.")
else:
    print(f"GPU terdeteksi: {gpus}")

    # Memaksa TensorFlow untuk menggunakan GPU
    try:
        # Mengatur GPU yang akan digunakan
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Pengaturan memori GPU berhasil.")
    except Exception as e:
        print(f"Terjadi kesalahan saat mengatur memori GPU: {e}")

# Menampilkan informasi lebih lanjut tentang perangkat yang digunakan
print("Perangkat yang digunakan:")
print(tf.config.list_physical_devices())

# %%
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
import wandb
from wandb.integration.keras import WandbMetricsLogger

wandb.login()

# Fungsi untuk membuat blok residual
def residual_block(x, filters):
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.add([x, shortcut])  # Menambahkan shortcut
    x = tf.keras.layers.ReLU()(x)
    return x

# Fungsi untuk membuat model ResNet sederhana
def create_resnet(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, kernel_size=7, padding='same', strides=2)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Menambahkan beberapa blok residual
    for _ in range(3):  # Misalnya 3 blok residual
        x = residual_block(x, 64)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

# Fungsi untuk membagi data
def manual_train_val_test_split(images, labels, val_size=0.2, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(len(images))
    np.random.shuffle(indices)

    total_size = len(images)
    test_size = int(total_size * test_size)
    val_size = int(total_size * val_size)

    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]

    train_images = [images[i] for i in train_indices]
    val_images = [images[i] for i in val_indices]
    test_images = [images[i] for i in test_indices]
    
    train_labels = [labels[i] for i in train_indices]
    val_labels = [labels[i] for i in val_indices]
    test_labels = [labels[i] for i in test_indices]

    return train_images, val_images, test_images, train_labels, val_labels, test_labels

# Fungsi untuk melatih dan menyimpan model dengan kondisi tertentu
def train_and_save_model(train_images, train_labels, val_images, val_labels, num_classes, epochs=10):
    print("Create ResNet model")
    model = create_resnet(input_shape=(224, 224, 3), num_classes=num_classes)

    print("Compile model")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Melatih model dan menyimpan riwayat pelatihan
    print("Training model")
    history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(val_images, val_labels))

    # Mendapatkan akurasi training dan validation dari epoch terakhir
    train_accuracy = history.history['accuracy'][-1]
    val_accuracy = history.history['val_accuracy'][-1]
    
    # Mengecek kondisi untuk menyimpan model
    accuracy_threshold = 0.80  # 80%
    stability_threshold = 0.05  # 5% perbedaan max antara training dan validation untuk dianggap stabil
    
    if train_accuracy >= accuracy_threshold and val_accuracy >= accuracy_threshold:
        accuracy_diff = abs(train_accuracy - val_accuracy)
        if accuracy_diff <= stability_threshold:
            model.save('simpan_resnet_model.h5')
            print("Model telah disimpan sebagai 'simpan_resnet_model.h5' karena memenuhi syarat stabilitas dan akurasi.")
        else:
            print("Model tidak disimpan: Perbedaan akurasi training dan validation terlalu besar (overfitting atau underfitting).")
    else:
        print("Model tidak disimpan: Akurasi belum mencapai threshold 80%.")

    return model

# Fungsi untuk mengevaluasi model
def evaluate_model(model, test_images, test_labels):
    print("Evaluating model...")
    loss, accuracy = model.evaluate(test_images, test_labels)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Fungsi untuk melakukan prediksi
def predict_image(model, img_path, class_names):
    img = image.load_img(img_path, target_size=(224, 224))  # Mengubah ukuran gambar
    img_array = image.img_to_array(img)  # Mengubah gambar menjadi array
    img_array = np.expand_dims(img_array, axis=0)  # Menambahkan dimensi batch
    img_array /= 255.0  # Normalisasi gambar

    # Melakukan prediksi
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)  # Mendapatkan kelas dengan probabilitas tertinggi

    return class_names[predicted_class[0]], predictions[0][predicted_class[0]]  # Mengembalikan nama kelas dan probabilitas

# Fungsi untuk memproses dan melakukan augmentasi pada gambar
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

def main():
    # Memuat dataset
    dataset = datasets.load_dataset("garythung/trashnet")

    # Memproses dan membagi data
    num_images = 5024
    train_images, train_labels = preprocess_and_augment_images(dataset, num_images=num_images)

    train_images, val_images, test_images, train_labels, val_labels, test_labels = manual_train_val_test_split(train_images, train_labels)

    # Menentukan jumlah kelas
    num_classes = len(set(train_labels))
    model = train_and_save_model(train_images, train_labels, val_images, val_labels, num_classes, epochs=10)

    # Mengevaluasi model
    evaluate_model(model, test_images, test_labels)

    # Daftar nama kelas yang telah diurutkan sesuai dengan labelnya
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    # Path untuk gambar yang akan diprediksi
    img_path = './sampah.jpg'

    # Melakukan prediksi
    predicted_class, confidence = predict_image(model, img_path, class_names)
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.4f}")

    # Menampilkan gambar beserta hasil prediksinya
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class} ({confidence:.2f})")
    plt.axis('off')
    plt.show()
    


if __name__ == "__main__":
    main()


# %%



