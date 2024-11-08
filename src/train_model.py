import datasets
import tensorflow as tf
import numpy as np
import random
from PIL import Image
from tensorflow.keras.preprocessing import image  # type: ignore
from tensorflow.keras.applications import ResNet50  # type: ignore
from tensorflow.keras import layers, models  # type: ignore
import wandb
import tensorflow as tf
import sys

# Memeriksa dan mengatur GPU
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("Tidak ada GPU yang terdeteksi. Program ini memerlukan GPU untuk dijalankan.")
    sys.exit(1)  # Menghentikan program dengan kode status 1

try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("Pengaturan memori GPU berhasil.")
except Exception as e:
    print(f"Kesalahan saat mengatur memori GPU: {e}")
    sys.exit(1)  # Menghentikan program jika terjadi kesalahan saat mengatur GPU

# Fungsi untuk memproses dan melakukan augmentasi pada gambar
def preprocess_and_augment_images(dataset, num_images=0, target_size=(224, 224)):
    train_images = []
    train_labels = []

    for i in range(min(num_images, len(dataset['train']))):
        try:
            img = dataset['train'][i]['image'].convert('RGB').resize(target_size)
            label = dataset['train'][i]['label']
            img_array = np.array(img) / 255.0
            
            # Deteksi blur dan augmentasi gambar
            if np.std(img_array) < 0.05:
                img_array *= 2  # Tingkatkan kontras
            
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

    return np.array(train_images), np.array(train_labels)

# Fungsi untuk membuat model ResNet50
def create_resnet50(input_shape, num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)

    for layer in base_model.layers:
        layer.trainable = False  # Membekukan lapisan dasar

    return model

# Fungsi untuk melatih dan menyimpan model dengan kondisi tertentu
def train_and_save_model(train_images, train_labels, val_images, val_labels, num_classes, epochs=10):
    model = create_resnet50(input_shape=(224, 224, 3), num_classes=num_classes)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_images, train_labels,
                        epochs=epochs,
                        validation_data=(val_images, val_labels))

    # Menyimpan model jika memenuhi syarat akurasi
    if history.history['val_accuracy'][-1] >= 0.80:
        model.save('simpan_resnet50_model.h5')
        print("Model telah disimpan.")
    
    return model

# Fungsi untuk mengevaluasi model
def evaluate_model(model, test_images, test_labels):
    loss, accuracy = model.evaluate(test_images, test_labels)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Fungsi utama untuk menjalankan program
def main():
    wandb.login(key="967fafd5558eeeff3ce5681cf55c71633438428d")  # Ganti dengan kunci API Anda
    # Memuat dataset
    ds = datasets.load_dataset("garythung/trashnet")

    num_images = 5024
    train_images, train_labels = preprocess_and_augment_images(ds, num_images=num_images)

    # Memisahkan data menjadi set pelatihan dan validasi
    split_index = int(0.8 * len(train_images))  # 80% untuk pelatihan
    val_images, val_labels = train_images[split_index:], train_labels[split_index:]  # 20% untuk validasi
    train_images, train_labels = train_images[:split_index], train_labels[:split_index]  # 80% untuk pelat ihan

    # Menentukan jumlah kelas
    num_classes = len(set(train_labels))

    # Mengupdate parameter config berdasarkan dataset
    config = {
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 64,
        "architecture": "ResNet50",
        "dataset": ds.info.description,  # Deskripsi dataset
        "num_classes": num_classes  # Jumlah kelas
    }
    
    wandb.init(project='my-tf-trash', config=config)

    model = train_and_save_model(train_images, train_labels, val_images, val_labels, num_classes) 

if __name__ == "__main__":
    main()