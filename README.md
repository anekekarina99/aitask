Tentu, berikut adalah penjelasan langkah-langkah lengkap untuk menjalankan proyek ini, termasuk memasang semua *requirements* sebelum menjalankan file pertama (`Trash_Classification.ipynb`).

### Langkah-langkah Menjalankan Proyek

#### 1. **Menginstal Dependencies (Requirements)**
   Sebelum menjalankan `Trash_Classification.ipynb`, pastikan Anda menginstal semua *dependencies* yang diperlukan. Tambahkan semua *dependencies* dalam file `requirements.txt`, lalu jalankan perintah berikut untuk menginstalnya:

   ```bash
   pip install -r requirements.txt
   ```

   **Isi `requirements.txt` mungkin termasuk:**
   ```plaintext
   tensorflow
   numpy
   pandas
   matplotlib
   wandb
   huggingface_hub
   datasets
   pillow
   ```

   Pastikan semua *libraries* yang tercantum di atas telah terinstal, karena masing-masing diperlukan dalam notebook.

#### 2. **Menjalankan File `Trash_Classification.ipynb` untuk Training Model**

   Setelah dependencies terpasang, berikut langkah-langkah untuk menjalankan file ini:
   
   - **Buka Notebook**: Buka Jupyter Notebook di lokasi `/workspaces/aitask/notebooks/Trash_Classification.ipynb`.
   - **Login ke W&B**: Di dalam notebook, pastikan Anda telah memasukkan kunci API W&B yang benar pada baris `wandb.login(key="967fafd5558eeeff3ce5681cf55c71633438428d")`.
   - **Jalankan Semua Cell**: Eksekusi cell secara berurutan untuk memuat dataset, melakukan preprocessing dan augmentasi gambar, lalu melatih model menggunakan ResNet50.
   - **Simpan Model**: Model akan otomatis disimpan sebagai `simpan_resnet50_model.h5` di akhir proses jika akurasi validasi mencapai minimal 80%.

