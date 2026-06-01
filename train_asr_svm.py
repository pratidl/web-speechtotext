import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib


# FUNGSI EKSTRAKSI FITUR (MFCC)
def extract_mfcc(file_path, n_mfcc=13):
    """
    Fungsi untuk membaca file .wav, memotong bagian suara (detik 4.5 - 6.0),
    dan mengekstrak fitur MFCC.
    """
    try:
        # offset=4.5: Mengabaikan keheningan di 4.5 detik pertama
        # duration=1.5: Mengambil 1.5 detik sisa (hingga detik ke-6) tempat user berbicara
        audio, sr = librosa.load(file_path, sr=16000, offset=4.5, duration=1.5)
        
        # Ekstraksi MFCC (Menghasilkan matriks [n_mfcc x frame_waktu])
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # Meratakan dimensi waktu dengan mengambil nilai rata-rata (mean)
        # Hasil akhirnya berupa array 1 dimensi dengan panjang 13 (searah jumlah n_mfcc)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print(f"Error membaca {file_path}: {e}")
        return None


# PROSES PEMBACAAN DATASET & LABELING
DATASET_PATH = "dataset" 

X = [] # List untuk menyimpan semua fitur MFCC
y = [] # List untuk menyimpan semua label (nama folder kota)

print("="*60)
print("PROSES EKSTRAKSI FITUR AUDIO (.WAV)")
print("="*60)

# Validasi apakah folder dataset ada
if not os.path.exists(DATASET_PATH):
    print(f"ERROR: Folder '{DATASET_PATH}' tidak ditemukan. Pastikan nama foldernya sesuai.")
    exit()

# Looping membaca isi folder dataset
for folder_name in sorted(os.listdir(DATASET_PATH)):
    folder_path = os.path.join(DATASET_PATH, folder_name)
    
    # Pastikan yang diproses hanya sub-folder (10 folder kota)
    if os.path.isdir(folder_path):
        print(f"-> Memproses folder kota: {folder_name}")
        count_file = 0
        
        for file_name in os.listdir(folder_path):
            # Validasi ketat hanya membaca file berformat .wav
            if file_name.lower().endswith('.wav'):
                file_path = os.path.join(folder_path, file_name)
                
                # Ekstrak fitur MFCC
                features = extract_mfcc(file_path)
                
                if features is not None:
                    X.append(features)
                    y.append(folder_name)
                    count_file += 1
        
        print(f"   Berhasil mengekstrak {count_file} file .wav.")

# Mengubah list Python menjadi Array NumPy agar bisa diproses scikit-learn
X = np.array(X)
y = np.array(y)

print("\n" + "="*60)
print(f"TOTAL DATA SUKSES: {X.shape[0]} Sampel Audio")
print(f"DIMENSI INPUT SVM: {X.shape[1]} Fitur MFCC per sampel")
print("="*60)


# 3. PEMBAGIAN DATA (SPLIT DATASET)
# Membagi data: 80% untuk Training, 20% untuk Testing
# stratify=y memastikan pembagian 10 kelas kota terbagi secara adil dan merata
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==============================================================================
# 4. PELATIHAN MODEL SVM
# ==============================================================================
print("\nMelatih model SVM (Support Vector Machine)...")

# Menggunakan kernel 'rbf' dengan regularisasi C=10.0 agar model lebih optimal
model_svm = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True)
model_svm.fit(X_train, y_train)

print("Training Selesai!")


# EVALUASI MODEL
y_pred = model_svm.predict(X_test)

print("\n" + "="*60)
print("HASIL EVALUASI MODEL SVM")
print("="*60)
print(f"Akurasi Global: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Laporan Klasifikasi Per Kota:")
print(classification_report(y_test, y_pred))

# Menampilkan matriks kebingungan (Confusion Matrix) sederhana di terminal
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("="*60)

# PENYIMPANAN MODEL
# Menyimpan model ke dalam file biner agar bisa digunakan berulang kali
model_filename = 'model_asr_svm.pkl'
joblib.dump(model_svm, model_filename)
print(f"\nModel sukses disimpan dengan nama: '{model_filename}'")
print("Siap digunakan untuk tahap inferensi/uji coba tebak kata!")