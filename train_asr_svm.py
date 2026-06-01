import os
import librosa
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler


# =========================================================
# FUNGSI EKSTRAKSI FITUR MFCC
# FIX: Tidak lagi pakai offset=4.5 yang fixed.
#      Sebagai gantinya, trim silence lalu ambil speech inti.
#      Ini konsisten dengan cara kerja test_prediksi.py (realtime).
# =========================================================
def extract_mfcc(file_path, n_mfcc=13, target_duration=1.5, sr_target=16000):

    try:
        # Load full audio dengan sample rate target
        audio, sr = librosa.load(file_path, sr=sr_target)

        # =====================================================
        # FIX 1: Trim silence dulu, bukan pakai offset fixed
        # offset=4.5 sebelumnya bisa kena silence di beberapa file
        # =====================================================
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=25)

        # =====================================================
        # FIX 2: Padding jika audio terlalu pendek
        # =====================================================
        target_samples = int(target_duration * sr_target)
        if len(audio_trimmed) < target_samples:
            # Pad dengan zeros di akhir
            audio_trimmed = np.pad(
                audio_trimmed,
                (0, target_samples - len(audio_trimmed))
            )

        # Potong ke target_duration dari awal speech
        audio_fixed = audio_trimmed[:target_samples]

        # =====================================================
        # NORMALISASI AUDIO
        # =====================================================
        if np.max(np.abs(audio_fixed)) > 0:
            audio_fixed = audio_fixed / np.max(np.abs(audio_fixed))

        # =====================================================
        # DATA AUGMENTATION (NOISE)
        # =====================================================
        noise = np.random.randn(len(audio_fixed)) * 0.003
        audio_noisy = audio_fixed + noise

        # =====================================================
        # FIX 3: Tambah delta MFCC untuk info temporal
        # Hanya mean+std dari 13 MFCC terlalu sedikit fitur
        # =====================================================
        def get_features(y):
            mfccs = librosa.feature.mfcc(y=y, sr=sr_target, n_mfcc=n_mfcc)
            delta = librosa.feature.delta(mfccs)

            mfcc_mean = np.mean(mfccs.T, axis=0)
            mfcc_std  = np.std(mfccs.T, axis=0)
            delta_mean = np.mean(delta.T, axis=0)
            delta_std  = np.std(delta.T, axis=0)

            return np.hstack((mfcc_mean, mfcc_std, delta_mean, delta_std))

        features       = get_features(audio_fixed)
        features_noisy = get_features(audio_noisy)

        return features, features_noisy

    except Exception as e:
        print(f"Error membaca {file_path}: {e}")
        return None, None


# =========================================================
# DATASET
# =========================================================
DATASET_PATH = "dataset"

X = []
y = []

print("=" * 60)
print("PROSES EKSTRAKSI FITUR AUDIO (FIXED VERSION)")
print("=" * 60)

if not os.path.exists(DATASET_PATH):
    print(f"ERROR: Folder '{DATASET_PATH}' tidak ditemukan.")
    exit()

# =========================================================
# MEMBACA DATASET
# =========================================================
for folder_name in sorted(os.listdir(DATASET_PATH)):

    folder_path = os.path.join(DATASET_PATH, folder_name)

    if os.path.isdir(folder_path):

        print(f"\n-> Memproses kota: {folder_name}")

        count_file = 0

        for file_name in os.listdir(folder_path):

            if file_name.lower().endswith(".wav"):

                file_path = os.path.join(folder_path, file_name)

                features, features_noisy = extract_mfcc(file_path)

                if features is not None:

                    # Audio asli
                    X.append(features)
                    y.append(folder_name)

                    # Audio noisy
                    X.append(features_noisy)
                    y.append(folder_name)

                    count_file += 2

        print(f"   Total data masuk: {count_file}")

# =========================================================
# KONVERSI KE NUMPY
# =========================================================
X = np.array(X)
y = np.array(y)

print("\n" + "=" * 60)
print(f"TOTAL DATA : {X.shape[0]}")
print(f"TOTAL FITUR: {X.shape[1]}  (sebelumnya 26, sekarang 52 dengan delta)")
print("=" * 60)

# =========================================================
# STANDARD SCALER
# =========================================================
scaler = StandardScaler()

X = scaler.fit_transform(X)

# Simpan scaler
joblib.dump(scaler, "scaler.pkl")

# =========================================================
# SPLIT DATA
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# TRAINING SVM
# =========================================================
print("\nMelatih model SVM...")

model_svm = SVC(
    kernel='rbf',
    C=10.0,
    gamma='scale',
    probability=True
)

model_svm.fit(X_train, y_train)

print("Training selesai!")

# =========================================================
# EVALUASI
# =========================================================
y_pred = model_svm.predict(X_test)

print("\n" + "=" * 60)
print("HASIL EVALUASI")
print("=" * 60)

print(f"Akurasi: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================================================
# SIMPAN MODEL
# =========================================================
joblib.dump(model_svm, "model_asr_svm.pkl")

print("\nModel berhasil disimpan!")
print("File:")
print("- model_asr_svm.pkl")
print("- scaler.pkl")