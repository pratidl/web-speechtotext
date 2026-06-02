import joblib
import librosa
import numpy as np
import sounddevice as sd
import time


# =========================================================
# LOAD MODEL
# =========================================================
MODEL_PATH  = 'model_asr_svm.pkl'
SCALER_PATH = 'scaler.pkl'

try:
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print(f"-> Model berhasil dimuat")
    print(f"-> Kelas kota: {list(model.classes_)}")

except Exception as e:
    print(f"ERROR load model: {e}")
    exit()


# =========================================================
# EKSTRAKSI FITUR REALTIME
# =========================================================
def extract_features_from_buffer(audio_data, sr=16000, n_mfcc=13, target_duration=1.5):

    try:
        # =====================================================
        # TRIM SILENCE
        # =====================================================
        audio_trimmed, _ = librosa.effects.trim(
            audio_data,
            top_db=25          # FIX: sama dengan training (25 bukan 20)
        )

        # =====================================================
        # PADDING
        # =====================================================
        target_samples = int(target_duration * sr)
        if len(audio_trimmed) < target_samples:
            audio_trimmed = np.pad(
                audio_trimmed,
                (0, target_samples - len(audio_trimmed))
            )

        # Potong ke target_duration
        audio_fixed = audio_trimmed[:target_samples]

        # =====================================================
        # NORMALISASI AUDIO
        # =====================================================
        if np.max(np.abs(audio_fixed)) > 0:
            audio_fixed = audio_fixed / np.max(np.abs(audio_fixed))

        # =====================================================
        # MFCC + Delta
        # =====================================================
        mfccs  = librosa.feature.mfcc(y=audio_fixed, sr=sr, n_mfcc=n_mfcc)
        delta  = librosa.feature.delta(mfccs)

        mfcc_mean  = np.mean(mfccs.T, axis=0)
        mfcc_std   = np.std(mfccs.T, axis=0)
        delta_mean = np.mean(delta.T, axis=0)
        delta_std  = np.std(delta.T, axis=0)

        features = np.hstack((mfcc_mean, mfcc_std, delta_mean, delta_std))

        return features

    except Exception as e:
        print(f"Error ekstraksi fitur: {e}")
        return None


# =========================================================
# REKAM & PREDIKSI
# Durasi rekam dikurangi ke 4.0 detik
# (3 detik rekam + 1 detik buffer = cukup untuk 1.5 detik speech)
# =========================================================
def rekam_dan_prediksi(durasi=4.0, sr=16000):

    print("\n" + "=" * 60)
    print("Persiapan bicara...")
    print("=" * 60)

    for i in range(3, 0, -1):
        print(f"Mulai dalam {i}...")
        time.sleep(1)

    print("\n>>> SILAKAN UCAPKAN NAMA KOTA <<<")

    # =====================================================
    # REKAM SUARA
    # =====================================================
    audio_recorded = sd.rec(
        int(durasi * sr),
        samplerate=sr,
        channels=1,
        dtype='float32'
    )

    sd.wait()

    print("Memproses audio...\n")

    # =====================================================
    # FLATTEN AUDIO
    # =====================================================
    audio_data = audio_recorded.flatten()

    # =====================================================
    # Energy check
    # FIX: tambah validasi agar tidak prediksi saat sunyi
    # =====================================================
    if np.max(np.abs(audio_data)) < 0.01:
        print("Suara terlalu pelan atau tidak terdeteksi. Coba lagi.")
        return

    # =====================================================
    # EKSTRAKSI FITUR
    # =====================================================
    fitur = extract_features_from_buffer(audio_data, sr)

    if fitur is not None:

        fitur_input = fitur.reshape(1, -1)

        # =================================================
        # SCALING
        # =================================================
        fitur_input = scaler.transform(fitur_input)

        # =================================================
        # PREDIKSI
        # =================================================
        prediksi_kota = model.predict(fitur_input)[0]

        probabilitas = model.predict_proba(fitur_input)[0]
        confidence   = np.max(probabilitas) * 100

        # =================================================
        # DEBUG
        # =================================================
        print("=" * 60)
        print("HASIL PREDIKSI")
        print("=" * 60)

        print(f"Kota Terdeteksi : {prediksi_kota}")
        print(f"Confidence      : {confidence:.2f}%")

        # Tampilkan top-3 kandidat
        sorted_idx = np.argsort(probabilitas)[::-1]
        print("\nTop-3 Kandidat:")
        for i in range(min(3, len(model.classes_))):
            idx = sorted_idx[i]
            print(f"  {i+1}. {model.classes_[idx]:12s}: {probabilitas[idx]*100:.1f}%")

        print("=" * 60)

    else:
        print("Gagal memproses suara.")


# =========================================================
# LOOP PROGRAM
# =========================================================
if __name__ == "__main__":

    while True:

        rekam_dan_prediksi(durasi=4.0)

        pilihan = input(
            "\nKetik 'y' untuk mencoba lagi: "
        )

        if pilihan.lower() != 'y':
            print("Program selesai.")
            break