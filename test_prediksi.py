import joblib
import librosa
import numpy as np
import sounddevice as sd
import time

# ==============================================================================
# 1. LOAD MODEL PINTAR YANG SUDAH DITRAINING
# ==============================================================================
MODEL_PATH = 'model_asr_svm.pkl'
try:
    model = joblib.load(MODEL_PATH)
    print(f"-> Sukses memuat model: '{MODEL_PATH}'")
    print(f"-> Kelas kota yang dikenali: {list(model.classes_)}\n")
except Exception as e:
    print(f"ERROR: Gagal memuat model. Pastikan sudah running 'train_asr_svm.py'. {e}")
    exit()

# ==============================================================================
# 2. FUNGSI EKSTRAKSI FITUR REAL-TIME (WAJIB IDENTIK DENGAN TRAINING)
# ==============================================================================
def extract_features_from_buffer(audio_data, sr=16000):
    try:
        # Otomatis buang silence di awal dan akhir rekaman mikrofon
        audio_trimmed, _ = librosa.effects.trim(audio_data, top_db=20)
        
        # Batasi panjang audio maksimal 1.5 detik
        max_samples = int(1.5 * sr)
        audio_fixed = audio_trimmed[:max_samples]
        
        # Ekstrak MFCC (13 koefisien)
        n_mfcc = 13
        mfccs = librosa.feature.mfcc(y=audio_fixed, sr=sr, n_mfcc=n_mfcc)
        
        # Hitung Mean dan Std (Total 26 fitur)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        
        features_combined = np.hstack((mfccs_mean, mfccs_std))
        return features_combined
    except Exception as e:
        print(f"Error pemrosesan audio: {e}")
        return None

# ==============================================================================
# 3. FUNGSI PEREKAMAN MIKROFON & INFERENSI
# ==============================================================================
def rekam_dan_prediksi(durasi=2.0, sr=16000):
    print("\n" + "="*50)
    print("Persiapan... Bersiaplah untuk berbicara.")
    print("="*50)
    
    # Hitung mundur sebelum merekam
    for i in range(3, 0, -1):
        print(f"Mulai dalam {i}...")
        time.sleep(0.7)
        
    print("\n>>> [MIKROFON AKTIF] Silakan ucapkan nama kota sekarang! <<<")
    
    # Proses perekaman dari hardware mikrofon
    # channels=1 artinya mono (standar audio pemrosesan sinyal)
    audio_recorded = sd.rec(int(durasi * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Tunggu sampai durasi rekam 2 detik selesai
    
    print(">>> [MIKROFON MATI] Rekaman selesai, sedang memproses... <<<\n")
    
    # Meratakan array hasil rekaman sounddevice (dari 2D ke 1D)
    audio_data = audio_recorded.flatten()
    
    # Ekstraksi fitur dari suara yang baru direkam
    fitur = extract_features_from_buffer(audio_data, sr=sr)
    
    if fitur is not None:
        # Reshape menjadi [[fitur]] karena SVM menerima input 2D untuk prediksi tunggal
        fitur_input = fitur.reshape(1, -1)
        
        # Jalankan prediksi kelas kota
        prediksi_kota = model.predict(fitur_input)[0]
        
        # Hitung Confidence Score (Skor Keyakinan)
        probabilitas = model.predict_proba(fitur_input)[0]
        confidence_score = np.max(probabilitas) * 100
        
        # Cetak Hasil Akhir ke Terminal
        print("="*50)
        print("               HASIL PREDIKSI REALTIME           ")
        print("="*50)
        print(f"Kata Terdeteksi  : {prediksi_kota.upper()}")
        print(f"Confidence Score : {confidence_score:.2f}%")
        print("="*50)
    else:
        print("Gagal memproses suara. Pastikan mikrofon Anda menangkap suara dengan jelas.")

# ==============================================================================
# 4. LOOP PROGRAM UTAMA
# ==============================================================================
if __name__ == "__main__":
    while True:
        rekam_dan_prediksi(durasi=2.0) # Durasi rekam dikunci 2 detik langsung bicara
        
        # Opsi untuk mencoba kembali atau keluar
        pilihan = input("\nKetik 'y' untuk mencoba lagi, atau tombol lain untuk keluar: ")
        if pilihan.lower() != 'y':
            print("Program dihentikan. Terima kasih!")
            break