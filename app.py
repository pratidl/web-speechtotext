import customtkinter as ctk
from tkinter import messagebox
import asyncio
import edge_tts
import os
import threading
from datetime import datetime

ctk.set_appearance_mode("light")   
ctk.set_default_color_theme("blue")

VOICE_MALE = "id-ID-ArdiNeural"
VOICE_FEMALE = "id-ID-GadisNeural"

last_output_file = None


def get_rate(speed_choice):
    speed_map = {
        "Slow": "-30%",
        "Normal": "+0%",
        "Fast": "+30%"
    }
    return speed_map.get(speed_choice, "+0%")


def get_voice(gender_choice):
    if gender_choice == "Laki-laki":
        return VOICE_MALE
    return VOICE_FEMALE


async def generate_tts(text, voice, rate, output_file):
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate
    )
    await communicate.save(output_file)


def update_status(message, color="#1f6aa5"):
    status_label.configure(text=message, text_color=color)


def update_char_count(event=None):
    text = text_input.get("1.0", "end").strip()
    char_count_label.configure(text=f"Jumlah karakter: {len(text)}")


def create_output_folder():
    folder_name = "output_audio"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


def run_tts_process():
    global last_output_file

    text = text_input.get("1.0", "end").strip()
    gender = gender_var.get()
    speed = speed_var.get()

    if not text:
        messagebox.showwarning("Peringatan", "Masukkan teks terlebih dahulu!")
        return

    generate_button.configure(state="disabled")
    update_status("Sedang membuat audio...", "orange")

    voice = get_voice(gender)
    rate = get_rate(speed)

    folder = create_output_folder()
    filename = f"hasil_tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
    output_path = os.path.join(folder, filename)

    try:
        asyncio.run(generate_tts(text, voice, rate, output_path))
        last_output_file = output_path

        app.after(0, lambda: update_status(f"Audio berhasil dibuat: {filename}", "green"))
        app.after(0, lambda: output_file_label.configure(text=output_path))
        app.after(0, lambda: messagebox.showinfo("Berhasil", f"Audio berhasil disimpan:\n{output_path}"))
    except Exception as e:
        app.after(0, lambda: update_status("Gagal membuat audio.", "red"))
        app.after(0, lambda: messagebox.showerror("Error", f"Gagal membuat audio:\n{e}"))
    finally:
        app.after(0, lambda: generate_button.configure(state="normal"))


def convert_text_to_speech():
    threading.Thread(target=run_tts_process, daemon=True).start()


def play_audio():
    global last_output_file

    if not last_output_file:
        messagebox.showwarning("Peringatan", "Belum ada audio yang dibuat!")
        return

    if os.path.exists(last_output_file):
        try:
            os.startfile(last_output_file)  # untuk Windows
        except Exception as e:
            messagebox.showerror("Error", f"Tidak bisa membuka audio:\n{e}")
    else:
        messagebox.showerror("Error", "File audio tidak ditemukan!")


def open_output_folder():
    folder = create_output_folder()
    try:
        os.startfile(folder)  # untuk Windows
    except Exception as e:
        messagebox.showerror("Error", f"Tidak bisa membuka folder:\n{e}")


def clear_text():
    text_input.delete("1.0", "end")
    update_char_count()
    update_status("Teks dibersihkan.", "#555555")


def change_theme(choice):
    ctk.set_appearance_mode(choice.lower())


# =========================
# MEMBUAT WINDOW
# =========================
app = ctk.CTk()
app.title("Text to Speech Indonesia")
app.geometry("1100x650")
app.minsize(1000, 600)

# layout utama
app.grid_columnconfigure(0, weight=0)
app.grid_columnconfigure(1, weight=1)
app.grid_rowconfigure(0, weight=1)

# =========================
# SIDEBAR KIRI
# =========================
sidebar = ctk.CTkFrame(app, width=260, corner_radius=0)
sidebar.grid(row=0, column=0, sticky="nsw")
sidebar.grid_rowconfigure(8, weight=1)

logo_label = ctk.CTkLabel(
    sidebar,
    text="🔊 TTS Studio",
    font=ctk.CTkFont(size=24, weight="bold")
)
logo_label.grid(row=0, column=0, padx=25, pady=(30, 10), sticky="w")

desc_label = ctk.CTkLabel(
    sidebar,
    text="Aplikasi Text to Speech\nBahasa Indonesia\n\nModern • Cepat • Profesional",
    justify="left",
    font=ctk.CTkFont(size=14)
)
desc_label.grid(row=1, column=0, padx=25, pady=(0, 20), sticky="w")

feature_title = ctk.CTkLabel(
    sidebar,
    text="Fitur Utama",
    font=ctk.CTkFont(size=16, weight="bold")
)
feature_title.grid(row=2, column=0, padx=25, pady=(10, 5), sticky="w")

feature_text = ctk.CTkLabel(
    sidebar,
    text="• Input teks\n• Bahasa Indonesia\n• Simpan MP3\n• Atur kecepatan\n• Pilihan gender suara\n• Putar hasil audio",
    justify="left",
    font=ctk.CTkFont(size=13)
)
feature_text.grid(row=3, column=0, padx=25, pady=(0, 20), sticky="w")

theme_label = ctk.CTkLabel(
    sidebar,
    text="Mode Tampilan",
    font=ctk.CTkFont(size=14, weight="bold")
)
theme_label.grid(row=4, column=0, padx=25, pady=(10, 5), sticky="w")

theme_option = ctk.CTkOptionMenu(
    sidebar,
    values=["Light", "Dark", "System"],
    command=change_theme,
    width=180
)
theme_option.set("Light")
theme_option.grid(row=5, column=0, padx=25, pady=(0, 15), sticky="w")

open_folder_button = ctk.CTkButton(
    sidebar,
    text="Buka Folder Output",
    command=open_output_folder,
    width=180,
    height=40
)
open_folder_button.grid(row=6, column=0, padx=25, pady=10, sticky="w")

footer_label = ctk.CTkLabel(
    sidebar,
    text="Developed with Python + Edge TTS",
    font=ctk.CTkFont(size=12),
    text_color="gray"
)
footer_label.grid(row=9, column=0, padx=25, pady=20, sticky="sw")

# =========================
# AREA UTAMA
# =========================
main_frame = ctk.CTkFrame(app, corner_radius=0, fg_color="transparent")
main_frame.grid(row=0, column=1, sticky="nsew", padx=25, pady=25)
main_frame.grid_columnconfigure(0, weight=1)
main_frame.grid_rowconfigure(2, weight=1)

header_label = ctk.CTkLabel(
    main_frame,
    text="Modul Text to Speech Bahasa Indonesia",
    font=ctk.CTkFont(size=28, weight="bold")
)
header_label.grid(row=0, column=0, sticky="w", pady=(0, 5))

subheader_label = ctk.CTkLabel(
    main_frame,
    text="Masukkan teks, pilih suara, atur kecepatan, lalu ubah menjadi audio MP3.",
    font=ctk.CTkFont(size=14),
    text_color="gray"
)
subheader_label.grid(row=1, column=0, sticky="w", pady=(0, 20))

# card input
input_card = ctk.CTkFrame(main_frame, corner_radius=16)
input_card.grid(row=2, column=0, sticky="nsew", pady=(0, 20))
input_card.grid_columnconfigure(0, weight=1)
input_card.grid_rowconfigure(1, weight=1)

input_title = ctk.CTkLabel(
    input_card,
    text="Input Teks",
    font=ctk.CTkFont(size=18, weight="bold")
)
input_title.grid(row=0, column=0, sticky="w", padx=20, pady=(20, 10))

text_input = ctk.CTkTextbox(
    input_card,
    height=250,
    corner_radius=12,
    font=ctk.CTkFont(size=14)
)
text_input.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 10))
text_input.insert("1.0", "Tulis teks Bahasa Indonesia di sini...")
text_input.bind("<KeyRelease>", update_char_count)

char_count_label = ctk.CTkLabel(
    input_card,
    text="Jumlah karakter: 0",
    font=ctk.CTkFont(size=12),
    text_color="gray"
)
char_count_label.grid(row=2, column=0, sticky="e", padx=20, pady=(0, 15))

# opsi
options_frame = ctk.CTkFrame(main_frame, corner_radius=16)
options_frame.grid(row=3, column=0, sticky="ew", pady=(0, 20))
options_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

gender_label = ctk.CTkLabel(
    options_frame,
    text="Gender Suara",
    font=ctk.CTkFont(size=14, weight="bold")
)
gender_label.grid(row=0, column=0, padx=20, pady=(20, 5), sticky="w")

gender_var = ctk.StringVar(value="Perempuan")
gender_menu = ctk.CTkOptionMenu(
    options_frame,
    values=["Perempuan", "Laki-laki"],
    variable=gender_var,
    height=38
)
gender_menu.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")

speed_label = ctk.CTkLabel(
    options_frame,
    text="Kecepatan Bicara",
    font=ctk.CTkFont(size=14, weight="bold")
)
speed_label.grid(row=0, column=1, padx=20, pady=(20, 5), sticky="w")

speed_var = ctk.StringVar(value="Normal")
speed_menu = ctk.CTkOptionMenu(
    options_frame,
    values=["Slow", "Normal", "Fast"],
    variable=speed_var,
    height=38
)
speed_menu.grid(row=1, column=1, padx=20, pady=(0, 20), sticky="ew")

generate_button = ctk.CTkButton(
    options_frame,
    text="Generate Audio",
    command=convert_text_to_speech,
    height=42,
    font=ctk.CTkFont(size=14, weight="bold")
)
generate_button.grid(row=1, column=2, padx=20, pady=(0, 20), sticky="ew")

play_button = ctk.CTkButton(
    options_frame,
    text="Putar Audio",
    command=play_audio,
    height=42,
    fg_color="#2FA572",
    hover_color="#278a60",
    font=ctk.CTkFont(size=14, weight="bold")
)
play_button.grid(row=1, column=3, padx=20, pady=(0, 20), sticky="ew")

# tombol tambahan
extra_button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
extra_button_frame.grid(row=4, column=0, sticky="w", pady=(0, 10))

clear_button = ctk.CTkButton(
    extra_button_frame,
    text="Bersihkan Teks",
    command=clear_text,
    width=140,
    fg_color="#888888",
    hover_color="#6f6f6f"
)
clear_button.grid(row=0, column=0, padx=(0, 10))

# status
status_card = ctk.CTkFrame(main_frame, corner_radius=16)
status_card.grid(row=5, column=0, sticky="ew")
status_card.grid_columnconfigure(0, weight=1)

status_title = ctk.CTkLabel(
    status_card,
    text="Status",
    font=ctk.CTkFont(size=16, weight="bold")
)
status_title.grid(row=0, column=0, sticky="w", padx=20, pady=(15, 5))

status_label = ctk.CTkLabel(
    status_card,
    text="Siap digunakan.",
    font=ctk.CTkFont(size=13),
    text_color="#1f6aa5"
)
status_label.grid(row=1, column=0, sticky="w", padx=20, pady=(0, 8))

output_file_label = ctk.CTkLabel(
    status_card,
    text="Belum ada file audio yang dihasilkan.",
    font=ctk.CTkFont(size=12),
    text_color="gray",
    wraplength=700,
    justify="left"
)
output_file_label.grid(row=2, column=0, sticky="w", padx=20, pady=(0, 15))

update_char_count()
app.mainloop()