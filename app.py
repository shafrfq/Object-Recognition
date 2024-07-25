import cv2
import numpy as np
import os
import streamlit as st
import tempfile
import subprocess
import gdown

import os
os.system('python -m pip install --upgrade pip')

# Fungsi untuk mengunduh file dari Google Drive
def download_from_google_drive(file_id, destination):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)

# Fungsi untuk mendeteksi objek menggunakan Darknet
def detect_objects_with_darknet(image_path, config_path, weights_path, data_file):
    # Simpan hasil deteksi dalam file output.txt
    output_file = 'output.txt'
    command = f"./darknet detector test {data_file} {config_path} {weights_path} {image_path} -dont_show -out {output_file}"
    
    # Jalankan Darknet
    subprocess.run(command, shell=True)

    # Baca hasil deteksi
    with open(output_file, 'r') as f:
        results = f.read()

    # Mengembalikan hasil deteksi
    return results

# Fungsi utama untuk Streamlit
def main():
    st.title("Deteksi Objek Menggunakan Darknet YOLOv3")
    st.write("Unggah gambar untuk mendeteksi objek menggunakan Darknet")

    # ID file dari Google Drive
    cfg_file_id = '108kFZ9ltANJW7He-Kujzn7f1FYerf2qA'
    weights_file_id = '1-Z_hwylsqXf86t9a8CjvfRgHDs_B_7eh'
    data_file_id = '1iEDj2biLlviApMcAhPIUbcKXFqEHp1_o'

    # Lokasi file yang akan diunduh
    cfg_path = 'yolov3_custom.cfg'
    weights_path = 'yolov3.weights'
    data_file = 'obj.data'

    # Unduh file dari Google Drive
    if not os.path.exists(cfg_path):
        download_from_google_drive(cfg_file_id, cfg_path)
    if not os.path.exists(weights_path):
        download_from_google_drive(weights_file_id, weights_path)
    if not os.path.exists(data_file):
        download_from_google_drive(data_file_id, data_file)

    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Simpan gambar yang diunggah
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        image_path = tfile.name

        # Jalankan deteksi objek
        results = detect_objects_with_darknet(image_path, cfg_path, weights_path, data_file)
        st.text(results)

        # Tampilkan gambar dengan bounding box
        image = cv2.imread(image_path)
        st.image(image, channels="BGR", caption='Gambar dengan Deteksi', use_column_width=True)

if __name__ == "__main__":
    main()
