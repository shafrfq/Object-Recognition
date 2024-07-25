import requests
import cv2
import numpy as np
import os
import streamlit as st
import tempfile

def download_from_google_drive(drive_url, destination_dir):
    try:
        response = requests.get(drive_url, stream=True)
        if response.status_code == 200:
            filename = os.path.basename(drive_url.split("?")[0])
            filepath = os.path.join(destination_dir, filename)
            
            # Cek apakah direktori sudah ada, buat jika belum
            os.makedirs(destination_dir, exist_ok=True)
            
            with open(filepath, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            return filepath
        else:
            st.error(f"Gagal mengunduh file: {response.reason}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Fungsi untuk mendeteksi objek menggunakan Darknet
def detect_objects_with_darknet(image_path, config_path, weights_path, data_file):
    output_file = 'output.txt'
    command = f"./darknet detector test {data_file} {config_path} {weights_path} {image_path} -dont_show -out {output_file}"
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"Darknet command failed with return code {result.returncode}")
            st.error(f"stderr: {result.stderr}")
        return result.stdout
    except Exception as e:
        st.error(f"Error running Darknet: {str(e)}")

# Fungsi utama untuk Streamlit
def main():
    st.title("Deteksi Objek Menggunakan Darknet YOLOv3")
    st.write("Unggah gambar untuk mendeteksi objek menggunakan Darknet")

    # Link unduhan langsung
    cfg_drive_url = 'https://drive.google.com/uc?export=download&id=108kFZ9ltANJW7He-Kujzn7f1FYerf2qA'
    weights_drive_url = 'https://drive.google.com/uc?export=download&id=1-Z_hwylsqXf86t9a8CjvfRgHDs_B_7eh'
    data_drive_url = 'https://drive.google.com/uc?export=download&id=1iEDj2biLlviApMcAhPIUbcKXFqEHp1_o'

    # Lokasi file yang akan diunduh
    cfg_path = 'yolov3_custom.cfg'
    weights_path = 'yolov3.weights'
    data_file = 'obj.data'

    # Unduh file dari Google Drive
    if not os.path.exists(cfg_path):
        download_from_google_drive(cfg_drive_url, cfg_path)
    if not os.path.exists(weights_path):
        download_from_google_drive(weights_drive_url, weights_path)
    if not os.path.exists(data_file):
        download_from_google_drive(data_drive_url, data_file)

    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        image_path = tfile.name

        # Jalankan deteksi objek
        results = detect_objects_with_darknet(image_path, cfg_path, weights_path, data_file)
        if results:
            st.text(results)

        # Tampilkan gambar dengan bounding box
        image = cv2.imread(image_path)
        st.image(image, channels="BGR", caption='Gambar dengan Deteksi', use_column_width=True)

if __name__ == "__main__":
    main()
