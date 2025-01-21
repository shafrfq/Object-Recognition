import streamlit as st
import cv2
import numpy as np
import os
import gdown
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fungsi untuk mengunduh file dari Google Drive
def download_file_from_gdrive(file_id, output_path):
    if not os.path.exists(output_path):
        logger.info(f"Downloading file from Google Drive to {output_path}...")
        try:
            gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
        except Exception as e:
            st.error(f"Error downloading file from Google Drive: {e}")

# Mengunduh model YOLOv3
@st.cache_resource
def load_yolo():
    os.makedirs('yolov3', exist_ok=True)
    # Replace these IDs with your actual file IDs from Google Drive
    download_file_from_gdrive('1ATv5Z0f8Ph-vpfOOGw92uCutbMbVWWd0', 'yolov3/yolov3_5k_parameter.weights')
    download_file_from_gdrive('1eD6z6IHG0WwDPLJWyB4dJTFJ6-6DKA8E', 'yolov3/yolov3_custom.cfg')
    download_file_from_gdrive('15MziLJaMBGVMayE2t3EqubSJvrPaehN3', 'yolov3/custom.names')

    net = cv2.dnn.readNet('yolov3/yolov3_custom_last.weights', 'yolov3/yolov3_custom.cfg')
    with open('yolov3/custom.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return net, classes, output_layers

# Definisikan subset label yang diizinkan
allowed_labels = {"Bus", "Car", "Motorcycle", "Person", "Truck"}

# Membuat warna acak untuk setiap kelas
colors = {}
def generate_colors(classes):
    for class_name in classes:
        colors[class_name] = [random.randint(0, 255) for _ in range(3)]
    return colors

# Fungsi untuk deteksi objek
def detect_objects(net, classes, output_layers, image, allowed_labels):
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in allowed_labels:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            class_name = classes[class_ids[i]]
            label = f"{class_name} {confidences[i]*100:.2f}%"
            color = colors[class_name]  # Warna spesifik untuk setiap kelas
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

# Fungsi utama untuk aplikasi Streamlit
def main():
    st.title("Pengenalan Objek dengan YOLO")
    st.write("Pembuatan web sederhana ini sebagai sarana untuk mengevaluasi kemampuan algoritma YOLOv3 dalam mengenali objek di jalan raya")

    net, classes, output_layers = load_yolo()
    generate_colors(classes)  # Generate warna untuk setiap kelas

    uploaded_file = st.file_uploader("Upload Gambarnya Disini...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.image(image, channels="BGR", caption='Uploaded Image.', use_column_width=True)

        if st.button("Detect Objects"):
            st.write("Detecting...")
            detected_image = detect_objects(net, classes, output_layers, image, allowed_labels)
            st.image(detected_image, channels="BGR", caption='Detected Image.', use_column_width=True)

            # Opsi unduh gambar hasil deteksi
            is_success, buffer = cv2.imencode(".jpg", detected_image)
            if is_success:
                st.download_button(
                    label="Download Detected Image",
                    data=buffer.tobytes(),
                    file_name="detected_image.jpg",
                    mime="image/jpeg"
                )

            # Opsi kembali ke tampilan awal
            if st.button("Back to Start"):
                st.experimental_rerun()

if __name__ == "__main__":
    main()

st.markdown("<p style='text-align: center;'>Copyright (C) Shafira Fimelita Q - 2024</p>", unsafe_allow_html=True)
