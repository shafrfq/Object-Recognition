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
    download_file_from_gdrive('10AY0QTG_XbvcRqONfbAVa-ahJrrA40lw', 'yolov3/yolov3_custom_last.weights')
    download_file_from_gdrive('1JP4lJn4OwdK04nxZiaC_Ykz0WQLvbn2U', 'yolov3/yolov3_custom.cfg')
    download_file_from_gdrive('1edWJefoldOZlPKsPwe5ofEY7SewwRvwY', 'yolov3/custom.names')

    net = cv2.dnn.readNet('yolov3/yolov3_custom_last.weights', 'yolov3/yolov3_custom.cfg')
    with open('yolov3/custom.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return net, classes, output_layers

# Definisikan subset label yang diizinkan
allowed_labels = {"Bus", "Car", "Motorcycle", "Person", "Truck"}

# Fungsi untuk deteksi objek
def detect_objects(net, classes, output_layers, image, allowed_labels):
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]  # Warna berbeda untuk setiap kelas

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
            label = f"{classes[class_ids[i]]} {confidences[i]*100:.2f}%"
            color = colors[class_ids[i] % len(colors)]  # Pilih warna berdasarkan class_id
            text_color = (255, 255, 255)  # Putih untuk teks
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    return image

# Fungsi utama untuk aplikasi Streamlit
def main():
    st.title("Object Detection using YOLOv3")
    st.write("Upload an image for object detection")

    net, classes, output_layers = load_yolo()

    # Menambahkan CSS untuk membuat kotak drag and drop berwarna
    st.markdown("""
        <style>
        .drag-and-drop {
            border: 2px dashed #4CAF50;
            background-color: #f0f8ff;
            padding: 20px;
            text-align: center;
            font-weight: bold;
            color: #4CAF50;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Tampilan kotak drag and drop
    st.markdown("<div class='drag-and-drop'>Drag and Drop your files here</div>", unsafe_allow_html=True)

    # Input file uploader
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
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

    # Caption di bawah aplikasi
    st.caption("<div style='text-align: center;'>Copyright (C) Shafira Fimelita Q - 2024</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
