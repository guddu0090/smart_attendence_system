import streamlit as st
import os
import requests
from main_attendance_system import main

# Public model file URLs (replace these with your real links)
SHAPE_PREDICTOR_URL = "https://drive.google.com/file/d/1j1G_iUcvSRt8VUTVMer3fyGDuUoVIBSP/view?usp=drive_link"
FACE_REC_MODEL_URL = "https://drive.google.com/file/d/1j1G_iUcvSRt8VUTVMer3fyGDuUoVIBSP/view?usp=drive_link"

DATA_DIR = "data"
SHAPE_PREDICTOR_PATH = os.path.join(DATA_DIR, "data_dlib", "shape_predictor_68_face_landmarks.dat")
FACE_REC_MODEL_PATH = os.path.join(DATA_DIR, "data_dlib", "dlib_face_recognition_resnet_model_v1.dat")

def download_file(url, dest_path):
    if not os.path.exists(dest_path):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        st.info(f"Downloading model from {url}... Please wait.")
        r = requests.get(url, stream=True)
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.success(f"Model saved to {dest_path}")

download_file(SHAPE_PREDICTOR_URL, SHAPE_PREDICTOR_PATH)
download_file(FACE_REC_MODEL_URL, FACE_REC_MODEL_PATH)

def run():
    st.set_page_config(page_title="Attendance System", layout="wide")
    main()

if __name__ == "__main__":
    run()
