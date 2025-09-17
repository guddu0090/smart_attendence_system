import streamlit as st
import dlib
import cv2
import os
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from PIL import Image
import shutil
import logging

# --- Setup and Configuration ---
st.set_page_config(page_title="Attendance System", layout="wide")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "attendance.db")
SHAPE_PREDICTOR_PATH = os.path.join(DATA_DIR, "data_dlib", "shape_predictor_68_face_landmarks.dat")
FACE_REC_MODEL_PATH = os.path.join(DATA_DIR, "data_dlib", "dlib_face_recognition_resnet_model_v1.dat")
FACES_DATA_DIR = os.path.join(DATA_DIR, "data_faces_from_camera")
FEATURES_CSV_PATH = os.path.join(DATA_DIR, "features_all.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "data_dlib"), exist_ok=True)
os.makedirs(FACES_DATA_DIR, exist_ok=True)

def check_required_files():
    if not os.path.exists(SHAPE_PREDICTOR_PATH) or not os.path.exists(FACE_REC_MODEL_PATH):
        st.error(
            "**Required Model Files Not Found!**\n\n"
            "Please make sure these are downloaded and placed in:\n\n"
            f"`{os.path.join(DATA_DIR, 'data_dlib')}` folder.\n\n"
            "Download links:\n"
            "[Shape Predictor](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)"
        )
        st.stop()
    return True

def setup_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            class TEXT NOT NULL,
            UNIQUE(name, class)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            time TEXT NOT NULL,
            date DATE NOT NULL,
            UNIQUE(name, date)
        )
    """)
    conn.commit()
    conn.close()

@st.cache_resource
def load_models():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    face_rec_model = dlib.face_recognition_model_v1(FACE_REC_MODEL_PATH)
    return detector, predictor, face_rec_model

def return_128d_features(img_bgr, face_rect, predictor, face_rec_model):
    try:
        shape = predictor(img_bgr, face_rect)
        return np.array(face_rec_model.compute_face_descriptor(img_bgr, shape))
    except Exception as e:
        logging.error(f"Error computing 128D features: {e}")
        return None

def get_all_known_faces():
    if os.path.exists(FEATURES_CSV_PATH):
        try:
            df = pd.read_csv(FEATURES_CSV_PATH, header=None)
            person_names = df.iloc[:, 0].values.tolist()
            features_known_arr = df.iloc[:, 1:].values.astype(np.float64)
            return person_names, features_known_arr
        except Exception as e:
            st.error(f"Error loading '{FEATURES_CSV_PATH}': {e}")
            return [], np.array([])
    return [], np.array([])

def compute_and_save_features_from_images():
    person_folders = [f for f in os.listdir(FACES_DATA_DIR) if os.path.isdir(os.path.join(FACES_DATA_DIR, f))]
    if not person_folders:
        open(FEATURES_CSV_PATH, 'w').close()
        return

    detector, predictor, face_rec_model = load_models()
    all_features_list = []
    progress_bar = st.progress(0, "Processing student images...")

    for i, person_folder in enumerate(person_folders):
        person_path = os.path.join(FACES_DATA_DIR, person_folder)
        photo_files = [f for f in os.listdir(person_path) if f.endswith(('.jpg', '.png'))]

        for photo in photo_files:
            img_path = os.path.join(person_path, photo)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            faces = detector(img_bgr, 1)
            if len(faces) == 1:
                features = return_128d_features(img_bgr, faces[0], predictor, face_rec_model)
                if features is not None:
                    all_features_list.append([person_folder] + list(features))

        progress_bar.progress((i + 1) / len(person_folders), f"Processing: {person_folder}")

    if all_features_list:
        df = pd.DataFrame(all_features_list)
        df.to_csv(FEATURES_CSV_PATH, header=False, index=False)
        st.success(f"Processed {len(all_features_list)} images and updated features.")
    else:
        st.warning("No valid faces found. Features file is empty.")
        open(FEATURES_CSV_PATH, 'w').close()

    progress_bar.empty()

def login_page():
    st.header("Attendance System Login")
    login_as = st.radio("Login as:", ("Teacher", "Student"), horizontal=True)

    if login_as == "Teacher":
        with st.form("teacher_login_form"):
            username = st.text_input("Username", value="admin")
            password = st.text_input("Password", type="password", value="admin")
            if st.form_submit_button("Login"):
                if username == "admin" and password == "admin":
                    st.session_state.logged_in = True
                    st.session_state.role = "teacher"
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    elif login_as == "Student":
        conn = sqlite3.connect(DB_PATH)
        students = pd.read_sql_query("SELECT name FROM students ORDER BY name", conn)['name'].tolist()
        conn.close()

        if not students:
            st.info("No students registered.")
            return

        with st.form("student_login_form"):
            selected_student = st.selectbox("Select Your Name", options=students)
            if st.form_submit_button("Login"):
                st.session_state.logged_in = True
                st.session_state.role = "student"
                st.session_state.student_name = selected_student
                st.rerun()

def teacher_dashboard():
    st.title("Teacher Dashboard")
    st.sidebar.button("Logout", on_click=logout, type="primary")

    tabs = st.tabs(["Take Attendance", "Register Student", "Manage Students", "View Records", "Midday Meal Report"])
    with tabs[0]: take_attendance_ui()
    with tabs[1]: register_student_ui()
    with tabs[2]: manage_students_ui()
    with tabs[3]: view_attendance_ui()
    with tabs[4]: midday_meal_report_ui()

def register_student_ui():
    st.header("Register a New Student")

    st.info("Please take at least 3 high-quality photos (straight, left, right).")

    if 'captured_images' not in st.session_state:
        st.session_state.captured_images = []

    img_file_buffer = st.camera_input("Take Photo")

    if img_file_buffer:
        pil_image = Image.open(img_file_buffer)
        np_image = np.array(pil_image)
        st.session_state.captured_images.append(np_image)

    if st.session_state.captured_images:
        st.write("Captured Photos:")
        cols = st.columns(5)
        for i, img in enumerate(st.session_state.captured_images):
            cols[i % 5].image(img, width=100)
        
        if st.button("Clear Photos"):
            st.session_state.captured_images = []
            st.rerun()

    with st.form("registration_form", clear_on_submit=True):
        student_name = st.text_input("Student Name", placeholder="e.g., John Doe")
        student_class = st.text_input("Student Class", placeholder="e.g., Class 10A")
        submitted = st.form_submit_button("Register Student")
        
        if submitted:
            if not student_name.strip() or not student_class.strip():
                st.error("Name and class are required.")
            elif len(st.session_state.captured_images) < 3:
                st.error("At least 3 photos are required.")
            else:
                register_student(student_name.strip(), student_class.strip())
                st.session_state.captured_images = []
                st.rerun()

def register_student(student_name, student_class):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO students (name, class) VALUES (?, ?)", (student_name, student_class))
        conn.commit()
        conn.close()

        person_dir = os.path.join(FACES_DATA_DIR, student_name)
        os.makedirs(person_dir, exist_ok=True)
        
        detector, _, _ = load_models()
        valid_photos_saved = 0
        for i, img_array in enumerate(st.session_state.captured_images):
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            faces = detector(img_bgr, 1)
            if len(faces) == 1:
                cv2.imwrite(os.path.join(person_dir, f"{valid_photos_saved}.jpg"), img_bgr)
                valid_photos_saved += 1

        if valid_photos_saved < 1:
            st.error("Registration failed: No valid photos provided.")
            shutil.rmtree(person_dir)
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM students WHERE name = ? AND class = ?", (student_name, student_class))
            conn.commit()
            conn.close()
            return

        st.success(f"Student '{student_name}' registered with {valid_photos_saved} valid photos.")
        compute_and_save_features_from_images()

    except sqlite3.IntegrityError:
        st.error(f"Student '{student_name}' already exists.")
    except Exception as e:
        st.error(f"Error: {e}")

def take_attendance_ui():
    st.header("Take Attendance")
    person_names, features_known_arr = get_all_known_faces()
    if not person_names:
        st.warning("No students registered.")
        return

    img_file_buffer = st.camera_input("Scan Faces")
    if img_file_buffer:
        img_np = np.array(Image.open(img_file_buffer))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        detector, predictor, face_rec_model = load_models()
        faces = detector(img_bgr, 1)
        
        recognized_in_frame = []
        for face in faces:
            features = return_128d_features(img_bgr, face, predictor, face_rec_model)
            if features is not None:
                distances = np.linalg.norm(features_known_arr - features, axis=1)
                min_idx = np.argmin(distances)
                name = "Unknown"
                color = (0, 0, 255)
                if distances[min_idx] < 0.4:
                    name = person_names[min_idx]
                    color = (0, 255, 0)
                    recognized_in_frame.append(name)
                
                d = face
                cv2.rectangle(img_bgr, (d.left(), d.top()), (d.right(), d.bottom()), color, 2)
                cv2.putText(img_bgr, name, (d.left(), d.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if recognized_in_frame:
            marked_now, already_marked = mark_attendance(recognized_in_frame)
            if marked_now: st.success(f"Attendance marked for: {', '.join(marked_now)}")
            if already_marked: st.info(f"Already present: {', '.join(already_marked)}")
        elif faces:
            st.warning("No registered students recognized.")
        else:
            st.info("No faces detected.")

        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def mark_attendance(names):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    marked_now, already_marked = [], []
    for name in set(names):
        try:
            cursor.execute("INSERT INTO attendance (name, time, date) VALUES (?, ?, ?)",
                           (name, datetime.now().strftime("%H:%M:%S"), today))
            marked_now.append(name)
        except sqlite3.IntegrityError:
            already_marked.append(name)
    conn.commit()
    conn.close()
    return marked_now, already_marked

def manage_students_ui():
    st.header("Manage Registered Students")
    conn = sqlite3.connect(DB_PATH)
    students_df = pd.read_sql_query("SELECT name, class FROM students ORDER BY name", conn)
    conn.close()

    if students_df.empty:
        st.info("No students to manage.")
        return

    for index, row in students_df.iterrows():
        col1, col2, col3 = st.columns([2, 2, 1])
        col1.write(f"**{row['name']}**")
        col2.write(row['class'])
        if col3.button("Delete", key=f"del_{row['name']}"):
            delete_student(row['name'])
            st.rerun()

def delete_student(student_name):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM students WHERE name = ?", (student_name,))
        cursor.execute("DELETE FROM attendance WHERE name = ?", (student_name,))
        conn.commit()
        conn.close()
        
        person_dir = os.path.join(FACES_DATA_DIR, student_name)
        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)
        
        st.success(f"Deleted {student_name}. Re-processing data...")
        compute_and_save_features_from_images()
    except Exception as e:
        st.error(f"Error deleting student: {e}")

def view_attendance_ui():
    st.header("View Attendance Records")
    conn = sqlite3.connect(DB_PATH)
    classes = ["All Classes"] + pd.read_sql_query("SELECT DISTINCT class FROM students ORDER BY class", conn)['class'].tolist()
    
    col1, col2 = st.columns(2)
    selected_date = col1.date_input("Select Date", datetime.now())
    selected_class = col2.selectbox("Filter by Class", options=classes)

    query = """
        SELECT T1.name, T2.class, T1.time
        FROM attendance AS T1 JOIN students AS T2 ON T1.name = T2.name
        WHERE T1.date = ?
    """
    params = [selected_date.strftime("%Y-%m-%d")]
    if selected_class != "All Classes":
        query += " AND T2.class = ?"
        params.append(selected_class)
    
    df = pd.read_sql_query(query + " ORDER BY T2.class, T1.name", conn, params=params)
    conn.close()

    if df.empty:
        st.info(f"No records found for {selected_date.strftime('%B %d, %Y')}.")
    else:
        st.metric(f"Total Present ({selected_class})", len(df))
        st.dataframe(df, use_container_width=True)

def midday_meal_report_ui():
    st.header("Midday Meal Report")
    selected_date = st.date_input("Select Date for Report", datetime.now(), key="mdm_date")

    if st.button("Generate Report"):
        conn = sqlite3.connect(DB_PATH)
        query = """
            SELECT
                T2.class AS "Class",
                COUNT(T1.name) AS "Number of Present Students"
            FROM
                attendance AS T1
            JOIN
                students AS T2 ON T1.name = T2.name
            WHERE
                T1.date = ?
            GROUP BY
                T2.class
            ORDER BY
                T2.class
        """
        params = [selected_date.strftime("%Y-%m-%d")]
        
        try:
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()

            if df.empty:
                st.info(f"No attendance records found for {selected_date.strftime('%B %d, %Y')}.")
            else:
                total_present = df["Number of Present Students"].sum()
                st.metric("Total Students for Midday Meal", total_present)
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Report as CSV",
                    data=csv,
                    file_name=f"midday_meal_report_{selected_date.strftime('%Y-%m-%d')}.csv",
                    mime='text/csv',
                )
        except Exception as e:
            st.error(f"Error generating report: {e}")
            conn.close()

def student_dashboard():
    student_name = st.session_state.student_name
    st.title(f"Welcome, {student_name}!")
    st.sidebar.button("Logout", on_click=logout, type="primary")
    st.header("Your Attendance Record")
    
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT date, time FROM attendance WHERE name = ? ORDER BY date DESC", conn, params=(student_name,))
    conn.close()

    if df.empty:
        st.warning("No attendance records found.")
    else:
        st.metric("Total Days Present", len(df))
        st.dataframe(df, use_container_width=True)

def logout():
    for key in list(st.session_state.keys()):
        if key not in ['theme']:
            del st.session_state[key]
    st.rerun()

def main():
    check_required_files()
    setup_database()

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        login_page()
    else:
        if st.session_state.get("role") == "teacher":
            teacher_dashboard()
        elif st.session_state.get("role") == "student":
            student_dashboard()
        else:
            login_page()
