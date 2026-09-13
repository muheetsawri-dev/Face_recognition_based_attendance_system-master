"""
FaceTrack Pro — Streamlit Edition

Converted from the original CustomTkinter desktop app to a Streamlit web app
so it can be deployed on Streamlit Community Cloud and opened from a link on
any phone or PC browser.

KEY DIFFERENCE FROM THE DESKTOP VERSION:
The desktop app used a live OpenCV video loop (cv2.VideoCapture + cv2.imshow)
where you could hold a button and it kept capturing frames continuously.
Streamlit's camera widget (st.camera_input) only gives ONE photo per click —
there is no live video feed in a browser-based Streamlit app. So:
    - "Take Images" becomes: click camera -> take snapshot -> auto-saves ->
      camera resets -> take another snapshot -> repeat until you click "Finish".
    - "Take Attendance" becomes: click camera -> take one snapshot -> it runs
      face recognition on that single photo and marks attendance if matched.

Everything else (Excel storage via openpyxl, LBPH face recognizer training,
folder layout) is kept the same as your original code.

FOLDER LAYOUT (same idea as your desktop app):
    your-project/
      app.py                                <- this file
      requirements.txt
      haarcascade_frontalface_default.xml   <- MUST be uploaded to your repo
      StudentDetails/StudentDetails.xlsx
      TrainingImage/<name>.<serial>.<id>.<n>.jpg
      TrainingImageLabel/Trainner.yml, psd.txt, settings.json
      Attendance/Attendance_<dd-mm-yyyy>.xlsx

IMPORTANT DEPLOYMENT NOTE:
Streamlit Community Cloud's filesystem is NOT permanent. Every time the app
restarts (goes to sleep after inactivity, or you push a new commit), any
files written while it was running (registered students, trained model,
attendance records) are WIPED and reset back to whatever is in your GitHub
repo. This is fine for a demo/portfolio project, but it is NOT suitable for
real production attendance tracking unless you connect it to an external
database (that's a separate, bigger step).

Requires (see requirements.txt):
    streamlit, opencv-contrib-python-headless, numpy, pillow, openpyxl, pandas
"""

import os
import io
import json
import time
import hashlib
import datetime
from collections import Counter, defaultdict

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from openpyxl import Workbook, load_workbook

# ----------------------------------------------------------------------
# Paths — all relative to this file, same layout as the desktop app
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENT_DETAILS_DIR = os.path.join(BASE_DIR, "StudentDetails")
STUDENT_DETAILS_XLSX = os.path.join(STUDENT_DETAILS_DIR, "StudentDetails.xlsx")
TRAINING_IMAGE_DIR = os.path.join(BASE_DIR, "TrainingImage")
TRAINING_LABEL_DIR = os.path.join(BASE_DIR, "TrainingImageLabel")
TRAINER_FILE = os.path.join(TRAINING_LABEL_DIR, "Trainner.yml")
PASSWORD_FILE = os.path.join(TRAINING_LABEL_DIR, "psd.txt")
SETTINGS_FILE = os.path.join(TRAINING_LABEL_DIR, "settings.json")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "Attendance")
HAARCASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
HAARCASCADE_RAW_URL = ("https://raw.githubusercontent.com/opencv/opencv/master/data/"
                        "haarcascades/haarcascade_frontalface_default.xml")

MAX_SAMPLES = 60
MIN_SAMPLES_WARN = 20

for _d in (STUDENT_DETAILS_DIR, TRAINING_IMAGE_DIR, TRAINING_LABEL_DIR, ATTENDANCE_DIR):
    os.makedirs(_d, exist_ok=True)

# ----------------------------------------------------------------------
# Settings (same idea as the desktop Settings tab)
# ----------------------------------------------------------------------
DEFAULT_SETTINGS = {
    "class_start_time": "09:00",
    "confidence_threshold": 50,
    "sound_on": True,
}


def load_settings():
    if os.path.isfile(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                data = json.load(f)
            merged = dict(DEFAULT_SETTINGS)
            merged.update(data)
            return merged
        except Exception:
            pass
    return dict(DEFAULT_SETTINGS)


def save_settings(new_settings):
    os.makedirs(TRAINING_LABEL_DIR, exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(new_settings, f, indent=2)


if "settings" not in st.session_state:
    st.session_state.settings = load_settings()

SETTINGS = st.session_state.settings


def class_start_time_obj():
    h, m = SETTINGS["class_start_time"].split(":")
    return datetime.time(int(h), int(m), 0)


# ----------------------------------------------------------------------
# Small helpers (same logic as the desktop app)
# ----------------------------------------------------------------------
def assure_path_exists(path):
    directory = path if os.path.isdir(path) or path.endswith(os.sep) else os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def hash_password(raw):
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cascade_is_valid(path):
    if not os.path.isfile(path):
        return False, f"File not found at:\n{path}"
    if os.path.getsize(path) < 10_000:
        return False, ("The file is far too small to be a real cascade file "
                        "(it may have been saved as an HTML page instead of the raw XML).")
    try:
        clf = cv2.CascadeClassifier(path)
    except Exception as exc:
        return False, f"OpenCV could not parse the file: {exc}"
    if clf.empty():
        return False, "OpenCV loaded the file but it is not a valid cascade."
    return True, ""


def compute_status(dt_obj):
    return "Late" if dt_obj.time() > class_start_time_obj() else "On Time"


def compute_status_from_timestr(tstamp):
    try:
        t = datetime.datetime.strptime(str(tstamp), "%H:%M:%S").time()
        return "Late" if t > class_start_time_obj() else "On Time"
    except Exception:
        return "On Time"


def decode_camera_image(camera_file):
    """Convert a Streamlit camera_input file into an OpenCV BGR image."""
    bytes_data = camera_file.getvalue()
    np_arr = np.frombuffer(bytes_data, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def detect_largest_face(gray_img, cascade):
    faces = cascade.detectMultiScale(gray_img, 1.2, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


# ----------------------------------------------------------------------
# Excel storage helpers (same logic as the desktop app)
# ----------------------------------------------------------------------
def _ensure_student_workbook():
    assure_path_exists(STUDENT_DETAILS_DIR)
    if not os.path.isfile(STUDENT_DETAILS_XLSX):
        wb = Workbook()
        ws = wb.active
        ws.title = "StudentDetails"
        ws.append(['SERIAL NO.', 'ID', 'NAME'])
        wb.save(STUDENT_DETAILS_XLSX)
        wb.close()


def read_student_details():
    _ensure_student_workbook()
    wb = load_workbook(STUDENT_DETAILS_XLSX, read_only=True)
    ws = wb.active
    data = {}
    max_serial = 0
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or row[0] is None:
            continue
        serial, sid, name = row[0], row[1], row[2]
        try:
            serial = int(serial)
        except (TypeError, ValueError):
            continue
        data[serial] = (str(sid), str(name))
        max_serial = max(max_serial, serial)
    wb.close()
    return max_serial + 1, data


def append_student_detail(serial, Id, name):
    _ensure_student_workbook()
    wb = load_workbook(STUDENT_DETAILS_XLSX)
    ws = wb.active
    ws.append([serial, Id, name])
    wb.save(STUDENT_DETAILS_XLSX)
    wb.close()


def remove_student_detail(serial):
    _ensure_student_workbook()
    wb = load_workbook(STUDENT_DETAILS_XLSX)
    ws = wb.active
    rows_to_delete = []
    for i, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        if row and row[0] is not None:
            try:
                if int(row[0]) == serial:
                    rows_to_delete.append(i)
            except (TypeError, ValueError):
                continue
    for i in reversed(rows_to_delete):
        ws.delete_rows(i)
    wb.save(STUDENT_DETAILS_XLSX)
    wb.close()


def delete_training_images(serial):
    if not os.path.isdir(TRAINING_IMAGE_DIR):
        return 0
    removed = 0
    for fname in os.listdir(TRAINING_IMAGE_DIR):
        parts = fname.split(".")
        if len(parts) >= 3:
            try:
                if int(parts[1]) == serial:
                    os.remove(os.path.join(TRAINING_IMAGE_DIR, fname))
                    removed += 1
            except (ValueError, OSError):
                continue
    return removed


def photo_count_for_serial(serial):
    if not os.path.isdir(TRAINING_IMAGE_DIR):
        return 0
    n = 0
    for fname in os.listdir(TRAINING_IMAGE_DIR):
        parts = fname.split(".")
        if len(parts) >= 3:
            try:
                if int(parts[1]) == serial:
                    n += 1
            except ValueError:
                continue
    return n


def append_attendance_rows(date_str, rows):
    assure_path_exists(ATTENDANCE_DIR)
    path = os.path.join(ATTENDANCE_DIR, f"Attendance_{date_str}.xlsx")
    if os.path.isfile(path):
        wb = load_workbook(path)
        ws = wb.active
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = "Attendance"
        ws.append(['Id', 'Name', 'Date', 'In-Time', 'Status'])
    for r in rows:
        ws.append(r)
    wb.save(path)
    wb.close()
    return path


def load_attendance_for_date(date_str):
    path = os.path.join(ATTENDANCE_DIR, f"Attendance_{date_str}.xlsx")
    records = []
    if os.path.isfile(path):
        wb = load_workbook(path, read_only=True)
        ws = wb.active
        for row in ws.iter_rows(min_row=2, values_only=True):
            if not row or row[0] is None:
                continue
            vals = list(row) + [None] * (5 - len(row))
            sid, name_, date_val, tstamp, status = vals[:5]
            if not status:
                status = compute_status_from_timestr(tstamp)
            records.append((str(sid), str(name_), str(date_val), str(tstamp), str(status)))
        wb.close()
    records.reverse()
    return records


def load_today_attendance():
    date_str = datetime.datetime.now().strftime('%d-%m-%Y')
    return load_attendance_for_date(date_str), date_str


def load_attendance_range(start_date, end_date):
    all_records = []
    d = start_date
    while d <= end_date:
        date_str = d.strftime('%d-%m-%Y')
        all_records.extend(load_attendance_for_date(date_str))
        d += datetime.timedelta(days=1)
    return all_records


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    faces, Ids = [], []
    for imagePath in imagePaths:
        try:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            serial = int(os.path.split(imagePath)[-1].split(".")[1])
            faces.append(imageNp)
            Ids.append(serial)
        except (IndexError, ValueError):
            continue
    return faces, Ids


# ========================================================================
# STREAMLIT APP
# ========================================================================
st.set_page_config(page_title="FaceTrack Pro", page_icon="🎯", layout="wide")

PAGES = ["Dashboard", "Take Attendance", "Students", "Register", "Reports", "Settings"]

st.sidebar.markdown("## 🎯 FaceTrack Pro")
st.sidebar.caption("Attendance suite")
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

ok_cascade, cascade_reason = cascade_is_valid(HAARCASCADE_PATH)
if ok_cascade:
    st.sidebar.success("Cascade file OK")
else:
    st.sidebar.error("Cascade file missing/invalid")

st.title(page)

# ------------------------------------------------------------------
# DASHBOARD
# ------------------------------------------------------------------
if page == "Dashboard":
    records, today_str = load_today_attendance()
    _next_serial, students = read_student_details()

    total_reg = len(students)
    present = len(records)
    late = sum(1 for r in records if r[4] == "Late")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Registered", total_reg)
    col2.metric("Present Today", present)
    col3.metric("Late Today", late)

    st.subheader("Today's ledger")
    if records:
        df = pd.DataFrame(records, columns=["ID", "Name", "Date", "Time", "Status"])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No attendance marked yet today.")

# ------------------------------------------------------------------
# TAKE ATTENDANCE
# ------------------------------------------------------------------
elif page == "Take Attendance":
    if not ok_cascade:
        st.error(f"{cascade_reason}\n\nDownload the raw file from:\n{HAARCASCADE_RAW_URL}\n\n"
                  f"and place it in your repo as haarcascade_frontalface_default.xml")
    elif not os.path.isfile(TRAINER_FILE):
        st.warning("No trained model found yet. Go to Register, add someone, then Save Profile first.")
    else:
        st.write("Take a photo below — if a registered face is recognized, attendance is marked automatically.")
        cam_photo = st.camera_input("Look at the camera and take a photo")

        if cam_photo is not None:
            if not hasattr(cv2, "face"):
                st.error("cv2.face not available. Make sure opencv-contrib-python-headless is installed.")
            else:
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                recognizer.read(TRAINER_FILE)
                _next_serial, student_lookup = read_student_details()
                today_records, date_str = load_today_attendance()
                already_marked_ids = {r[0] for r in today_records}

                img = decode_camera_image(cam_photo)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faceCascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
                face_box = detect_largest_face(gray, faceCascade)

                if face_box is None:
                    st.warning("No face detected in that photo — try again with better lighting.")
                else:
                    x, y, w, h = face_box
                    threshold = SETTINGS.get("confidence_threshold", 50)
                    serial, conf = recognizer.predict(gray[y:y + h, x:x + w])

                    if conf < threshold and serial in student_lookup:
                        student_id, student_name = student_lookup[serial]
                        if student_id in already_marked_ids:
                            st.info(f"{student_name} (ID {student_id}) is already marked present today.")
                        else:
                            now_dt = datetime.datetime.now()
                            now_str = now_dt.strftime('%H:%M:%S')
                            status = compute_status(now_dt)
                            append_attendance_rows(date_str, [[student_id, student_name, date_str, now_str, status]])
                            st.success(f"Marked present: {student_name} (ID {student_id}) — {status} at {now_str}")
                    else:
                        st.error("Face not recognized. Make sure this person is registered and the model is trained.")

# ------------------------------------------------------------------
# STUDENTS
# ------------------------------------------------------------------
elif page == "Students":
    _next_serial, students = read_student_details()
    search = st.text_input("Search students by name or ID")

    if not students:
        st.info("No students registered yet — use the Register page.")
    else:
        ft = search.strip().lower()
        for serial, (sid, name) in sorted(students.items(), key=lambda kv: kv[1][1].lower()):
            if ft and ft not in name.lower() and ft not in sid.lower():
                continue
            count = photo_count_for_serial(serial)
            c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
            c1.write(f"**{name}**")
            c2.write(f"ID {sid}")
            warn = " ⚠️" if count < MIN_SAMPLES_WARN else ""
            c3.write(f"{count} photos{warn}")
            if c4.button("Delete", key=f"del_{serial}"):
                remove_student_detail(serial)
                removed = delete_training_images(serial)
                st.success(f"{name} removed ({removed} photo(s) deleted). Retrain the model in Register.")
                st.rerun()

# ------------------------------------------------------------------
# REGISTER
# ------------------------------------------------------------------
elif page == "Register":
    st.subheader("New registration")
    st.caption("Fill in details, take several photos of the same person, then train the model.")

    if "reg_capture_count" not in st.session_state:
        st.session_state.reg_capture_count = 0
    if "reg_serial" not in st.session_state:
        st.session_state.reg_serial = None

    reg_id = st.text_input("ID (numbers only)", key="reg_id")
    reg_name = st.text_input("Name (letters and spaces only)", key="reg_name")

    id_ok = reg_id.strip().isdigit() if reg_id else False
    name_ok = all(c.isalpha() or c.isspace() for c in reg_name) if reg_name else False

    if not ok_cascade:
        st.error(f"{cascade_reason}\n\nDownload from:\n{HAARCASCADE_RAW_URL}")
    elif reg_id and reg_name and id_ok and name_ok:
        st.write(f"**Captured so far: {st.session_state.reg_capture_count} / {MAX_SAMPLES}**")
        cam_key = f"reg_cam_{st.session_state.reg_capture_count}"
        cam_photo = st.camera_input("Take a photo (take several from different angles)", key=cam_key)

        if cam_photo is not None and st.session_state.reg_capture_count < MAX_SAMPLES:
            if st.session_state.reg_serial is None:
                serial, _existing = read_student_details()
                st.session_state.reg_serial = serial

            img = decode_camera_image(cam_photo)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faceCascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
            face_box = detect_largest_face(gray, faceCascade)

            if face_box is None:
                st.warning("No face detected in that photo — it was not saved. Try again.")
            else:
                x, y, w, h = face_box
                st.session_state.reg_capture_count += 1
                fname = f"{reg_name}.{st.session_state.reg_serial}.{reg_id}.{st.session_state.reg_capture_count}.jpg"
                cv2.imwrite(os.path.join(TRAINING_IMAGE_DIR, fname), gray[y:y + h, x:x + w])
                st.success(f"Saved photo {st.session_state.reg_capture_count}")
                st.rerun()

        colA, colB = st.columns(2)
        if colA.button("Finish registration", disabled=st.session_state.reg_capture_count == 0):
            append_student_detail(st.session_state.reg_serial, reg_id, reg_name)
            st.success(f"Registered {reg_name} (ID {reg_id}) with "
                       f"{st.session_state.reg_capture_count} photo(s). Now train the model below.")
            st.session_state.reg_capture_count = 0
            st.session_state.reg_serial = None

        if colB.button("Cancel / start over"):
            st.session_state.reg_capture_count = 0
            st.session_state.reg_serial = None
            st.rerun()
    else:
        st.info("Enter a valid numeric ID and a name (letters only) to start capturing photos.")

    st.divider()
    st.subheader("Train the model (Save Profile)")

    if not os.path.isfile(PASSWORD_FILE):
        st.write("No password is set yet. Set one now — you'll use it every time you train the model.")
        new_pw = st.text_input("Set a password", type="password", key="set_pw")
        if st.button("Set password"):
            if new_pw.strip():
                assure_path_exists(TRAINING_LABEL_DIR)
                with open(PASSWORD_FILE, "w") as f:
                    f.write(hash_password(new_pw.strip()))
                st.success("Password set. Now enter it below to train.")
                st.rerun()
            else:
                st.error("Password cannot be empty.")
    else:
        pw = st.text_input("Enter password to train", type="password", key="train_pw")
        if st.button("Save Profile (train model)"):
            with open(PASSWORD_FILE, "r") as f:
                key = f.read().strip()
            if hash_password(pw.strip()) != key:
                st.error("Wrong password.")
            elif not hasattr(cv2, "face"):
                st.error("cv2.face not available. Make sure opencv-contrib-python-headless is installed.")
            else:
                faces, ids = getImagesAndLabels(TRAINING_IMAGE_DIR)
                if not faces:
                    st.error("No training photos found. Register someone first.")
                else:
                    counts = Counter(ids)
                    _next_serial, students = read_student_details()
                    low = [(serial, students.get(serial, ('?', 'Unknown'))[1], c)
                           for serial, c in counts.items() if c < MIN_SAMPLES_WARN]
                    if low:
                        for _serial, name, c in low:
                            st.warning(f"{name} has only {c} photo(s) — recognition may be unreliable.")
                    with st.spinner("Training model..."):
                        recognizer = cv2.face.LBPHFaceRecognizer_create()
                        recognizer.train(faces, np.array(ids))
                        assure_path_exists(TRAINING_LABEL_DIR)
                        recognizer.save(TRAINER_FILE)
                    st.success("Profile saved successfully. The model is ready for Take Attendance.")

# ------------------------------------------------------------------
# REPORTS
# ------------------------------------------------------------------
elif page == "Reports":
    today = datetime.date.today()
    week_ago = today - datetime.timedelta(days=7)

    c1, c2 = st.columns(2)
    start = c1.date_input("From", value=week_ago, format="DD-MM-YYYY")
    end = c2.date_input("To", value=today, format="DD-MM-YYYY")

    if st.button("Run report"):
        if start > end:
            st.error("The From date must be before the To date.")
        elif (end - start).days > 92:
            st.error("Please pick a range of 92 days or fewer.")
        else:
            records = load_attendance_range(start, end)
            total_days = (end - start).days + 1
            per_student = defaultdict(lambda: {"name": "", "present": 0, "late": 0})
            for sid, name, _date, _time, status in records:
                per_student[sid]["name"] = name
                per_student[sid]["present"] += 1
                if status == "Late":
                    per_student[sid]["late"] += 1

            rows = []
            for sid, d in sorted(per_student.items(), key=lambda kv: -kv[1]["present"]):
                pct = round(100 * d["present"] / total_days) if total_days else 0
                rows.append((sid, d["name"], d["present"], d["late"], pct))

            if not rows:
                st.info("No attendance recorded in this range.")
            else:
                df = pd.DataFrame(rows, columns=["ID", "Name", "Days Present", "Late", "Attendance %"])
                st.dataframe(df, use_container_width=True, hide_index=True)
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button("Export CSV", data=csv_bytes,
                                    file_name=f"Report_{start}_{end}.csv", mime="text/csv")
                st.caption(f"{len(rows)} student(s) over {total_days} day(s) "
                           f"({start.strftime('%d-%m-%Y')} to {end.strftime('%d-%m-%Y')})")

# ------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------
elif page == "Settings":
    time_txt = st.text_input("Class start time (HH:MM, 24h)", value=SETTINGS["class_start_time"])
    conf = st.slider("Recognition confidence threshold (lower = stricter)", 20, 90,
                      value=SETTINGS["confidence_threshold"])
    sound_on = st.checkbox("Play a sound on each successful recognition", value=SETTINGS["sound_on"])

    if st.button("Save settings"):
        try:
            h, m = time_txt.strip().split(":")
            datetime.time(int(h), int(m))
        except Exception:
            st.error("Use HH:MM 24-hour format, e.g. 09:00.")
        else:
            new_settings = {
                "class_start_time": time_txt.strip(),
                "confidence_threshold": int(conf),
                "sound_on": bool(sound_on),
            }
            save_settings(new_settings)
            st.session_state.settings = new_settings
            st.success("Settings saved. They'll apply to the next recognition.")
            st.rerun()

    st.divider()
    st.caption(
        "Note: on Streamlit Community Cloud, files written while the app runs "
        "(registrations, trained model, attendance records) reset whenever the "
        "app restarts or you push a new commit. This demo does not use persistent "
        "cloud storage."
    )
