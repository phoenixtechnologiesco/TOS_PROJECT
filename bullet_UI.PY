import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import json
import os
import time

# LOAD YOLO MODEL
model = YOLO("best.pt")  

# GLOBAL VARIABLES
cap = None
running = False
bullet_count = 0
results_data = []
fire_distances = []
current_user_file = ""

# Camera calibration / bullet size constants (meters and pixels)
ACTUAL_BULLET_WIDTH = 0.009  # 9 mm bullet
FOCAL_LENGTH = 800           # pixels, set after calibration

# CREATE RESULTS FOLDER
if not os.path.exists("results"):
    os.makedirs("results")

# START DETECTION
def start_detection():
    global cap, running, bullet_count, results_data, fire_distances, current_user_file

    bullet_count = 0
    results_data = []
    fire_distances = []

    user_name = name_var.get().replace(" ", "_")
    user_id = id_var.get()

    if not user_name or not user_id:
        messagebox.showerror("Error", "Please enter Name and ID")
        return

    current_user_file = f"results/{user_name}_{user_id}.json"

    # RTSP CAMERA URL
    stream_url = "rtsp://admin:phoenix0332@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0" 

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open RTSP stream!")
        return

    running = True
    user_frame.pack_forget()
    camera_frame.pack()
    update_frame()

# UPDATE FRAME
def update_frame():
    global cap, running, bullet_count

    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        return

    # YOLO Detection
    results = model.predict(frame, conf=0.3, iou=0.45, verbose=False)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)

        # Draw rectangle and label
        label = f"Bullet {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # BULLET COUNT
        bullet_count += 1
        bullet_label.config(text=f"Bullets Detected: {bullet_count}")

        # ACTUAL DISTANCE CALCULATION
        box_width = x2 - x1
        if box_width > 0:
            distance = (ACTUAL_BULLET_WIDTH * FOCAL_LENGTH) / box_width
        else:
            distance = 0

        fire_distances.append(distance)

        # Save detection info
        results_data.append({"time": time.time(), "confidence": conf, "distance": distance})

    # Convert frame to Tkinter Image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)

    camera_label.after(10, update_frame)

# GET LONGEST FIRE
def get_longest_fire():
    if not fire_distances:
        return None, None
    max_distance = max(fire_distances)
    fire_number = fire_distances.index(max_distance) + 1
    return fire_number, max_distance

# FINISH USER
def finish_user():
    global running
    running = False
    if cap:
        cap.release()

    camera_frame.pack_forget()
    result_frame.pack()

    fire_no, max_dist = get_longest_fire()

    # Write JSON file
    data = {
        "name": name_var.get(),
        "id": id_var.get(),
        "total_bullets": bullet_count,
        "detections": results_data,
        "longest_fire_number": fire_no,
        "longest_fire_distance": max_dist
    }

    with open(current_user_file, "w") as f:
        json.dump(data, f, indent=4)

    res_label.config(
        text=f"Saved results: {current_user_file}\nLongest distance fire: #{fire_no} ({max_dist:.2f} m)"
    )

# NEXT USER
def next_user():
    result_frame.pack_forget()
    name_var.set("")
    id_var.set("")
    user_frame.pack()

# TKINTER UI
root = tk.Tk()
root.title("YOLO Bullet Detection UI")
root.geometry("600x700")

# USER INPUT FRAME
user_frame = tk.Frame(root)
user_frame.pack(pady=50)

name_var = tk.StringVar()
id_var = tk.StringVar()

tk.Label(user_frame, text="Enter Name:").pack()
tk.Entry(user_frame, textvariable=name_var).pack()

tk.Label(user_frame, text="Enter ID:").pack()
tk.Entry(user_frame, textvariable=id_var).pack()

tk.Button(user_frame, text="Start Detection", command=start_detection).pack(pady=20)

# CAMERA FRAME
camera_frame = tk.Frame(root)

camera_label = tk.Label(camera_frame)
camera_label.pack()

bullet_label = tk.Label(camera_frame, text="Bullets Detected: 0", font=("Arial", 14))
bullet_label.pack(pady=10)

tk.Button(camera_frame, text="FINISH", bg="red", fg="white", font=("Arial", 14), command=finish_user).pack(pady=20)

# RESULT FRAME
result_frame = tk.Frame(root)

res_label = tk.Label(result_frame, text="", font=("Arial", 12))
res_label.pack(pady=20)

tk.Button(result_frame, text="NEXT USER", command=next_user).pack(pady=20)

root.mainloop()
