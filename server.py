import json
import time
import cv2
import asyncio
import os
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from typing import List, Dict, Any, Tuple
import numpy as np
from pathlib import Path
from itertools import combinations
import math 
import signal

# --- FastAPI Initialization ---
app = FastAPI()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration & Model Loading ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt" 

model: YOLO = None
try:
    model = YOLO(str(MODEL_PATH))
    print(f"✅ YOLO Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"🛑 CRITICAL ERROR: Could not load YOLO model from {MODEL_PATH}. Check file name and path.")
    print(f"Error details: {e}")

CAMERA_STREAM = "rtsp://admin:phoenix0332@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"

# --- GLOBAL DETECTION/SESSION VARIABLES ---
PIXELS_PER_INCH = 186
PROXIMITY_THRESHOLD_PX = 25 
CONFIDENCE_THRESHOLD = 0.2
IOU_THRESHOLD = 0.5
BOX_SCALE_FACTOR = 1.5
SAVE_DIR = "shooting_sessions"
os.makedirs(SAVE_DIR, exist_ok=True)

COLOR_PALETTE = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
]

all_sessions: List[Dict[str, Any]] = [] 
current_session_index = -1
last_processed_frame: np.ndarray = None # Last frame for saving summary image

# Global Variables (FastAPI)
clients: List[WebSocket] = []
session_running = False
RESULTS_FILE = "session_results.json"

# --- Camera Initialization ---
cap = cv2.VideoCapture(CAMERA_STREAM) 
if not cap.isOpened():
    print(f"⚠️ Warning: RTSP Camera stream not opened successfully. Server will run, but live video/detection won't work.")

class SessionDetails(BaseModel):
    serviceNo: str
    serviceYear: str
    studentName: str

# --- Detection Utility Functions ---

def get_all_existing_centers():
    """Returns all confirmed hole centers from all sessions."""
    centers = []
    for s in all_sessions:
        for entry in s["holes"]:
            centers.append((entry[0], entry[1]))  # (cx, cy)
    return centers

def calculate_group_size(holes: List[Tuple[int, int, int, int, int, int]]) -> float:
    """Calculates the longest distance between any two holes in inches."""
    global PIXELS_PER_INCH
    if len(holes) < 2:
        return 0.0

    centers = [(cx, cy) for (cx, cy, *_) in holes]
    max_dist_px = 0
    
    for (x1, y1), (x2, y2) in combinations(centers, 2):
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist > max_dist_px:
            max_dist_px = dist
            
    return max_dist_px / PIXELS_PER_INCH

# --- FastAPI Utility Functions ---

def save_result_to_file(new_result):
    """Saves session results to a JSON file."""
    data = []
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print("Warning: Existing results file is corrupt. Starting new file.")
            data = []
        
    data.append(new_result)
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"✅ Session results saved to {RESULTS_FILE}")


async def broadcast(message: dict):
    """Broadcasts a JSON message to all connected WebSocket clients."""
    dead_clients = []
    for ws in clients:
        try:
            await ws.send_json(message)
        except:
            dead_clients.append(ws)

    for dc in dead_clients:
        clients.remove(dc)

# --- Endpoints ---

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)

    try:
        while True:
            # Send current session status on connect for client UI
            current_shooter_name = all_sessions[current_session_index]['details']['studentName'] if current_session_index != -1 else "N/A"
            status_message = {
                "event": "status_update",
                "is_active": session_running,
                "current_shooter": current_shooter_name,
                "total_sessions": len(all_sessions)
            }
            # Initial status send
            await websocket.send_json(status_message)
            
            # Keep connection alive: Wait for client message (or just sleep)
            # Using receive_text() is better to detect client disconnect cleanly
            await websocket.receive_text() 
    except WebSocketDisconnect:
        clients.remove(websocket)


@app.post("/start_session") 
async def start_session(details: SessionDetails):
    global session_running, all_sessions, current_session_index
    
    if session_running:
        return {"status": "error", "message": "Another session is already running. Please stop it first."}
        
    current_session_details = details.dict() 
    session_running = True
    
    # 1. Add new session (Multi-Shooter Logic)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    color_index = len(all_sessions) % len(COLOR_PALETTE)
    color = COLOR_PALETTE[color_index]
    
    all_sessions.append({
        "timestamp": timestamp,
        "holes": [],  # Each: (cx, cy, x1, y1, x2, y2)
        "color": color,
        "details": current_session_details 
    })
    current_session_index = len(all_sessions) - 1
    
    print(f"==================================================")
    print(f"▶ New session started! Shooter #{current_session_index + 1} | Student: {details.studentName}")
    print(f"==================================================")
    
    # 2. Inform Front-end
    await broadcast({
        "event": "session_started", 
        "student_name": details.studentName,
        "session_id": current_session_index + 1,
        "total_hits": 0
    })
    
    return {"status": "session started", "details": details}


async def process_frame_for_detection(frame: np.ndarray):
    """Runs detection and updates session hole list and frame drawing."""
    global all_sessions, current_session_index, last_processed_frame
    
    # --- DRAW OLD SESSIONS (ALWAYS DRAW) ---
    processed_frame = frame.copy()
    
    for sess in all_sessions:
        color = sess["color"]
        for (_, _, x1, y1, x2, y2) in sess["holes"]:
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 3)

    if current_session_index == -1 or not session_running:
        # No active session, just draw old boxes and return
        return processed_frame

    # --- RUN DETECTION FOR ACTIVE SESSION ---
    
    session = all_sessions[current_session_index]
    
    # Run YOLO prediction (model is global)
    try:
        results = model.predict(
            processed_frame, 
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=0,
            verbose=False
        )
    except Exception as e:
        # Handle case where model fails (e.g., if model wasn't loaded correctly)
        print(f"⚠️ Error during model prediction: {e}")
        return processed_frame

    current_detections = [] 
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Scaling logic for drawing/storage
        w, h = x2 - x1, y2 - y1
        new_w, new_h = int(w * BOX_SCALE_FACTOR), int(h * BOX_SCALE_FACTOR)
        scaled_x1 = max(0, cx - new_w // 2)
        scaled_y1 = max(0, cy - new_h // 2)
        scaled_x2 = min(frame.shape[1], cx + new_w // 2)
        scaled_y2 = min(frame.shape[0], cy + new_h // 2)

        current_detections.append((cx, cy, scaled_x1, scaled_y1, scaled_x2, scaled_y2))

    # === ADD NEW HOLES TO CURRENT SESSION ===
    all_existing = get_all_existing_centers()
    newly_added_count = 0
    
    for (cx, cy, sx1, sy1, sx2, sy2) in current_detections:
        is_duplicate = False
        
        # Check against ALL existing holes (Multi-Shooter Logic)
        for (ox, oy) in all_existing:
            if math.hypot(cx - ox, cy - oy) < PROXIMITY_THRESHOLD_PX:
                is_duplicate = True
                break
        
        if not is_duplicate:
            # Add the new hole and update the existing centers list 
            all_sessions[current_session_index]["holes"].append((cx, cy, sx1, sy1, sx2, sy2))
            # Append to all_existing so subsequent detections in the same frame don't double count
            all_existing.append((cx, cy)) 
            newly_added_count += 1
            print(f"  ➕ New hit added to Shooter #{current_session_index + 1}. Total: {len(session['holes'])}")

    
    # --- BROADCAST & DRAW ACTIVE TEXT ---
    if newly_added_count > 0:
        await broadcast({
            "event": "hit_update",
            "bullet_count": len(session["holes"]),
        })

    # Draw active shooter text on the frame
    shooter_name = session.get("details", {}).get("studentName", "N/A")
    cv2.putText(processed_frame, 
                f"SHOOTER: {shooter_name} ({len(session['holes'])} Hits)", 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return processed_frame 


@app.get("/stop_session")
async def stop_session():
    global session_running, all_sessions, current_session_index
    
    if current_session_index == -1:
        # Force session_running to false just in case
        session_running = False 
        return {"status": "session ended", "error": "No active session to stop."}

    session_running = False

    session_data = all_sessions[current_session_index]
    holes = session_data["holes"]
    
    total_bullets = len(holes)
    group_size_inch = calculate_group_size(holes) 

    # --- SAVE RESULT ---
    session_result = {
        "timestamp": datetime.now().isoformat(),
        "student_name": session_data["details"].get("studentName", "N/A"),
        "service_no": session_data["details"].get("serviceNo", "N/A"),
        "service_year": session_data["details"].get("serviceYear", "N/A"),
        "total_bullets": total_bullets,
        "group_size_inch": round(group_size_inch, 2), 
    }
    save_result_to_file(session_result)
    
    # Save final frame 
    if last_processed_frame is not None:
        save_path = os.path.join(SAVE_DIR, f"session_summary_{session_data['timestamp']}.jpg")
        cv2.imwrite(save_path, last_processed_frame)
        print(f"✅ Session image saved: {save_path}")

    # --- BROADCAST & RESET ---
    await broadcast({
        "event": "session_ended",
        "result": session_result
    })

    current_session_index = -1
    
    print("✅ STOP SESSION endpoint FINISHED and returning success.") 
    
    return {"status": "session ended", "result": session_result}


def generate_frames():
    global cap, last_processed_frame
    
    while True:
        # Robustness: Check if camera is still closed/disconnected
        if not cap.isOpened():
            # Create a black placeholder frame for the client to display
            # This prevents the client-side <img> tag from failing completely
            placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder_frame, 
                        "CAMERA DISCONNECTED", 
                        (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', placeholder_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1) # Slow down polling if camera is off
            continue
            
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue
        
        # Run synchronous function with asyncio.run (used here because generator must be sync)
        try:
            processed_frame = asyncio.run(process_frame_for_detection(frame))
        except Exception as e:
            # Handle deep errors in detection logic
            print(f"🛑 Error running detection task sync: {e}")
            processed_frame = frame
            
        last_processed_frame = processed_frame # Store the frame for save_image on stop

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.get("/video_feed")
async def video_feed():
    # Fix: Global declaration at the start of the function if it's going to be re-assigned
    global cap 
    
    if not cap.isOpened():
        # Attempt to reopen camera if it failed initially
        cap = cv2.VideoCapture(CAMERA_STREAM)
        if not cap.isOpened():
            # If still closed, the generator will provide the placeholder image
            pass 
            
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# Clean up function for graceful shutdown
@app.on_event("shutdown")
def shutdown_event():
    global cap
    if cap.isOpened():
        cap.release()
        print("Camera released on shutdown.")