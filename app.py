import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import mediapipe as mp
# FIX: Explicit sub-module import to bypass Linux namespace bugs on headless cloud servers
from mediapipe.python.solutions import hands as mp_hands
import numpy as np

# Configure page settings
st.set_page_config(page_title="AI Virtual Whiteboard", layout="wide")
st.title("🎨 AI Virtual Whiteboard")
st.subheader("Draw in the air using your webcam and MediaPipe!")

# Sidebar Controls for the UI
st.sidebar.header("🎨 Canvas Settings")
color_choice = st.sidebar.selectbox("Select Color:", ["Blue", "Green", "Red", "Eraser"])
brush_thickness = st.sidebar.slider("Brush Thickness:", min_value=5, max_value=50, value=10)

if st.sidebar.button("Clear Canvas"):
    st.session_state["canvas"] = None
    st.success("Canvas cleared!")

# Map colors to BGR (OpenCV format)
color_map = {
    "Blue": (255, 0, 0),
    "Green": (0, 255, 0),
    "Red": (0, 0, 255),
    "Eraser": (0, 0, 0)
}
current_color = color_map[color_choice]

# WebRTC configuration for STUN servers (needed for secure cloud streaming)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Initialize canvas state securely
if "canvas" not in st.session_state:
    st.session_state["canvas"] = None

# MediaPipe Hands Setup using the direct module reference
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

class WhiteboardProcessor(VideoTransformerBase):
    def __init__(self):
        self.xp, self.yp = 0, 0  # Previous drawing coordinates tracking

    def transform(self, frame):
        # 1. Convert WebRTC frame to numpy array (BGR)
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror image for natural movement matching
        h, w, c = img.shape

        # 2. Lazy initialization of the persistent drawing matrix canvas
        if st.session_state["canvas"] is None or st.session_state["canvas"].shape[:2] != (h, w):
            st.session_state["canvas"] = np.zeros((h, w, 3), dtype=np.uint8)

        # 3. Process the frame through MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                
                # Fetch index tip (ID 8) and middle tip (ID 12) normalized coordinates
                cx_idx, cy_idx = int(landmarks[8].x * w), int(landmarks[8].y * h)
                cx_mid, cy_mid = int(landmarks[12].x * w), int(landmarks[12].y * h)

                # Flag check: Is tip coordinate higher than the knuckle joint coordinate?
                index_up = landmarks[8].y < landmarks[6].y
                middle_up = landmarks[12].y < landmarks[10].y

                # MODE 1: Selection/Hover Mode (Both Index and Middle fingers raised)
                if index_up and middle_up:
                    self.xp, self.yp = 0, 0 # Lift brush anchor
