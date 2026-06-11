import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np

# 1. FORCE THE DIRECT SUB-MODULE IMPORTS TO BYPASS LAZY LOADING ISSUES
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing

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

# WebRTC configuration for STUN servers
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Initialize canvas state securely
if "canvas" not in st.session_state:
    st.session_state["canvas"] = None

# 2. USE CACHING TO INITIALIZE THE DETECTOR SECURELY ACROSS THREADS
@st.cache_resource
def load_hands_model():
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

hands = load_hands_model()

class WhiteboardProcessor(VideoTransformerBase):
    def __init__(self):
        self.xp, self.yp = 0, 0  # Drawing coordinate tracking anchor

    def transform(self, frame):
        # Convert WebRTC frame to numpy array (BGR)
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror image matching
        h, w, c = img.shape

        # Initialize the persistent canvas overlay surface matrix
        if st.session_state["canvas"] is None or st.session_state["canvas"].shape[:2] != (h, w):
            st.session_state["canvas"] = np.zeros((h, w, 3), dtype=np.uint8)

        # Process the frame frame through the cached hands instance
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                
                # Fetch index tip (ID 8) and middle tip (ID 12) positions
                cx_idx, cy_idx = int(landmarks[8].x * w), int(landmarks[8].y * h)
                cx_mid, cy_mid = int(landmarks[12].x * w), int(landmarks[12].y * h)

                # Flag checking: Is tip Y lower than joint Y? (Lower Y value means higher on screen)
                index_up = landmarks[8].y < landmarks[6].y
                middle_up = landmarks[12].y < landmarks[10].y

                # MODE 1: Selection/Hover Mode (Both Index and Middle fingers raised)
                if index_up and middle_up:
                    self.xp, self.yp = 0, 0 # Break drawing link line
                    cv2.circle(img, (cx_idx, cy_idx), 15, (255, 255, 255), cv2.FILLED)

                # MODE 2: Active Drawing Mode (Only Index finger raised)
                elif index_up and not middle_up:
                    cv2.circle(img, (cx_idx, cy_idx), brush_thickness, current_color, cv2.FILLED)
                    
                    if self.xp == 0 and self.yp == 0:
                        self.xp, self.yp = cx_idx, cy_idx

                    # Apply drawing straight onto session state canvas overlay mask array
                    if color_choice == "Eraser":
                        cv2.line(st.session_state["canvas"], (self.xp, self.yp), (cx_idx, cy_idx), (0, 0, 0), brush_thickness * 2)
                    else:
                        cv2.line(st.session_state["canvas"], (self.xp, self.yp), (cx_idx, cy_idx), current_color, brush_thickness)
                    
                    self.xp, self.yp = cx_idx, cy_idx
                else:
                    self.xp, self.yp = 0, 0
        else:
            self.xp, self.yp = 0, 0

        # Bitwise masking step: Blend the canvas overlay directly onto live camera video stream
        img_gray = cv2.cvtColor(st.session_state["canvas"], cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, st.session_state["canvas"])

        return img

# Mount the video streaming handler frame engine
webrtc_streamer(
    key="whiteboard",
    video_processor_factory=WhiteboardProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

# Instructions markdown template
st.markdown("""
---
### 🖐️ How to Use:
* **Two Fingers Up (Index + Middle):** Selection Mode. Hover without drawing. Moves your 'cursor'.
* **One Finger Up (Index Only):** Drawing Mode. Start painting in the air!
* Adjust color and brush size seamlessly using the **Sidebar Controls**.
""")
