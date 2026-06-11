import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
import mediapipe as mp

# Use standard module mapping for modern MediaPipe distributions
mp_hands = mp.solutions.hands

# Configure page settings
st.set_page_config(page_title="AI Virtual Whiteboard", layout="wide")
st.title("🎨 AI Virtual Whiteboard")
st.subheader("Draw in the air using your webcam and MediaPipe!")

# Sidebar Controls for the UI
st.sidebar.header("🎨 Canvas Settings")
color_choice = st.sidebar.selectbox("Select Color:", ["Blue", "Green", "Red", "Eraser"])
brush_thickness = st.sidebar.slider("Brush Thickness:", min_value=5, max_value=50, value=10)

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

# Secure the model instance via Streamlit resource caching across background frames
@st.cache_resource
def get_hands_detector():
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

hands = get_hands_detector()

class WhiteboardProcessor(VideoTransformerBase):
    def __init__(self):
        self.xp, self.yp = 0, 0  # Drawing tracking points anchor
        self.canvas = None       # THREAD-SAFE: Canvas stays isolated inside the video thread
        self.clear_request = False

    def transform(self, frame):
        # 1. Convert WebRTC frame to numpy array (BGR)
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror image matching
        h, w, c = img.shape

        # 2. Local canvas initialization (Prevents global thread collisions)
        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

        if self.clear_request:
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
            self.clear_request = False

        # 3. MediaPipe processing pipeline
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                
                # Convert normalized coordinates to pixel locations
                cx_idx, cy_idx = int(landmarks[8].x * w), int(landmarks[8].y * h)
                cx_mid, cy_mid = int(landmarks[12].x * w), int(landmarks[12].y * h)

                # Finger detection math checking
                index_up = landmarks[8].y < landmarks[6].y
                middle_up = landmarks[12].y < landmarks[10].y

                # MODE 1: Selection Mode (Index + Middle finger raised)
                if index_up and middle_up:
                    self.xp, self.yp = 0, 0 # Lift drawing pen
                    cv2.circle(img, (cx_idx, cy_idx), 15, (255, 255, 255), cv2.FILLED)

                # MODE 2: Active Drawing Mode (Index finger only)
                elif index_up and not middle_up:
                    cv2.circle(img, (cx_idx, cy_idx), brush_thickness, current_color, cv2.FILLED)
                    
                    if self.xp == 0 and self.yp == 0:
                        self.xp, self.yp = cx_idx, cy_idx

                    # Apply line logic directly onto the localized thread canvas matrix
                    if color_choice == "Eraser":
                        cv2.line(self.canvas, (self.xp, self.yp), (cx_idx, cy_idx), (0, 0, 0), brush_thickness * 2)
                    else:
                        cv2.line(self.canvas, (self.xp, self.yp), (cx_idx, cy_idx), current_color, brush_thickness)
                    
                    self.xp, self.yp = cx_idx, cy_idx
                else:
                    self.xp, self.yp = 0, 0
        else:
            self.xp, self.yp = 0, 0

        # 4. Alpha mask fusion: Layer canvas overlay above original camera stream
        img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, self.canvas)

        return img

# Initialize WebRTC Stream layout component and link context
ctx = webrtc_streamer(
    key="whiteboard",
    video_processor_factory=WhiteboardProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

# Handle UI Clear Canvas events out of the frame processing pathway securely
if st.sidebar.button("Clear Canvas"):
    if ctx.video_processor:
        ctx.video_processor.clear_request = True
        st.sidebar.success("Canvas reset successfully!")

# App interface guide text instructions
st.markdown("""
---
### 🖐️ How to Use:
* **Two Fingers Up (Index + Middle):** Selection Mode. Hover without drawing. Moves your 'cursor'.
* **One Finger Up (Index Only):** Drawing Mode. Start painting in the air!
* Adjust color and brush size seamlessly using the **Sidebar Controls**.
""")
