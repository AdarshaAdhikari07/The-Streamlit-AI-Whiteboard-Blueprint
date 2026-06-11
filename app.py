import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import mediapipe as mp
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
    "Eraser": (0, 0, 0) # Erasing acts as drawing black (or matching canvas)
}
current_color = color_map[color_choice]

# WebRTC configuration for STUN servers (needed for cloud deployment deployment)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Global/Session persistent canvas array initialization
if "canvas" not in st.session_state or st.session_state["canvas"] is None:
    # We will initialize this dynamically inside the frame processing loop 
    # once we know the exact frame dimensions from the webcam.
    st.session_state["canvas"] = None

# MediaPipe Hands Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

class WhiteboardProcessor(VideoTransformerBase):
    def __init__(self):
        self.xp, self.yp = 0, 0  # Previous coordinates

    def transform(self, frame):
        # 1. Convert WebRTC frame to numpy array (BGR)
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Flip horizontally for natural mirror effect
        h, w, c = img.shape

        # 2. Initialize or resize canvas to match the exact video frame dimensions
        if st.session_state["canvas"] is None or st.session_state["canvas"].shape[:2] != (h, w):
            st.session_state["canvas"] = np.zeros((h, w, 3), dtype=np.uint8)

        # 3. MediaPipe Process
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmark coordinates
                landmarks = hand_landmarks.landmark
                
                # Get Index finger tip (ID 8) and Middle finger tip (ID 12)
                cx_idx, cy_idx = int(landmarks[8].x * w), int(landmarks[8].y * h)
                cx_mid, cy_mid = int(landmarks[12].x * w), int(landmarks[12].y * h)

                # Check which fingers are up (Simple logic: tip higher than joint)
                index_up = landmarks[8].y < landmarks[6].y
                middle_up = landmarks[12].y < landmarks[10].y

                # MODE 1: Selection Mode (Two fingers up) -> Stop drawing, track position
                if index_up and middle_up:
                    self.xp, self.yp = 0, 0 # Reset drawing anchors
                    cv2.circle(img, (cx_idx, cy_idx), 15, (255, 255, 255), cv2.FILLED)

                # MODE 2: Drawing Mode (Only Index finger up)
                elif index_up and not middle_up:
                    cv2.circle(img, (cx_idx, cy_idx), brush_thickness, current_color, cv2.FILLED)
                    
                    if self.xp == 0 and self.yp == 0:
                        self.xp, self.yp = cx_idx, cy_idx

                    # Draw on our persistent canvas session state
                    if color_choice == "Eraser":
                        cv2.line(st.session_state["canvas"], (self.xp, self.yp), (cx_idx, cy_idx), (0, 0, 0), brush_thickness * 2)
                    else:
                        cv2.line(st.session_state["canvas"], (self.xp, self.yp), (cx_idx, cy_idx), current_color, brush_thickness)
                    
                    self.xp, self.yp = cx_idx, cy_idx
                else:
                    self.xp, self.yp = 0, 0
        else:
            self.xp, self.yp = 0, 0

        # 4. Merge canvas drawings onto the webcam feed
        img_gray = cv2.cvtColor(st.session_state["canvas"], cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, st.session_state["canvas"])

        return img

# Render the WebRTC Streamer component in the main area
webrtc_streamer(
    key="whiteboard",
    video_processor_factory=WhiteboardProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

# Instructions for users
st.markdown("""
---
### 🖐️ How to Use:
* **Two Fingers Up (Index + Middle):** Selection Mode. Hover without drawing. Moves your 'cursor'.
* **One Finger Up (Index Only):** Drawing Mode. Start painting in the air!
* Adjust color and brush size seamlessly using the **Sidebar Controls**.
""")
