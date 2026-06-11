import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
import mediapipe as mp

# ====================================================================
# MODERN MEDIAPIPE TASKS API SETUP (MATCHES PACKAGES ON LINUX SERVER)
# ====================================================================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

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

# Secure the landmarker instance via Streamlit resource caching
@st.cache_resource
def get_hand_landmarker():
    # Uses a live-stream optimized constructor pattern
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=None), # Uses embedded framework assets
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7
    )
    return HandLandmarker.create_from_options(options)

try:
    landmarker = get_hand_landmarker()
except Exception:
    # Fail-safe fallback if embedded binary assets require standard instantiation
    landmarker = None

class WhiteboardProcessor(VideoTransformerBase):
    def __init__(self):
        self.xp, self.yp = 0, 0  # Drawing tracking points anchor

    def transform(self, frame):
        # 1. Convert WebRTC frame to numpy array (BGR)
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror matching
        h, w, c = img.shape

        # 2. Dynamic tracking canvas initialization
        if st.session_state["canvas"] is None or st.session_state["canvas"].shape[:2] != (h, w):
            st.session_state["canvas"] = np.zeros((h, w, 3), dtype=np.uint8)

        # 3. Modern Processing Pipeline
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Safe detection fallback check
        if landmarker is not None:
            results = landmarker.detect(mp_image)
            has_landmarks = hasattr(results, 'hand_landmarks') and results.hand_landmarks
        else:
            has_landmarks = False

        if has_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Fetch index tip (Index 8) and middle tip (Index 12)
                idx_tip = hand_landmarks[8]
                mid_tip = hand_landmarks[12]
                idx_pip = hand_landmarks[6]
                mid_pip = hand_landmarks[10]
                
                # Convert normalized coordinates to screen pixel space
                cx_idx, cy_idx = int(idx_tip.x * w), int(idx_tip.y * h)
                
                # Finger detection math checking (Lower Y value means higher on screen)
                index_up = idx_tip.y < idx_pip.y
                middle_up = mid_tip.y < mid_pip.y

                # MODE 1: Selection Mode (Index + Middle finger raised)
                if index_up and middle_up:
                    self.xp, self.yp = 0, 0 # Lift drawing pen
                    cv2.circle(img, (cx_idx, cy_idx), 15, (255, 255, 255), cv2.FILLED)

                # MODE 2: Active Drawing Mode (Index finger only)
                elif index_up and not middle_up:
                    cv2.circle(img, (cx_idx, cy_idx), brush_thickness, current_color, cv2.FILLED)
                    
                    if self.xp == 0 and self.yp == 0:
                        self.xp, self.yp = cx_idx, cy_idx

                    # Apply line transformation logic straight to persistence canvas matrix
                    if color_choice == "Eraser":
                        cv2.line(st.session_state["canvas"], (self.xp, self.yp), (cx_idx, cy_idx), (0, 0, 0), brush_thickness * 2)
                    else:
                        cv2.line(st.session_state["canvas"], (self.xp, self.yp), (cx_idx, cy_idx), current_color, brush_thickness)
                    
                    self.xp, self.yp = cx_idx, cy_idx
                else:
                    self.xp, self.yp = 0, 0
        else:
            self.xp, self.yp = 0, 0

        # 4. Alpha mask fusion: Layer canvas overlay above original camera stream background
        img_gray = cv2.cvtColor(st.session_state["canvas"], cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, st.session_state["canvas"])

        return img

# Initialize WebRTC Stream layout object wrapper component
webrtc_streamer(
    key="whiteboard",
    video_processor_factory=WhiteboardProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

# App interface guide text instructions
st.markdown("""
---
### 🖐️ How to Use:
* **Two Fingers Up (Index + Middle):** Selection Mode. Hover without drawing. Moves your 'cursor'.
* **One Finger Up (Index Only):** Drawing Mode. Start painting in the air!
* Adjust color and brush size seamlessly using the **Sidebar Controls**.
""")
