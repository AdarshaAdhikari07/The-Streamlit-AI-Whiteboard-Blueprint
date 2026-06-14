import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
import mediapipe as mp

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

class WhiteboardProcessor(VideoTransformerBase):
    def __init__(self):
        self.xp, self.yp = 0, 0  
        self.canvas = None       
        self.clear_request = False
        
        # 1. INITIALIZE MODERN TASKS API STRUCTURE DIRECTLY IN WORKER THREAD
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=None # Bypasses custom file calls to load embedded weights
            ),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7
        )
        
        # Fallback initializer handling structural variations on headless containers
        try:
            self.detector = HandLandmarker.create_from_options(options)
            self.use_legacy_mode = False
        except Exception:
            # Safe runtime fallback path if system bindings revert to standard solutions maps
            try:
                self.legacy_hands = mp.solutions.hands.Hands(
                    static_image_mode=False, max_num_hands=1,
                    min_detection_confidence=0.7, min_tracking_confidence=0.7
                )
            except AttributeError:
                from mediapipe.python.solutions import hands as flat_hands
                self.legacy_hands = flat_hands.Hands(
                    static_image_mode=False, max_num_hands=1,
                    min_detection_confidence=0.7, min_tracking_confidence=0.7
                )
            self.use_legacy_mode = True

    def transform(self, frame):
        # 1. Convert WebRTC frame to numpy array (BGR)
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) 
        h, w, c = img.shape

        # 2. Local thread-safe canvas matrix surface initialization
        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

        if self.clear_request:
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
            self.clear_request = False

        # 3. Processing Pipeline Execution Block
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if not self.use_legacy_mode:
            # Modern Image format mapping pipeline
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            results = self.detector.detect(mp_image)
            has_landmarks = hasattr(results, 'hand_landmarks') and results.hand_landmarks
            landmarks_source = results.hand_landmarks if has_landmarks else []
        else:
            # Legacy solution process map fallback
            results = self.legacy_hands.process(img_rgb)
            has_landmarks = hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks
            landmarks_source = results.multi_hand_landmarks if has_landmarks else []

        if has_landmarks:
            for hand_landmarks in landmarks_source:
                # Map standard landmark node indexes cleanly regardless of engine mode
                if self.use_legacy_mode:
                    landmarks = hand_landmarks.landmark
                    idx_tip, idx_pip = landmarks[8], landmarks[6]
                    mid_tip, mid_pip = landmarks[12], landmarks[10]
                else:
                    idx_tip, idx_pip = hand_landmarks[8], hand_landmarks[6]
                    mid_tip, mid_pip = hand_landmarks[12], hand_landmarks[10]
                
                # Fetch positions mapped down to frame width and height resolution
                cx_idx, cy_idx = int(idx_tip.x * w), int(idx_tip.y * h)

                # Flag checking: Raised tracking analysis (Lower Y is higher on view canvas)
                index_up = idx_tip.y < idx_pip.y
                middle_up = mid_tip.y < mid_pip.y

                # MODE 1: Selection Mode (Index + Middle finger raised)
                if index_up and middle_up:
                    self.xp, self.yp = 0, 0 
                    cv2.circle(img, (cx_idx, cy_idx), 15, (255, 255, 255), cv2.FILLED)

                # MODE 2: Active Drawing Mode (Index finger only)
                elif index_up and not middle_up:
                    cv2.circle(img, (cx_idx, cy_idx), brush_thickness, current_color, cv2.FILLED)
                    
                    if self.xp == 0 and self.yp == 0:
                        self.xp, self.yp = cx_idx, cy_idx

                    if color_choice == "Eraser":
                        cv2.line(self.canvas, (self.xp, self.yp), (cx_idx, cy_idx), (0, 0, 0), brush_thickness * 2)
                    else:
                        cv2.line(self.canvas, (self.xp, self.yp), (cx_idx, cy_idx), current_color, brush_thickness)
                    
                    self.xp, self.yp = cx_idx, cy_idx
                else:
                    self.xp, self.yp = 0, 0
        else:
            self.xp, self.yp = 0, 0

        # 4. Alpha mask fusion layout blending step
        img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, self.canvas)

        return img

# Initialize WebRTC Stream layout
ctx = webrtc_streamer(
    key="whiteboard",
    video_processor_factory=WhiteboardProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

# Handle UI Clear Canvas securely
if st.sidebar.button("Clear Canvas"):
    if ctx.video_processor:
        ctx.video_processor.clear_request = True
        st.sidebar.success("Canvas reset successfully!")

st.markdown("""
---
### 🖐️ How to Use:
* **Two Fingers Up (Index + Middle):** Selection Mode. Hover without drawing. Moves your 'cursor'.
* **One Finger Up (Index Only):** Drawing Mode. Start painting in the air!
* Adjust color and brush size seamlessly using the **Sidebar Controls**.
""")
