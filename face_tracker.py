# face_tracker.py
import cv2
import mediapipe as mp
import time
import numpy as np
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION, FACEMESH_CONTOURS, FACEMESH_IRISES

# --- Helper Functions ---
def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points (2D or 3D)."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_vector(p1, p2):
    """Calculates the vector from p1 to p2."""
    return np.array(p2) - np.array(p1)

# --- Face Specific Landmark IDs ---
# These are approximate IDs based on the MediaPipe Face Mesh documentation
# See: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
LEFT_EYE_IDS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDS = [33, 160, 158, 133, 153, 144]
MOUTH_OUTER_IDS = [61, 291, 0, 17, 37, 78, 146, 308, 405, 181, 39, 80] # Approximate outer lip contour
MOUTH_INNER_IDS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324] # Approximate inner lip contour
LEFT_EYEBROW_IDS = [336, 296, 334, 293, 300] # Upper line
RIGHT_EYEBROW_IDS = [107, 66, 105, 63, 70]  # Upper line
NOSE_TIP_ID = 4
CHIN_ID = 152
LEFT_EYE_CENTER_APPROX = 130 # Rough center landmarks for reference, not official iris
RIGHT_EYE_CENTER_APPROX = 359

# For Head Pose Estimation
# A simplified set of 3D model points (based on canonical model, units are arbitrary but relative)
# Using a subset that is relatively stable
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (4)
    (0.0, -330.0, -65.0),        # Chin (152)
    (-225.0, 170.0, -135.0),       # Left eye left corner (33) - approximation
    (225.0, 170.0, -135.0),        # Right eye right corner (263) - approximation
    (-150.0, -150.0, -125.0),      # Left Mouth corner (61)
    (150.0, -150.0, -125.0)       # Right mouth corner (291)
], dtype=np.float64)

# Corresponding Landmark IDs for the MODEL_POINTS
MODEL_POINTS_LANDMARK_IDS = [4, 152, 33, 263, 61, 291]


class FaceTracker:
    def __init__(self, config):
        """Initializes the face tracker with MediaPipe Face Mesh."""
        self.config = config.get('face_mesh', {}) # Get face specific config
        self.display_config = config.get('display', {})
        self.camera_config = config.get('camera', {})

        self.mpFaceMesh = mp.solutions.face_mesh
        try:
            self.face_mesh = self.mpFaceMesh.FaceMesh(
                max_num_faces=self.config.get('max_num_faces', 1),
                refine_landmarks=self.config.get('refine_landmarks', True), # Use refined landmarks for irises etc.
                min_detection_confidence=self.config.get('min_detection_confidence', 0.5),
                min_tracking_confidence=self.config.get('min_tracking_confidence', 0.5)
            )
            print(f"MediaPipe FaceMesh initialized for up to {self.config.get('max_num_faces', 1)} face(s).")
        except TypeError as e:
            print(f"Warning: MediaPipe FaceMesh parameters might have changed ({e}). Using defaults.")
            self.face_mesh = self.mpFaceMesh.FaceMesh() # Fallback

        self.mpDraw = mp.solutions.drawing_utils
        self.drawing_spec_landmarks = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(0,200,0))
        self.drawing_spec_contours = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(80,180,80))
        self.drawing_spec_irises = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(0,220,220))
        self.drawing_spec_tesselation = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(192, 192, 192))

        self.results = None
        self.frame_height = self.camera_config.get('height', 480) # Initial estimates
        self.frame_width = self.camera_config.get('width', 640)

        # Camera matrix for solvePnP (simple estimation)
        self.focal_length = self.frame_width
        self.camera_center = (self.frame_width / 2, self.frame_height / 2)
        self.camera_matrix = np.array([
            [self.focal_length, 0, self.camera_center[0]],
            [0, self.focal_length, self.camera_center[1]],
            [0, 0, 1]], dtype="double")
        self.dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion

        print("FaceTracker initialized.")

    def _get_default_tracking_data(self):
        """Returns a dictionary with default values for face tracking metrics."""
        return {
            'found': False,
            'head_pitch': 0.0, # Radians, relative to camera forward
            'head_yaw': 0.0,   # Radians
            'head_roll': 0.0,  # Radians
            'left_ear': 0.35,  # Eye Aspect Ratio (typical open value)
            'right_ear': 0.35, # Eye Aspect Ratio
            'mar': 0.0,        # Mouth Aspect Ratio (closed)
            'jaw_openness': 0.0, # Normalized vertical distance inner lips
            'left_eyebrow_height': 0.0, # Normalized distance eyebrow to eye line
            'right_eyebrow_height': 0.0,
            'landmarks_pixels': [], # List of (x, y) pixel coordinates (all 478)
            'landmarks_normalized': [], # List of (x, y, z) normalized coordinates
            'raw_face_landmarks': None, # Store the raw landmarks object
            # Could add head translation x, y, z if needed
        }

    def _calculate_ear(self, eye_landmarks_pixels):
        """Calculates the Eye Aspect Ratio (EAR) for a single eye."""
        try:
            # Vertical distances
            v1 = calculate_distance(eye_landmarks_pixels[1], eye_landmarks_pixels[5]) # P2-P6
            v2 = calculate_distance(eye_landmarks_pixels[2], eye_landmarks_pixels[4]) # P3-P5
            # Horizontal distance
            h = calculate_distance(eye_landmarks_pixels[0], eye_landmarks_pixels[3]) # P1-P4
            if h < 1e-6: return 0.0 # Avoid division by zero
            ear = (v1 + v2) / (2.0 * h)
            return ear
        except IndexError:
            return 0.0 # Not enough landmarks

    def _calculate_mar(self, mouth_landmarks_pixels):
        """Calculates the Mouth Aspect Ratio (MAR)."""
        try:
            # Vertical distance (simplified: top inner lip to bottom inner lip midpoint)
            # Use inner lip landmarks: e.g., 13 (top) and 14 (bottom) for vertical center
            # Use corner landmarks: e.g., 61 (left) and 291 (right) for horizontal
            p_top = mouth_landmarks_pixels[13] # Upper lip inner center
            p_bottom = mouth_landmarks_pixels[14] # Lower lip inner center
            p_left = mouth_landmarks_pixels[61] # Left corner
            p_right = mouth_landmarks_pixels[291] # Right corner

            vertical_dist = calculate_distance(p_top, p_bottom)
            horizontal_dist = calculate_distance(p_left, p_right)

            if horizontal_dist < 1e-6: return 0.0
            mar = vertical_dist / horizontal_dist
            return mar
        except IndexError:
             # Fallback using outer points if needed (less precise for MAR)
            try:
                 # Vertical distance approx (0 <-> 17)
                 # Horizontal distance (61 <-> 291)
                vertical_dist = calculate_distance(mouth_landmarks_pixels[0], mouth_landmarks_pixels[17])
                horizontal_dist = calculate_distance(mouth_landmarks_pixels[61], mouth_landmarks_pixels[291])
                if horizontal_dist < 1e-6: return 0.0
                mar = vertical_dist / horizontal_dist
                return mar
            except IndexError:
                return 0.0

    def _calculate_jaw_openness(self, landmarks_pixels, ref_distance):
        """Calculates normalized distance between inner upper and lower lip centers."""
        try:
            p_top = landmarks_pixels[13]
            p_bottom = landmarks_pixels[14]
            dist = calculate_distance(p_top, p_bottom)
            if ref_distance < 1e-6: return 0.0
            return dist / ref_distance
        except IndexError:
            return 0.0

    def _calculate_eyebrow_height(self, eyebrow_landmarks_pixels, eye_landmarks_pixels, ref_distance):
        """Calculates normalized distance between eyebrow midpoint and eye horizontal line."""
        try:
            eyebrow_mid = np.mean(np.array(eyebrow_landmarks_pixels), axis=0)
            eye_p1 = np.array(eye_landmarks_pixels[0]) # Eye corner
            eye_p4 = np.array(eye_landmarks_pixels[3]) # Other eye corner
            eye_center_y = (eye_p1[1] + eye_p4[1]) / 2.0

            height = abs(eye_center_y - eyebrow_mid[1]) # Vertical distance in pixels
            if ref_distance < 1e-6: return 0.0
            return height / ref_distance
        except (IndexError, ValueError):
            return 0.0


    def process_frame(self, frame):
        """Processes a single video frame to find face landmarks and calculate metrics."""
        self.frame_height, self.frame_width, _ = frame.shape # Get actual dimensions

        # Update camera matrix if dimensions changed
        if self.frame_width != self.camera_center[0]*2 or self.frame_height != self.camera_center[1]*2:
            self.focal_length = self.frame_width
            self.camera_center = (self.frame_width / 2, self.frame_height / 2)
            self.camera_matrix = np.array([
                [self.focal_length, 0, self.camera_center[0]],
                [0, self.focal_length, self.camera_center[1]],
                [0, 0, 1]], dtype="double")


        # Optional horizontal flip (consistent with hand tracker)
        if self.display_config.get('flip_horizontal', True):
            frame = cv2.flip(frame, 1)

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgRGB.flags.writeable = False # Optimization

        self.results = self.face_mesh.process(imgRGB)

        imgRGB.flags.writeable = True

        # Initialize tracking data (only supports one face for now)
        face_data = self._get_default_tracking_data()

        if self.results.multi_face_landmarks:
            # Process the first detected face
            face_landmarks = self.results.multi_face_landmarks[0]
            face_data['found'] = True
            face_data['raw_face_landmarks'] = face_landmarks # Store for drawing

            # Store landmarks (using normalized for consistency, pixels for calculations)
            landmarks_norm = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
            landmarks_pixels = np.array([(lm.x * self.frame_width, lm.y * self.frame_height) for lm in face_landmarks.landmark])

            face_data['landmarks_pixels'] = landmarks_pixels.tolist() # Store pixel coordinates
            face_data['landmarks_normalized'] = landmarks_norm.tolist()

            # --- Calculate Metrics ---
            try:
                # Reference distance (e.g., between eye corners) for normalization
                ref_dist = calculate_distance(landmarks_pixels[LEFT_EYE_IDS[0]], landmarks_pixels[RIGHT_EYE_IDS[3]]) # Approx inter-pupillary distance proxy
                if ref_dist < 1e-3: ref_dist = 1.0 # Avoid zero division

                # 1. Eye Aspect Ratios (EAR)
                left_eye_lm = landmarks_pixels[LEFT_EYE_IDS]
                right_eye_lm = landmarks_pixels[RIGHT_EYE_IDS]
                face_data['left_ear'] = self._calculate_ear(left_eye_lm)
                face_data['right_ear'] = self._calculate_ear(right_eye_lm)

                # 2. Mouth Aspect Ratio (MAR)
                # Use all landmarks for MAR calculation method
                face_data['mar'] = self._calculate_mar(landmarks_pixels)

                # 3. Jaw Openness
                face_data['jaw_openness'] = self._calculate_jaw_openness(landmarks_pixels, ref_dist)

                # 4. Eyebrow Heights
                left_eyebrow_lm = landmarks_pixels[LEFT_EYEBROW_IDS]
                right_eyebrow_lm = landmarks_pixels[RIGHT_EYEBROW_IDS]
                face_data['left_eyebrow_height'] = self._calculate_eyebrow_height(left_eyebrow_lm, left_eye_lm, ref_dist)
                face_data['right_eyebrow_height'] = self._calculate_eyebrow_height(right_eyebrow_lm, right_eye_lm, ref_dist)


                # 5. Head Pose Estimation (Pitch, Yaw, Roll)
                # Select 2D image points corresponding to the 3D model points
                image_points = np.array([landmarks_pixels[p] for p in MODEL_POINTS_LANDMARK_IDS], dtype="double")

                if len(image_points) == len(MODEL_POINTS):
                    (success, rotation_vector, translation_vector) = cv2.solvePnP(
                        MODEL_POINTS, image_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE) # Or SOLVEPNP_SQPNP

                    if success:
                        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

                        # Combine with camera projection matrix (useful for projecting 3D points)
                        # projection_matrix = np.dot(self.camera_matrix, np.hstack((rotation_matrix, translation_vector)))

                        # Decompose rotation matrix to Euler angles
                        # Note: Order matters (e.g., XYZ, YXZ, etc.)
                        # cv2.decomposeProjectionMatrix gives angles but might need adjustment
                        # Alternative: manual calculation (e.g., from learnopencv.com)
                        sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] +  rotation_matrix[1,0] * rotation_matrix[1,0])
                        singular = sy < 1e-6

                        if not singular:
                            x = np.arctan2(rotation_matrix[2,1] , rotation_matrix[2,2]) # Roll
                            y = np.arctan2(-rotation_matrix[2,0], sy)                   # Yaw
                            z = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0]) # Pitch
                        else:
                            x = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1]) # Roll
                            y = np.arctan2(-rotation_matrix[2,0], sy)                   # Yaw
                            z = 0                                                       # Pitch

                        # Adjust angles based on coordinate system conventions if needed
                        # Our setup: Pitch (around X), Yaw (around Y), Roll (around Z)
                        # Convert to conventional ranges if desired (e.g., Yaw -180 to +180)
                        face_data['head_pitch'] = float(z) # Rotation around X-axis (nodding)
                        face_data['head_yaw'] = float(y)   # Rotation around Y-axis (shaking head)
                        face_data['head_roll'] = float(x)  # Rotation around Z-axis (tilting head)

                        # Store translation vector if needed (position relative to camera)
                        # face_data['head_x'] = translation_vector[0, 0]
                        # face_data['head_y'] = translation_vector[1, 0]
                        # face_data['head_z'] = translation_vector[2, 0] # Distance from camera

            except Exception as e:
                # print(f"Warning: Could not calculate face metrics: {e}")
                pass # Keep defaults if any calculation fails

        return frame, face_data

    def draw_visuals(self, frame, face_data):
        """Draws landmarks and mesh onto the frame for the detected face."""
        if face_data['found'] and face_data['raw_face_landmarks']:
            # Draw standard contours and irises if landmarks/connections enabled
            if self.display_config.get('draw_landmarks', True) or self.display_config.get('draw_connections', True):
                # Draw Contours
                if self.display_config.get('draw_connections', True):
                    self.mpDraw.draw_landmarks(
                        image=frame,
                        landmark_list=face_data['raw_face_landmarks'],
                        connections=FACEMESH_CONTOURS,
                        landmark_drawing_spec=None, # Use spec below if landmarks enabled
                        connection_drawing_spec=self.drawing_spec_contours)
                    # Draw Irises (if refined landmarks are enabled in config)
                    if self.config.get('refine_landmarks', True):
                         self.mpDraw.draw_landmarks(
                             image=frame,
                             landmark_list=face_data['raw_face_landmarks'],
                             connections=FACEMESH_IRISES,
                             landmark_drawing_spec=None,
                             connection_drawing_spec=self.drawing_spec_irises)

                # Draw Landmark points themselves (if enabled)
                # Draw these *after* contours so they appear on top if small radius
                if self.display_config.get('draw_landmarks', True):
                     # Draw all landmarks using the spec (includes those in contours/irises)
                     # This covers cases where only landmarks are shown, not connections
                    self.mpDraw.draw_landmarks(
                        image=frame,
                        landmark_list=face_data['raw_face_landmarks'],
                        connections=None, # Only draw points
                        landmark_drawing_spec=self.drawing_spec_landmarks,
                        connection_drawing_spec=None)


            # Draw the dense Tesselation ONLY if specifically enabled (performance hit)
            if self.display_config.get('draw_face_tesselation', False):
                self.mpDraw.draw_landmarks(
                    image=frame,
                    landmark_list=face_data['raw_face_landmarks'],
                    connections=FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.drawing_spec_tesselation)


            # Optional: Draw Head Pose axes (skipped for brevity)
        return frame

    def close(self):
        """Releases resources."""
        if hasattr(self, 'face_mesh') and hasattr(self.face_mesh, 'close'):
             self.face_mesh.close()
        print("FaceTracker closed.")