import cv2
import mediapipe as mp
import time
import numpy as np

def calculate_distance(p1, p2):
    """Calculates 3D Euclidean distance between two points."""
    return np.linalg.norm(p1 - p2)

def calculate_vector(p1, p2):
    """Calculates the vector from p1 to p2."""
    return p2 - p1

def calculate_angle_between_vectors(v1, v2):
    """Calculates the angle in radians between two vectors."""
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return 0.0 # Avoid division by zero
    unit_v1 = v1 / norm_v1
    unit_v2 = v2 / norm_v2
    dot_product = np.dot(unit_v1, unit_v2)
    # Clip dot product to avoid numerical errors with arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    return np.arccos(dot_product)


class HandTracker:
    def __init__(self, config):
        """Initializes the hand tracker with MediaPipe."""
        self.config = config['mediapipe']
        self.display_config = config['display']
        self.camera_config = config['camera']

        self.mpHands = mp.solutions.hands
        try:
            self.hands = self.mpHands.Hands(
                static_image_mode=self.config.get('static_image_mode', False),
                max_num_hands=self.config.get('max_num_hands', 1),
                min_detection_confidence=self.config.get('min_detection_confidence', 0.5),
                min_tracking_confidence=self.config.get('min_tracking_confidence', 0.5)
            )
            print(f"MediaPipe Hands initialized for up to {self.config.get('max_num_hands', 1)} hands.")
        except TypeError as e:
            print(f"Warning: MediaPipe Hands parameters might have changed ({e}). Using defaults.")
            self.hands = self.mpHands.Hands()

        self.mpDraw = mp.solutions.drawing_utils
        self.results = None
        self.frame_height = self.camera_config['height']
        self.frame_width = self.camera_config['width']

        print("HandTracker initialized.")

    def _get_default_tracking_data(self):
        """Returns a dictionary with default values for tracking metrics."""
        return {
            'found': False,
            'handedness': None, # 'Left' or 'Right'
            'centroid_x': 0.0,
            'centroid_y': 0.0,
            'wrist_x': 0.0,
            'wrist_y': 0.0,
            'wrist_z': 0.0, # Raw Z value from MediaPipe, smaller is closer
            'hand_pitch': 0.0, # Radians
            'hand_yaw': 0.0, # Radians
            'hand_roll': 0.0, # Radians
            'thumb_angle_curl': np.pi, # Default to straight (radians)
            'index_angle_curl': np.pi,
            'middle_angle_curl': np.pi,
            'ring_angle_curl': np.pi,
            'pinky_angle_curl': np.pi,
            'thumb_index_spread': 0.0, # Radians (Angle at wrist)
            'index_middle_spread': 0.0,
            'middle_ring_spread': 0.0,
            'ring_pinky_spread': 0.0,
            'thumb_index_pinch': 0.0, # Normalized distance (Tip to Tip)
            'thumb_middle_pinch': 0.0,
            'thumb_ring_pinch': 0.0,
            'thumb_pinky_pinch': 0.0,
            'all_landmarks_pixels': [], # List of (x, y) pixel coordinates
            'all_landmarks_normalized': [], # List of (x, y, z) normalized coordinates
            'raw_hand_landmarks': None, # Store the raw landmarks object for drawing
            'raw_handedness_info': None # Store raw handedness info
        }


    def process_frame(self, frame):
        """Processes a single video frame to find hand landmarks and calculate metrics for each hand."""
        self.frame_height, self.frame_width, _ = frame.shape

        # Optional horizontal flip for intuitive control
        if self.display_config.get('flip_horizontal', True):
            frame = cv2.flip(frame, 1)

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgRGB.flags.writeable = False

        self.results = self.hands.process(imgRGB)

        imgRGB.flags.writeable = True

        all_hands_data = []

        if self.results.multi_hand_landmarks and self.results.multi_handedness:
            # Limit to the configured max_num_hands
            num_hands_to_process = min(len(self.results.multi_hand_landmarks), self.config.get('max_num_hands', 1))

            for i in range(num_hands_to_process):
                handLms = self.results.multi_hand_landmarks[i]
                handedness_info = self.results.multi_handedness[i]

                hand_data = self._get_default_tracking_data()
                hand_data['found'] = True
                hand_data['raw_hand_landmarks'] = handLms
                hand_data['raw_handedness_info'] = handedness_info

                try:
                    hand_data['handedness'] = handedness_info.classification[0].label
                except (IndexError, AttributeError):
                     print("Warning: Could not determine handedness.")
                     hand_data['handedness'] = "Unknown"

                landmarks_norm = np.array([(lm.x, lm.y, lm.z) for lm in handLms.landmark])
                landmarks_pixels = np.array([(int(lm.x * self.frame_width), int(lm.y * self.frame_height)) for lm in handLms.landmark])

                hand_data['all_landmarks_pixels'] = landmarks_pixels.tolist()
                hand_data['all_landmarks_normalized'] = landmarks_norm.tolist()

                centroid_lm_id = self.config.get('centroid_landmark_id', 0)
                try:
                    # Ensure landmark ID is within the valid range (0-20)
                    valid_centroid_lm_id = max(0, min(20, centroid_lm_id))
                    if valid_centroid_lm_id != centroid_lm_id:
                        print(f"Warning: Centroid landmark ID {centroid_lm_id} out of range [0, 20]. Using {valid_centroid_lm_id}.")
                        centroid_lm_id = valid_centroid_lm_id

                    centroid_pixel = landmarks_pixels[centroid_lm_id]
                    hand_data['centroid_x'] = float(centroid_pixel[0])
                    hand_data['centroid_y'] = float(centroid_pixel[1])
                except IndexError:
                    print(f"Warning: IndexError accessing centroid landmark ID {centroid_lm_id}. Using landmark 0.")
                    if len(landmarks_pixels) > 0:
                         centroid_pixel = landmarks_pixels[0]
                         hand_data['centroid_x'] = float(centroid_pixel[0])
                         hand_data['centroid_y'] = float(centroid_pixel[1])

                # Overall Hand Position (Wrist)
                if len(landmarks_norm) > 0 and len(landmarks_pixels) > 0:
                    wrist_norm = landmarks_norm[0]
                    wrist_pixel = landmarks_pixels[0]
                    hand_data['wrist_x'] = float(wrist_pixel[0])
                    hand_data['wrist_y'] = float(wrist_pixel[1])
                    hand_data['wrist_z'] = float(wrist_norm[2]) # Use the raw Z value

                # Overall Hand Orientation
                # Requires at least 3 points to define a plane/orientation
                if len(landmarks_norm) > 9: # Need Wrist, Index MCP, Middle MCP etc.
                    try:
                        # Define axes based on landmarks (normalized coords)
                        p0, p5, p9, p17 = landmarks_norm[0], landmarks_norm[5], landmarks_norm[9], landmarks_norm[17]

                        # Y-axis: Up the arm/hand (Wrist to Middle MCP)
                        vec_0_9 = calculate_vector(p0, p9)
                        norm_vec_0_9 = np.linalg.norm(vec_0_9)
                        if norm_vec_0_9 < 1e-6: raise ValueError("Wrist and Middle MCP too close")
                        hand_y_axis = vec_0_9 / norm_vec_0_9

                        # X-axis: Across the knuckles (approximate: Pinky MCP to Index MCP)
                        vec_17_5 = calculate_vector(p17, p5)
                        # Project onto plane normal to hand_y_axis to make it orthogonal
                        hand_x_axis_initial = vec_17_5 - np.dot(vec_17_5, hand_y_axis) * hand_y_axis
                        norm_hand_x_initial = np.linalg.norm(hand_x_axis_initial)
                        if norm_hand_x_initial < 1e-6: raise ValueError("MCPs are collinear with wrist->MCP vector")
                        hand_x_axis = hand_x_axis_initial / norm_hand_x_initial

                        # Z-axis: Palm normal (using cross product)
                        hand_z_axis = np.cross(hand_x_axis, hand_y_axis)

                        # Calculate Pitch, Yaw, Roll (relative to camera frame)
                        # Assuming Camera looks along +Z, Y is Up, X is Right (OpenCV/MediaPipe view)
                        # If frame is flipped horizontally later, this calculation *might* need adjustment
                        # depending on desired intuitive mapping. Calculation based on *unflipped* coordinates system.

                        # Pitch: Angle of hand's Y-axis with the camera's XY plane (arcsin of Y component)
                        # Level hand -> pitch near 0. Pointing up -> pitch near pi/2. Pointing down -> pitch near -pi/2.
                        hand_data['hand_pitch'] = float(np.arcsin(np.clip(hand_y_axis[1], -1.0, 1.0)))

                        # Yaw: Angle of hand's Y-axis projected onto camera's XZ plane, relative to camera Z-axis (forward)
                        # Use atan2(x, z). Pointing forward -> yaw = 0. Pointing right -> yaw = pi/2. Pointing left -> yaw = -pi/2.
                        proj_y_on_xz = np.array([hand_y_axis[0], 0, hand_y_axis[2]])
                        if np.linalg.norm(proj_y_on_xz) > 1e-6:
                            hand_data['hand_yaw'] = float(np.arctan2(proj_y_on_xz[0], proj_y_on_xz[2]))
                        else: # Hand pointing straight up/down relative to camera
                            hand_data['hand_yaw'] = 0.0

                        # Roll: Angle of hand's X-axis projected onto camera's XY plane, relative to camera X-axis (right)
                        # Use atan2(y, x). Palm flat, fingers right -> roll = 0. Fingers up -> roll = -pi/2. Fingers down -> roll = pi/2.
                        proj_x_on_xy = np.array([hand_x_axis[0], hand_x_axis[1], 0])
                        if np.linalg.norm(proj_x_on_xy) > 1e-6:
                             hand_data['hand_roll'] = float(np.arctan2(proj_x_on_xy[1], proj_x_on_xy[0]))
                        else: # Hand edge-on to camera horizontally
                            hand_data['hand_roll'] = 0.0

                    except (ZeroDivisionError, ValueError, IndexError) as e:
                        # print(f"Warning: Could not calculate hand orientation for hand {i}: {e}")
                         pass # Keep defaults 0.0

                # Finger Curl (Angle Based)
                try:
                    if len(landmarks_norm) < 21: raise IndexError("Not enough landmarks for curl calculation.")

                    # Use Wrist (0) to Middle MCP (9) as reference distance for normalization if needed
                    ref_dist_vec = calculate_vector(landmarks_norm[0], landmarks_norm[9])
                    ref_dist = np.linalg.norm(ref_dist_vec)
                    if ref_dist < 1e-6: ref_dist = 1.0 # Avoid division by zero, use 1 as fallback scale

                    # Thumb: Angle at IP(3) between 3->2(MCP) and 3->4(Tip) - A simplification
                    v32 = calculate_vector(landmarks_norm[3], landmarks_norm[2])
                    v34 = calculate_vector(landmarks_norm[3], landmarks_norm[4])
                    hand_data['thumb_angle_curl'] = float(calculate_angle_between_vectors(v32, v34))

                    # Index: Angle at PIP (6) between vectors 6->5 (MCP) and 6->7 (DIP)
                    v65 = calculate_vector(landmarks_norm[6], landmarks_norm[5])
                    v67 = calculate_vector(landmarks_norm[6], landmarks_norm[7])
                    hand_data['index_angle_curl'] = float(calculate_angle_between_vectors(v65, v67))

                    # Middle: Angle at PIP (10) between vectors 10->9 (MCP) and 10->11 (DIP)
                    v10_9 = calculate_vector(landmarks_norm[10], landmarks_norm[9])
                    v10_11 = calculate_vector(landmarks_norm[10], landmarks_norm[11])
                    hand_data['middle_angle_curl'] = float(calculate_angle_between_vectors(v10_9, v10_11))

                    # Ring: Angle at PIP (14) between vectors 14->13 (MCP) and 14->15 (DIP)
                    v14_13 = calculate_vector(landmarks_norm[14], landmarks_norm[13])
                    v14_15 = calculate_vector(landmarks_norm[14], landmarks_norm[15])
                    hand_data['ring_angle_curl'] = float(calculate_angle_between_vectors(v14_13, v14_15))

                    # Pinky: Angle at PIP (18) between vectors 18->17 (MCP) and 18->19 (DIP)
                    v18_17 = calculate_vector(landmarks_norm[18], landmarks_norm[17])
                    v18_19 = calculate_vector(landmarks_norm[18], landmarks_norm[19])
                    hand_data['pinky_angle_curl'] = float(calculate_angle_between_vectors(v18_17, v18_19))

                except (ZeroDivisionError, ValueError, IndexError) as e:
                    # Keep default values (pi = straight) if calculation fails
                    pass # Silence warning

                # Finger Spread (Angle between MCP vectors from Wrist)
                try:
                    if len(landmarks_norm) < 21: raise IndexError("Not enough landmarks for spread calculation.")
                    p0, p1, p5, p9, p13, p17 = landmarks_norm[0], landmarks_norm[1], landmarks_norm[5], landmarks_norm[9], landmarks_norm[13], landmarks_norm[17]

                    vec_0_1 = calculate_vector(p0, p1)   # Wrist to Thumb CMC
                    vec_0_5 = calculate_vector(p0, p5)   # Wrist to Index MCP
                    vec_0_9 = calculate_vector(p0, p9)   # Wrist to Middle MCP
                    vec_0_13 = calculate_vector(p0, p13) # Wrist to Ring MCP
                    vec_0_17 = calculate_vector(p0, p17) # Wrist to Pinky MCP

                    # Calculate angles only if vectors are valid
                    hand_data['thumb_index_spread'] = float(calculate_angle_between_vectors(vec_0_1, vec_0_5))
                    hand_data['index_middle_spread'] = float(calculate_angle_between_vectors(vec_0_5, vec_0_9))
                    hand_data['middle_ring_spread'] = float(calculate_angle_between_vectors(vec_0_9, vec_0_13))
                    hand_data['ring_pinky_spread'] = float(calculate_angle_between_vectors(vec_0_13, vec_0_17))
                except (ZeroDivisionError, ValueError, IndexError) as e:
                     pass # Silence warning

                # Pinch Distances (Tip to Tip)
                # Use normalized coordinates, normalize by same reference distance as curls (wrist to middle MCP)
                try:
                    if len(landmarks_norm) < 21: raise IndexError("Not enough landmarks for pinch calculation.")
                    if ref_dist < 1e-6: ref_dist = 1.0 # Use pre-calculated ref_dist, fallback if needed

                    p4, p8, p12, p16, p20 = landmarks_norm[4], landmarks_norm[8], landmarks_norm[12], landmarks_norm[16], landmarks_norm[20]

                    dist_4_8 = calculate_distance(p4, p8)
                    dist_4_12 = calculate_distance(p4, p12)
                    dist_4_16 = calculate_distance(p4, p16)
                    dist_4_20 = calculate_distance(p4, p20)

                    hand_data['thumb_index_pinch'] = float(dist_4_8 / ref_dist)
                    hand_data['thumb_middle_pinch'] = float(dist_4_12 / ref_dist)
                    hand_data['thumb_ring_pinch'] = float(dist_4_16 / ref_dist)
                    hand_data['thumb_pinky_pinch'] = float(dist_4_20 / ref_dist)
                except (ZeroDivisionError, ValueError, IndexError) as e:
                     pass # Silence warning

                all_hands_data.append(hand_data)

        # If fewer hands were detected than max_num_hands, fill remaining slots with default data
        while len(all_hands_data) < self.config.get('max_num_hands', 1):
             all_hands_data.append(self._get_default_tracking_data())

        return frame, all_hands_data

    def draw_visuals(self, frame, all_hands_data):
        """Draws landmarks and other visuals onto the frame for all detected hands."""
        for i, hand_data in enumerate(all_hands_data):
            if hand_data['found'] and hand_data['raw_hand_landmarks']:
                # Determine color based on handedness
                color = (0, 255, 0) # Default Green
                handedness_label = hand_data.get('handedness', 'Unknown')
                if handedness_label == 'Left':
                    color = (255, 0, 0) # Blue for Left
                elif handedness_label == 'Right':
                    color = (0, 0, 255) # Red for Right

                landmark_drawing_spec = self.mpDraw.DrawingSpec(color=color, thickness=1, circle_radius=2)
                connection_drawing_spec = self.mpDraw.DrawingSpec(color=color, thickness=2)

                if self.display_config.get('draw_landmarks', True):
                    self.mpDraw.draw_landmarks(
                        frame,
                        hand_data['raw_hand_landmarks'],
                        self.mpHands.HAND_CONNECTIONS if self.display_config.get('draw_connections', True) else None,
                        landmark_drawing_spec,
                        connection_drawing_spec
                    )

                # Highlight the centroid landmark used for X/Y control
                centroid_lm_id = self.config.get('centroid_landmark_id', 0)
                try:
                    # Use calculated pixel coords stored in hand_data
                    cx = int(hand_data['centroid_x'])
                    cy = int(hand_data['centroid_y'])
                    cv2.circle(frame, (cx, cy), 8, color, cv2.FILLED)
                    cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 1) # White outline
                except Exception:
                    pass # Failed to draw centroid

                # Draw handedness label near wrist
                try:
                    wrist_x = int(hand_data['wrist_x'])
                    wrist_y = int(hand_data['wrist_y'])
                    cv2.putText(frame, handedness_label, (wrist_x + 10, wrist_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                except Exception:
                    pass

        return frame

    def close(self):
        """Releases resources."""
        if hasattr(self, 'hands') and hasattr(self.hands, 'close'):
             self.hands.close()
        print("HandTracker closed.")