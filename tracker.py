import cv2
import mediapipe as mp
import time
import numpy as np

# --- Helper Functions for Calculations ---

def calculate_distance(p1, p2):
    """Calculates 3D Euclidean distance between two points."""
    return np.linalg.norm(p1 - p2)

def calculate_vector(p1, p2):
    """Calculates the vector from p1 to p2."""
    return p2 - p1

def calculate_angle_between_vectors(v1, v2):
    """Calculates the angle in radians between two vectors."""
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    # Clip dot product to avoid numerical errors with arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    return np.arccos(dot_product)

# --- HandTracker Class ---

class HandTracker:
    def __init__(self, config):
        """Initializes the hand tracker with MediaPipe."""
        self.config = config['mediapipe']
        self.display_config = config['display']
        self.camera_config = config['camera']

        self.mpHands = mp.solutions.hands
        try:
            self.hands = self.mpHands.Hands(
                static_image_mode=self.config['static_image_mode'],
                max_num_hands=self.config['max_num_hands'],
                min_detection_confidence=self.config['min_detection_confidence'],
                min_tracking_confidence=self.config['min_tracking_confidence']
            )
        except TypeError:
            print("Warning: MediaPipe Hands parameters might have changed. Using defaults.")
            self.hands = self.mpHands.Hands() # Fallback

        self.mpDraw = mp.solutions.drawing_utils
        self.results = None
        self.frame_height = self.camera_config['height'] # Initial estimates
        self.frame_width = self.camera_config['width']

        print("HandTracker initialized.")

    def process_frame(self, frame):
        """Processes a single video frame to find hand landmarks and calculate metrics."""
        self.frame_height, self.frame_width, _ = frame.shape # Get actual dimensions

        # Optional horizontal flip for intuitive control
        if self.display_config.get('flip_horizontal', True):
            frame = cv2.flip(frame, 1)

        # Convert BGR image to RGB for MediaPipe
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgRGB.flags.writeable = False # Optimization

        # Process the frame
        self.results = self.hands.process(imgRGB)

        imgRGB.flags.writeable = True # Make writable again

        # Initialize tracking data dictionary with defaults
        tracking_data = {
            'found': False,
            'centroid_x': 0.0,
            'centroid_y': 0.0,
            'wrist_x': 0.0,
            'wrist_y': 0.0,
            'wrist_z': 0.0, # Raw Z value, smaller is closer
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
            'raw_results': self.results
        }

        if self.results.multi_hand_landmarks:
            # --- Focus on the first detected hand ---
            handLms = self.results.multi_hand_landmarks[0]
            tracking_data['found'] = True

            # Store landmarks in easily accessible formats
            landmarks_norm = np.array([(lm.x, lm.y, lm.z) for lm in handLms.landmark])
            landmarks_pixels = np.array([(int(lm.x * self.frame_width), int(lm.y * self.frame_height)) for lm in handLms.landmark])

            tracking_data['all_landmarks_pixels'] = landmarks_pixels.tolist()
            tracking_data['all_landmarks_normalized'] = landmarks_norm.tolist()

            # --- Calculate Centroid (as defined in config) ---
            centroid_lm_id = self.config.get('centroid_landmark_id', 0)
            try:
                centroid_pixel = landmarks_pixels[centroid_lm_id]
                tracking_data['centroid_x'] = float(centroid_pixel[0])
                tracking_data['centroid_y'] = float(centroid_pixel[1])
            except IndexError:
                print(f"Warning: Centroid landmark ID {centroid_lm_id} out of range. Using landmark 0.")
                centroid_pixel = landmarks_pixels[0]
                tracking_data['centroid_x'] = float(centroid_pixel[0])
                tracking_data['centroid_y'] = float(centroid_pixel[1])

            # --- 1. Overall Hand Position (Wrist) ---
            wrist_norm = landmarks_norm[0]
            wrist_pixel = landmarks_pixels[0]
            tracking_data['wrist_x'] = float(wrist_pixel[0])
            tracking_data['wrist_y'] = float(wrist_pixel[1])
            tracking_data['wrist_z'] = float(wrist_norm[2]) # Use the raw Z value

            # --- 2. Overall Hand Orientation ---
            # Requires at least 3 points to define a plane/orientation
            if len(landmarks_norm) > 9: # Need Wrist, Index MCP, Middle MCP etc.
                try:
                    # Define axes based on landmarks (normalized coords)
                    # Y-axis: Up the arm/hand (Wrist to Middle MCP)
                    vec_0_9 = calculate_vector(landmarks_norm[0], landmarks_norm[9])
                    hand_y_axis = vec_0_9 / np.linalg.norm(vec_0_9)

                    # X-axis: Across the knuckles (approximate: Pinky MCP to Index MCP)
                    # Flip direction if hand is flipped relative to expectation
                    vec_17_5 = calculate_vector(landmarks_norm[17], landmarks_norm[5])
                    # Project onto plane normal to hand_y_axis to make it orthogonal
                    hand_x_axis_initial = vec_17_5 - np.dot(vec_17_5, hand_y_axis) * hand_y_axis
                    hand_x_axis = hand_x_axis_initial / np.linalg.norm(hand_x_axis_initial)

                    # Z-axis: Palm normal (using cross product)
                    hand_z_axis = np.cross(hand_x_axis, hand_y_axis)

                    # --- Calculate Pitch, Yaw, Roll (relative to camera frame) ---
                    # Assuming Camera looks along +Z, Y is Up, X is Right
                    cam_x = np.array([1, 0, 0])
                    cam_y = np.array([0, 1, 0])
                    cam_z = np.array([0, 0, 1])

                    # Pitch: Angle of hand's Y-axis with the camera's XY plane (arcsin of Y component)
                    tracking_data['hand_pitch'] = float(np.arcsin(np.clip(hand_y_axis[1], -1.0, 1.0)))

                    # Yaw: Angle of hand's Y-axis projected onto camera's XZ plane, relative to camera Z-axis
                    # Use atan2 for full range
                    proj_y_on_xz = np.array([hand_y_axis[0], 0, hand_y_axis[2]])
                    # Prevent zero vector for atan2
                    if np.linalg.norm(proj_y_on_xz) > 1e-6:
                        tracking_data['hand_yaw'] = float(np.arctan2(proj_y_on_xz[0], proj_y_on_xz[2]))
                    else: # Hand pointing straight up/down relative to camera
                        tracking_data['hand_yaw'] = 0.0

                    # Roll: Angle of hand's X-axis projected onto camera's XY plane, relative to camera X-axis
                    proj_x_on_xy = np.array([hand_x_axis[0], hand_x_axis[1], 0])
                     # Prevent zero vector for atan2
                    if np.linalg.norm(proj_x_on_xy) > 1e-6:
                         tracking_data['hand_roll'] = float(np.arctan2(proj_x_on_xy[1], proj_x_on_xy[0]))
                    else: # Hand edge-on to camera horizontally
                        tracking_data['hand_roll'] = 0.0 # Or potentially pi/2 depending on definition

                except (ZeroDivisionError, ValueError, IndexError) as e:
                    print(f"Warning: Could not calculate hand orientation: {e}")
                    # Keep defaults 0.0

            # --- 3. Finger Curl (Angle Based) ---
            try:
                # landmarks_norm contains the (x, y, z) coordinates
                ref_dist = calculate_distance(landmarks_norm[0], landmarks_norm[9])
                if ref_dist < 1e-6: ref_dist = 1.0 # Avoid division by zero
                # Thumb: Angle at MCP (2) between vectors 2->1 (CMC) and 2->3 (IP)
                v32 = calculate_vector(landmarks_norm[3], landmarks_norm[2])
                v34 = calculate_vector(landmarks_norm[3], landmarks_norm[4])
                tracking_data['thumb_angle_curl'] = float(calculate_angle_between_vectors(v32, v34))

                # Index: Angle at PIP (6) between vectors 6->5 (MCP) and 6->7 (DIP)
                v65 = calculate_vector(landmarks_norm[6], landmarks_norm[5])
                v67 = calculate_vector(landmarks_norm[6], landmarks_norm[7])
                tracking_data['index_angle_curl'] = float(calculate_angle_between_vectors(v65, v67))

                # Middle: Angle at PIP (10) between vectors 10->9 (MCP) and 10->11 (DIP)
                v10_9 = calculate_vector(landmarks_norm[10], landmarks_norm[9])
                v10_11 = calculate_vector(landmarks_norm[10], landmarks_norm[11])
                tracking_data['middle_angle_curl'] = float(calculate_angle_between_vectors(v10_9, v10_11))

                # Ring: Angle at PIP (14) between vectors 14->13 (MCP) and 14->15 (DIP)
                v14_13 = calculate_vector(landmarks_norm[14], landmarks_norm[13])
                v14_15 = calculate_vector(landmarks_norm[14], landmarks_norm[15])
                tracking_data['ring_angle_curl'] = float(calculate_angle_between_vectors(v14_13, v14_15))

                # Pinky: Angle at PIP (18) between vectors 18->17 (MCP) and 18->19 (DIP)
                v18_17 = calculate_vector(landmarks_norm[18], landmarks_norm[17])
                v18_19 = calculate_vector(landmarks_norm[18], landmarks_norm[19])
                tracking_data['pinky_angle_curl'] = float(calculate_angle_between_vectors(v18_17, v18_19))

            except (ZeroDivisionError, ValueError, IndexError) as e:
                # Keep default values (pi = straight) if calculation fails
                print(f"Warning: Could not calculate finger angle curl: {e}")

            # --- 4. Finger Spread (Angle between MCP vectors from Wrist) ---
            try:
                vec_0_1 = calculate_vector(landmarks_norm[0], landmarks_norm[1])   # Wrist to Thumb CMC
                vec_0_5 = calculate_vector(landmarks_norm[0], landmarks_norm[5])   # Wrist to Index MCP
                vec_0_9 = calculate_vector(landmarks_norm[0], landmarks_norm[9])   # Wrist to Middle MCP
                vec_0_13 = calculate_vector(landmarks_norm[0], landmarks_norm[13]) # Wrist to Ring MCP
                vec_0_17 = calculate_vector(landmarks_norm[0], landmarks_norm[17]) # Wrist to Pinky MCP

                tracking_data['thumb_index_spread'] = float(calculate_angle_between_vectors(vec_0_1, vec_0_5))
                tracking_data['index_middle_spread'] = float(calculate_angle_between_vectors(vec_0_5, vec_0_9))
                tracking_data['middle_ring_spread'] = float(calculate_angle_between_vectors(vec_0_9, vec_0_13))
                tracking_data['ring_pinky_spread'] = float(calculate_angle_between_vectors(vec_0_13, vec_0_17))
            except (ZeroDivisionError, ValueError, IndexError) as e:
                 print(f"Warning: Could not calculate finger spread angles: {e}")


            # --- 5. Pinch Distances (Tip to Tip) ---
            # Use normalized coordinates, normalize by same reference distance as curls
            try:
                dist_4_8 = calculate_distance(landmarks_norm[4], landmarks_norm[8])
                dist_4_12 = calculate_distance(landmarks_norm[4], landmarks_norm[12])
                dist_4_16 = calculate_distance(landmarks_norm[4], landmarks_norm[16])
                dist_4_20 = calculate_distance(landmarks_norm[4], landmarks_norm[20])

                tracking_data['thumb_index_pinch'] = float(dist_4_8 / ref_dist)
                tracking_data['thumb_middle_pinch'] = float(dist_4_12 / ref_dist)
                tracking_data['thumb_ring_pinch'] = float(dist_4_16 / ref_dist)
                tracking_data['thumb_pinky_pinch'] = float(dist_4_20 / ref_dist)
            except (ZeroDivisionError, ValueError, IndexError) as e:
                 print(f"Warning: Could not calculate pinch distances: {e}")


        # Return the processed (potentially flipped) frame and the comprehensive tracking data
        return frame, tracking_data

    def draw_visuals(self, frame, tracking_data):
        """Draws landmarks and other visuals onto the frame."""
        if tracking_data['found'] and self.results.multi_hand_landmarks:
            handLms = self.results.multi_hand_landmarks[0] # Draw only first hand for now
            if self.display_config.get('draw_landmarks', True):
                # Draw basic landmarks and connections
                self.mpDraw.draw_landmarks(frame, handLms,
                                           self.mpHands.HAND_CONNECTIONS if self.display_config.get('draw_connections', True) else None)

            # Highlight the centroid landmark used for X/Y control
            centroid_lm_id = self.config.get('centroid_landmark_id', 0)
            try:
                cx = int(tracking_data['centroid_x'])
                cy = int(tracking_data['centroid_y'])
                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), cv2.FILLED) # Red circle for centroid
            except Exception:
                pass # Failed to draw centroid

            # Example: Draw line for pinch distance (Thumb tip to Index tip)
            # if tracking_data.get('all_landmarks_pixels') and len(tracking_data['all_landmarks_pixels']) > 8:
            #    try:
            #        p_thumb = tuple(tracking_data['all_landmarks_pixels'][4])
            #        p_index = tuple(tracking_data['all_landmarks_pixels'][8])
            #        cv2.line(frame, p_thumb, p_index, (255, 255, 0), 2) # Cyan line
            #    except Exception:
            #        pass

        return frame

    def close(self):
        """Releases resources."""
        # self.hands.close() # Consider adding if performance issues arise or using static mode
        print("HandTracker closed.")