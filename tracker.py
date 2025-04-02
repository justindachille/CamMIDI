import cv2
import mediapipe as mp
import time
import numpy as np

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
        """Processes a single video frame to find hand landmarks."""
        self.frame_height, self.frame_width, _ = frame.shape # Get actual dimensions

        # Optional horizontal flip for intuitive control
        if self.display_config.get('flip_horizontal', True):
             frame = cv2.flip(frame, 1)

        # Convert BGR image to RGB for MediaPipe
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgRGB.flags.writeable = False # Optimization

        # Process the frame
        self.results = self.hands.process(imgRGB)

        imgRGB.flags.writeable = True # Make writable again (though we draw on the BGR frame)

        tracking_data = {
            'found': False,
            'centroid_x': 0.0,
            'centroid_y': 0.0,
            'all_landmarks': [], # List of (x, y) pixel coordinates for the first detected hand
            # --- Future expansion: Add more extracted data here ---
            # 'pinch_distance': None,
            # 'gesture': None,
            'raw_results': self.results # Keep raw results if needed later
        }

        if self.results.multi_hand_landmarks:
            # --- Focus on the first detected hand for now ---
            handLms = self.results.multi_hand_landmarks[0] # TODO: Add logic for multiple hands if needed
            tracking_data['found'] = True
            all_landmarks_pixels = []

            # Calculate centroid based on configured landmark ID
            centroid_lm_id = self.config.get('centroid_landmark_id', 0)
            try:
                centroid_lm = handLms.landmark[centroid_lm_id]
                cx = centroid_lm.x * self.frame_width
                cy = centroid_lm.y * self.frame_height
                tracking_data['centroid_x'] = cx
                tracking_data['centroid_y'] = cy
            except IndexError:
                print(f"Warning: Centroid landmark ID {centroid_lm_id} out of range. Using landmark 0.")
                centroid_lm = handLms.landmark[0]
                cx = centroid_lm.x * self.frame_width
                cy = centroid_lm.y * self.frame_height
                tracking_data['centroid_x'] = cx
                tracking_data['centroid_y'] = cy


            # Store all landmark positions (pixels) for potential future use
            for id, lm in enumerate(handLms.landmark):
                 lx, ly = int(lm.x * self.frame_width), int(lm.y * self.frame_height)
                 all_landmarks_pixels.append((lx, ly))
            tracking_data['all_landmarks'] = all_landmarks_pixels

            # --- Add logic here to calculate other metrics like pinch distance ---
            # Example (needs corresponding mapping config):
            # if len(all_landmarks_pixels) > 8: # Ensure thumb and index tips exist
            #    thumb_tip = np.array(all_landmarks_pixels[4])
            #    index_tip = np.array(all_landmarks_pixels[8])
            #    dist = np.linalg.norm(thumb_tip - index_tip)
            #    # Normalize distance (example, needs tuning based on hand size/camera distance)
            #    normalized_dist = dist / self.frame_width
            #    tracking_data['pinch_distance'] = normalized_dist


        return frame, tracking_data # Return the (potentially flipped) frame and tracking data

    def draw_visuals(self, frame, tracking_data):
        """Draws landmarks and other visuals onto the frame."""
        if tracking_data['found'] and self.results.multi_hand_landmarks:
             handLms = self.results.multi_hand_landmarks[0] # Draw only first hand for now
             if self.display_config.get('draw_landmarks', True):
                self.mpDraw.draw_landmarks(frame, handLms,
                                           self.mpHands.HAND_CONNECTIONS if self.display_config.get('draw_connections', True) else None)

             # Highlight the centroid landmark
             centroid_lm_id = self.config.get('centroid_landmark_id', 0)
             try:
                 cx = int(tracking_data['centroid_x'])
                 cy = int(tracking_data['centroid_y'])
                 cv2.circle(frame, (cx, cy), 10, (0, 0, 255), cv2.FILLED) # Red circle for centroid
             except IndexError:
                  pass # Landmark wasn't found or calculation failed


        return frame

    def close(self):
        """Releases resources."""
        # self.hands.close() # Not strictly needed for this implementation but good practice if using static_image_mode often
        print("HandTracker closed.")