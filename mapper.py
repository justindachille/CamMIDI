import numpy as np

class MidiMapper:
    def __init__(self, config):
        """Initializes the MIDI mapper."""
        self.mappings = config.get('mappings', [])
        self.smoothing_factor = config['midi'].get('smoothing_factor', 0.5)
        self.camera_width = config['camera']['width'] # Expected width
        self.camera_height = config['camera']['height'] # Expected height
        self._smoothed_values = {} # Store {mapping_index: smoothed_value}
        self._last_raw_values = {} # Store {mapping_index: raw_value}
        print("MidiMapper initialized.")

    def _scale_value(self, value, min_in, max_in, min_out=0, max_out=127, invert=False):
        """Scales a value from one range to another, clamps, and handles inversion."""
        if max_in == min_in: return min_out if not invert else max_out # Avoid division by zero

        # Clamp input value
        value = max(min_in, min(max_in, value))

        # Perform scaling
        scaled = (value - min_in) / (max_in - min_in) * (max_out - min_out) + min_out

        # Handle inversion
        if invert:
            scaled = (max_out + min_out) - scaled

        # Clamp output and return integer
        return int(round(max(min_out, min(max_out, scaled))))

    def _apply_smoothing(self, current_value, mapping_index):
        """Applies Exponential Moving Average (EMA) smoothing."""
        if self.smoothing_factor <= 0: # No smoothing
            return current_value
        if self.smoothing_factor >= 1: # Max smoothing (value never changes after first)
            if mapping_index not in self._smoothed_values:
                 self._smoothed_values[mapping_index] = current_value
            return self._smoothed_values[mapping_index]

        alpha = 1.0 - self.smoothing_factor # More intuitive: higher factor = more smoothing
        previous_smoothed = self._smoothed_values.get(mapping_index, current_value) # Start with current if no history
        smoothed = alpha * current_value + (1.0 - alpha) * previous_smoothed
        self._smoothed_values[mapping_index] = smoothed
        return smoothed # Return float for potential precision needed before final int conversion


    def calculate_midi(self, tracking_data, frame_width, frame_height):
        """Calculates MIDI messages based on tracking data and configured mappings."""
        midi_messages = {} # {(channel, cc): value}

        if not tracking_data.get('found', False):
            # Return empty dict if no hand found - consider resetting values if needed
            return {}

        # Update actual frame dimensions used for scaling (relevant mainly for centroid/wrist X/Y)
        self.camera_width = frame_width
        self.camera_height = frame_height

        for i, mapping in enumerate(self.mappings):
            source_name = mapping['source']
            raw_value = tracking_data.get(source_name) # Get value using source name

            if raw_value is None:
                # Silently skip if source doesn't exist in tracking_data for this frame
                # (e.g., orientation calculation failed but other data is present)
                # Consider adding a warning here if it happens frequently for configured sources
                # print(f"Warning: Source '{source_name}' not found in tracking_data for frame. Skipping mapping.")
                continue

            # --- Define default input ranges based on source type ---
            # These are FALLBACKS if min/max_input are not in the config.
            # USER SHOULD DEFINE RANGES IN CONFIG for best results.
            default_min_input = 0.0
            default_max_input = 1.0 # Default for normalized values

            if source_name in ['centroid_x', 'wrist_x']:
                default_min_input = 0
                default_max_input = self.camera_width
            elif source_name in ['centroid_y', 'wrist_y']:
                default_min_input = 0
                default_max_input = self.camera_height
            elif source_name == 'wrist_z':
                # Z is tricky, depends on distance. Needs user calibration.
                default_min_input = -0.5 # Guess: Closer
                default_max_input = 0.5  # Guess: Farther
            elif source_name in ['hand_pitch', 'hand_yaw', 'hand_roll']:
                # Angles in radians
                default_min_input = -np.pi / 2 # Guess: -90 degrees
                default_max_input = np.pi / 2  # Guess: +90 degrees (adjust as needed)
            elif '_angle_curl' in source_name:
                # Angle in radians. Straight = Pi (~3.14), 90 deg = Pi/2 (~1.57)
                # We want straight (Pi) -> MIDI 0, and 90deg (Pi/2) -> MIDI 127
                # So, invert = True is needed.
                default_min_input = np.pi / 2.1 # Slightly less than 90deg (most curled state for max MIDI)
                default_max_input = np.pi * 0.98 # Slightly less than 180deg (straightest state for min MIDI)
            elif 'spread' in source_name:
                # Angle in radians
                default_min_input = 0.0    # Guess: Fingers together
                default_max_input = np.pi / 4 # Guess: Spread out (~45 deg)
            elif 'pinch' in source_name:
                # Normalized distance
                default_min_input = 0.01  # Guess: Pinching close
                default_max_input = 0.3   # Guess: Fingers far apart
                # Often inverted: small distance (pinch) -> high MIDI

            # Use user-defined min/max if available, otherwise use defaults
            min_input = mapping.get('min_input', default_min_input)
            max_input = mapping.get('max_input', default_max_input)

            # --- Apply Scaling ---
            scaled_value = self._scale_value(
                raw_value,
                min_in=min_input,
                max_in=max_input,
                invert=mapping.get('invert', False)
            )

            # --- Apply Smoothing (operates on the 0-127 scaled value) ---
            # Apply smoothing *after* scaling to keep smoothing consistent across ranges
            # Note: smoothing uses float intermediate values
            smoothed_value_float = self._apply_smoothing(float(scaled_value), i)

            # --- Prepare MIDI Message ---
            channel = mapping['channel']
            cc = mapping['cc']
            midi_value = int(round(smoothed_value_float)) # Final conversion to int
            midi_value = max(0, min(127, midi_value)) # Ensure clamp after smoothing

            message_key = (channel, cc)
            midi_messages[message_key] = midi_value

        return midi_messages