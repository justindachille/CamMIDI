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
        if max_in == min_in: return min_out # Avoid division by zero

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
        return smoothed


    def calculate_midi(self, tracking_data, frame_width, frame_height):
        """Calculates MIDI messages based on tracking data and configured mappings."""
        midi_messages = {} # {(channel, cc): value}

        if not tracking_data.get('found', False):
            # Optional: Send default values or do nothing when hand is lost
            # Example: Reset CCs associated with position when hand is lost?
            # Requires tracking which CCs are position-based.
            return {} # Return empty dict if no hand found

        # Update actual frame dimensions used for scaling
        self.camera_width = frame_width
        self.camera_height = frame_height

        for i, mapping in enumerate(self.mappings):
            source_name = mapping['source']
            raw_value = None

            # --- Get the raw value based on the source type ---
            if source_name == 'centroid_x':
                raw_value = tracking_data.get('centroid_x')
                min_input = mapping.get('min_input', 0) # Default to 0
                max_input = mapping.get('max_input', self.camera_width) # Default to frame width
            elif source_name == 'centroid_y':
                raw_value = tracking_data.get('centroid_y')
                min_input = mapping.get('min_input', 0) # Default to 0
                max_input = mapping.get('max_input', self.camera_height) # Default to frame height
            # --- Add more 'elif source_name == ...' blocks for future features ---
            # elif source_name == 'pinch_distance':
            #     # This value would need to be calculated in tracker.py
            #     raw_value = tracking_data.get('pinch_distance')
            #     # These ranges typically need calibration (often normalized 0-1ish)
            #     min_input = mapping.get('min_input', 0.0)
            #     max_input = mapping.get('max_input', 1.0)
            else:
                print(f"Warning: Unknown mapping source '{source_name}' in config. Skipping.")
                continue

            if raw_value is None:
                # print(f"Warning: Source '{source_name}' not found in tracking_data. Skipping mapping.")
                continue

            # --- Apply Scaling ---
            scaled_value = self._scale_value(
                raw_value,
                min_in=min_input,
                max_in=max_input,
                invert=mapping.get('invert', False)
            )

            # --- Apply Smoothing ---
            smoothed_value = self._apply_smoothing(scaled_value, i) # Use mapping index as unique ID

            # --- Prepare MIDI Message ---
            channel = mapping['channel']
            cc = mapping['cc']
            midi_value = int(round(smoothed_value))

            message_key = (channel, cc)
            midi_messages[message_key] = midi_value

        return midi_messages