# mapper.py
import numpy as np
import warnings

class MidiMapper:
    def __init__(self, config):
        """Initializes the MIDI mapper."""
        self.config = config
        self.mappings = config.get('mappings', [])
        self.smoothing_factor = config.get('midi', {}).get('smoothing_factor', 0.5)
        self.max_hands = config.get('mediapipe', {}).get('max_num_hands', 1)
        self.force_channel1_hand = config.get('midi', {}).get('force_channel1_hand', "None").capitalize()
        if self.force_channel1_hand not in ["None", "Left", "Right"]:
             print(f"Warning: Invalid 'force_channel1_hand' value '{self.force_channel1_hand}'. Defaulting to 'None'.")
             self.force_channel1_hand = "None"

        # Camera dimensions are updated dynamically in calculate_midi now
        self.camera_width = config.get('camera', {}).get('width', 640)
        self.camera_height = config.get('camera', {}).get('height', 480)

        self._smoothed_values = {} # Store {smooth_key: smoothed_value}

        # --- Define potential face sources BEFORE calling update_explicit ---
        self.potential_face_sources = {
            'head_pitch', 'head_yaw', 'head_roll',
            'left_ear', 'right_ear', 'mar', 'jaw_openness',
            'left_eyebrow_height', 'right_eyebrow_height'
            # Add others if defined in face_tracker
        }
        # --- Moved definition above ---

        # Pre-calculate sources explicitly defined for channel 2 (Hand specific)
        self.explicit_ch2_sources = set()
        if self.max_hands > 1:
             self._update_explicit_ch2_sources() # Calculate based on initial mappings


        # Also consider any source name starting with 'face_' or 'head_'
        # This check happens within calculate_midi

        print("MidiMapper initialized.")
        print(f" Force Channel 1 Hand: {self.force_channel1_hand}")
        if self.max_hands > 1:
            print(f" Implicit Channel 2 Hand mapping enabled (uses same CC as Channel 1).")
            print(f"  - Explicit Hand Ch2 sources (override implicit): {self.explicit_ch2_sources or 'None'}")
        # Commenting out noisy print:
        # print(f" Known face metric source names: {self.potential_face_sources}")

    # Rest of the MidiMapper class remains the same...
    # _update_explicit_ch2_sources, _scale_value, _apply_smoothing,
    # _get_value_and_smooth, calculate_midi, update_mappings

    def _update_explicit_ch2_sources(self):
        """Helper to recalculate the set of sources explicitly mapped to channel 2."""
        self.explicit_ch2_sources.clear()
        for mapping in self.mappings:
            if not isinstance(mapping, dict): continue # Ensure mapping is a dict

            # Only consider hand sources for implicit/explicit Ch2 logic for now
            source = mapping.get('source', '')
            if not source: continue # Skip if no source defined

            # Check if it's likely a face source
            is_potential_face_source = (source in self.potential_face_sources or
                                        source.startswith('face_') or
                                        source.startswith('head_'))

            # If it's NOT a face source AND is explicitly Ch2, add it
            if not is_potential_face_source:
                if mapping.get('channel') == 2:
                    self.explicit_ch2_sources.add(source)

    # --- scale_value function ---
    def _scale_value(self, value, min_in, max_in, min_out=0, max_out=127, invert=False):
        if value is None: return (min_out + max_out) // 2 # Handle None input early
        # Ensure min/max are valid numbers before conversion
        if min_in is None or max_in is None: return (min_out + max_out) // 2

        try:
            value_f = float(value)
            min_in_f = float(min_in)
            max_in_f = float(max_in)
        except (ValueError, TypeError):
            return (min_out + max_out) // 2 # Return midpoint if conversion fails

        if abs(max_in_f - min_in_f) < 1e-9: # Use epsilon for float comparison
            return min_out if not invert else max_out

        # Clamp input value
        value_f = max(min_in_f, min(max_in_f, value_f))

        # Perform scaling
        scaled = (value_f - min_in_f) / (max_in_f - min_in_f) * (max_out - min_out) + min_out

        if invert:
            scaled = (max_out + min_out) - scaled

        # Clamp output and round to integer
        return int(round(max(min_out, min(max_out, scaled))))

    # --- apply_smoothing function ---
    def _apply_smoothing(self, current_value, smooth_key):
        if self.smoothing_factor <= 0: return current_value # No smoothing
        if self.smoothing_factor >= 1: # Store last value, effectively no smoothing but keeps state
             if smooth_key not in self._smoothed_values: self._smoothed_values[smooth_key] = current_value
             return self._smoothed_values[smooth_key]

        try:
            current_value_float = float(current_value)
        except (ValueError, TypeError):
            return current_value # Return original if not numeric

        alpha = 1.0 - self.smoothing_factor # Smoothing coefficient for new value (closer to 0 = more smoothing)
        previous_smoothed = self._smoothed_values.get(smooth_key)

        # Initialize previous value if not present or invalid
        if previous_smoothed is None or not isinstance(previous_smoothed, (int, float, np.number)):
            previous_smoothed = current_value_float

        try:
            previous_smoothed_float = float(previous_smoothed) # Ensure previous is float for calculation
            smoothed = alpha * current_value_float + (1.0 - alpha) * previous_smoothed_float
        except (ValueError, TypeError):
            smoothed = current_value_float # Fallback if previous value is problematic

        self._smoothed_values[smooth_key] = smoothed
        return smoothed # Return the float smoothed value, round in the calling function

    # --- get_value_and_smooth function ---
    def _get_value_and_smooth(self, source_data_dict, mapping_params, smooth_key):
        """Gets raw value from source dict, scales, smooths, and returns final MIDI int."""
        source_name = mapping_params.get('source')
        raw_value = source_data_dict.get(source_name)

        if raw_value is None:
            return None # Source not found in data

        # --- Get min/max Input ---
        # Define defaults carefully
        default_min_input, default_max_input = None, None # Start with None
        if mapping_params.get('min_input') is not None and mapping_params.get('max_input') is not None:
            min_input = mapping_params['min_input']
            max_input = mapping_params['max_input']
        else:
            # Only apply dynamic/default ranges if not specified in mapping
            # Hand Defaults
            if source_name in ['centroid_x', 'wrist_x']: min_input, max_input = 0, self.camera_width
            elif source_name in ['centroid_y', 'wrist_y']: min_input, max_input = 0, self.camera_height
            elif source_name == 'wrist_z': min_input, max_input = -0.5, 0.5
            elif source_name in ['hand_pitch', 'hand_roll', 'hand_yaw']: min_input, max_input = -np.pi / 2, np.pi / 2
            elif '_angle_curl' in source_name: min_input, max_input = 1.0, np.pi
            elif 'spread' in source_name: min_input, max_input = 0.0, np.pi / 4
            elif 'pinch' in source_name: min_input, max_input = 0.01, 0.3
            # Face Defaults
            elif source_name in ['head_pitch', 'head_roll', 'head_yaw']: min_input, max_input = -0.7, 0.7 # Slightly wider default
            elif source_name in ['left_ear', 'right_ear']: min_input, max_input = 0.15, 0.4
            elif source_name == 'mar': min_input, max_input = 0.0, 0.8
            elif source_name == 'jaw_openness': min_input, max_input = 0.0, 0.2
            elif 'eyebrow' in source_name: min_input, max_input = 0.05, 0.25
            else:
                min_input, max_input = 0.0, 1.0 # Generic fallback if name not recognized

            # Apply overrides from mapping_params if they exist (even if only one exists)
            min_input = mapping_params.get('min_input', min_input)
            max_input = mapping_params.get('max_input', max_input)


        # --- Centroid Inset Logic (applies AFTER defaults/mapping overrides) ---
        centroid_inset_ratio = 0.125
        if source_name == 'centroid_x':
            inset_x = self.camera_width * centroid_inset_ratio
            min_input = inset_x
            max_input = self.camera_width - inset_x
        elif source_name == 'centroid_y':
            inset_y = self.camera_height * centroid_inset_ratio
            min_input = inset_y
            max_input = self.camera_height - inset_y

        # Check if min/max are valid before scaling
        if min_input is None or max_input is None:
             print(f"Warning: Missing min/max_input for source '{source_name}'. Cannot scale.")
             return None


        # Scale the raw value to 0-127 range (returns int)
        scaled_value_int = self._scale_value(
            raw_value,
            min_input,
            max_input,
            invert=mapping_params.get('invert', False)
        )
        if scaled_value_int is None: return None # Scaling failed

        # Apply smoothing to the scaled (0-127) value (returns float)
        smoothed_value_float = self._apply_smoothing(float(scaled_value_int), smooth_key)

        # Round the smoothed float value to the nearest integer and clamp
        midi_value = int(round(smoothed_value_float))
        return max(0, min(127, midi_value))

    # --- calculate_midi function ---
    # (No changes needed here based on the bug report)
    def calculate_midi(self, all_hands_data, face_data, frame_width, frame_height):
        """
        Calculates MIDI messages based on hand and face data.
        Assigns hand channels based on detection order or 'force_channel1_hand'.
        Face data is processed based on mapping channel.
        Returns a dictionary: {(channel, cc): {'value': value, 'source_type': 'hand'/'face', 'source_index': index}}
        """
        midi_messages = {}
        self.camera_width = frame_width
        self.camera_height = frame_height

        detected_hands = [(idx, hd) for idx, hd in enumerate(all_hands_data) if hd.get('found', False)]
        num_detected_hands = len(detected_hands)

        hand_for_ch1_index, hand_data_ch1 = -1, None
        hand_for_ch2_index, hand_data_ch2 = -1, None

        # Hand assignment logic (remains same)
        if num_detected_hands == 1:
            hand_for_ch1_index, hand_data_ch1 = detected_hands[0]
        elif num_detected_hands >= 2:
            hand1_idx, hand1_data = detected_hands[0]
            hand2_idx, hand2_data = detected_hands[1]
            hand1_label = hand1_data.get('handedness', 'Unknown')
            hand2_label = hand2_data.get('handedness', 'Unknown')
            if self.force_channel1_hand == "Left":
                if hand1_label == 'Left': hand_for_ch1_index, hand_data_ch1, hand_for_ch2_index, hand_data_ch2 = hand1_idx, hand1_data, hand2_idx, hand2_data
                elif hand2_label == 'Left': hand_for_ch1_index, hand_data_ch1, hand_for_ch2_index, hand_data_ch2 = hand2_idx, hand2_data, hand1_idx, hand1_data
                else: hand_for_ch1_index, hand_data_ch1, hand_for_ch2_index, hand_data_ch2 = hand1_idx, hand1_data, hand2_idx, hand2_data
            elif self.force_channel1_hand == "Right":
                if hand1_label == 'Right': hand_for_ch1_index, hand_data_ch1, hand_for_ch2_index, hand_data_ch2 = hand1_idx, hand1_data, hand2_idx, hand2_data
                elif hand2_label == 'Right': hand_for_ch1_index, hand_data_ch1, hand_for_ch2_index, hand_data_ch2 = hand2_idx, hand2_data, hand1_idx, hand1_data
                else: hand_for_ch1_index, hand_data_ch1, hand_for_ch2_index, hand_data_ch2 = hand1_idx, hand1_data, hand2_idx, hand2_data
            else: # "None" or invalid
                hand_for_ch1_index, hand_data_ch1, hand_for_ch2_index, hand_data_ch2 = hand1_idx, hand1_data, hand2_idx, hand2_data

        can_do_implicit = self.max_hands > 1 and hand_data_ch1 is not None and hand_data_ch2 is not None
        face_available = face_data is not None and face_data.get('found', False)

        for mapping_index, mapping in enumerate(self.mappings):
            if not isinstance(mapping, dict): continue
            source_name = mapping.get('source')
            mapping_channel = mapping.get('channel')
            mapping_cc = mapping.get('cc')
            if not source_name or mapping_channel is None or mapping_cc is None: continue

            is_face_source = (source_name in self.potential_face_sources or
                              source_name.startswith('face_') or
                              source_name.startswith('head_'))

            # --- Process Face Sources ---
            if is_face_source:
                if face_available:
                    smooth_key_face = f"face0_map{mapping_index}_ch{mapping_channel}_{source_name}"
                    midi_value_face = self._get_value_and_smooth(face_data, mapping, smooth_key_face)
                    if midi_value_face is not None:
                        midi_messages[(mapping_channel, mapping_cc)] = {'value': midi_value_face, 'source_type': 'face', 'source_index': 0}

            # --- Process Hand Sources ---
            else:
                # Channel 1 Hand
                if mapping_channel == 1 and hand_data_ch1 is not None:
                    smooth_key_ch1 = f"hand{hand_for_ch1_index}_map{mapping_index}_ch1_{source_name}"
                    midi_value_ch1 = self._get_value_and_smooth(hand_data_ch1, mapping, smooth_key_ch1)
                    if midi_value_ch1 is not None:
                        midi_messages[(1, mapping_cc)] = {'value': midi_value_ch1, 'source_type': 'hand', 'source_index': hand_for_ch1_index}

                    # Implicit Ch 2 Hand
                    if can_do_implicit and source_name not in self.explicit_ch2_sources:
                        implicit_channel, implicit_cc = 2, mapping_cc
                        smooth_key_implicit = f"hand{hand_for_ch2_index}_map{mapping_index}_ch1_implicit_ch2_{source_name}"
                        midi_value_implicit = self._get_value_and_smooth(hand_data_ch2, mapping, smooth_key_implicit)
                        if midi_value_implicit is not None:
                            # Add/overwrite implicit mapping (explicit below takes precedence)
                            midi_messages[(implicit_channel, implicit_cc)] = {'value': midi_value_implicit, 'source_type': 'hand', 'source_index': hand_for_ch2_index}

                # Explicit Channel 2 Hand
                elif mapping_channel == 2 and hand_data_ch2 is not None:
                    smooth_key_ch2_expl = f"hand{hand_for_ch2_index}_map{mapping_index}_ch2_{source_name}"
                    midi_value_ch2_expl = self._get_value_and_smooth(hand_data_ch2, mapping, smooth_key_ch2_expl)
                    if midi_value_ch2_expl is not None:
                        midi_messages[(2, mapping_cc)] = {'value': midi_value_ch2_expl, 'source_type': 'hand', 'source_index': hand_for_ch2_index}

        return midi_messages


    def update_mappings(self, new_mappings):
        """Updates the mappings used by the mapper."""
        self.mappings = new_mappings
        self._update_explicit_ch2_sources() # Recalculate using the corrected method
        print("MidiMapper mappings updated.")
        print(f"  - New Explicit Hand Ch2 sources: {self.explicit_ch2_sources or 'None'}")
        self._smoothed_values.clear()
        print("  - Smoothed values cleared.")