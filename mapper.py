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
        # Read and store the channel forcing option, ensure capitalization
        self.force_channel1_hand = config.get('midi', {}).get('force_channel1_hand', "None").capitalize()
        if self.force_channel1_hand not in ["None", "Left", "Right"]:
             print(f"Warning: Invalid 'force_channel1_hand' value '{self.force_channel1_hand}'. Defaulting to 'None'.")
             self.force_channel1_hand = "None"


        self.camera_width = config.get('camera', {}).get('width', 640)
        self.camera_height = config.get('camera', {}).get('height', 480)

        self._smoothed_values = {} # Store {smooth_key: smoothed_value}

        # Pre-calculate sources explicitly defined for channel 2
        self.explicit_ch2_sources = set()
        if self.max_hands > 1:
             self._update_explicit_ch2_sources() # Calculate based on initial mappings

        print("MidiMapper initialized.")
        print(f" Force Channel 1 Hand: {self.force_channel1_hand}")
        if self.max_hands > 1:
            print(f" Implicit Channel 2 mapping enabled (uses same CC as Channel 1).")
            print(f"  - Explicit Ch2 sources (will override implicit): {self.explicit_ch2_sources or 'None'}")

    def _update_explicit_ch2_sources(self):
        """Helper to recalculate the set of sources explicitly mapped to channel 2."""
        self.explicit_ch2_sources.clear()
        for mapping in self.mappings:
            if isinstance(mapping, dict) and mapping.get('channel') == 2 and mapping.get('source'):
                self.explicit_ch2_sources.add(mapping['source'])

    def _scale_value(self, value, min_in, max_in, min_out=0, max_out=127, invert=False):
        # (Function remains the same)
        if value is None or min_in is None or max_in is None: return (min_out + max_out) // 2
        try: value, min_in, max_in = float(value), float(min_in), float(max_in)
        except (ValueError, TypeError): return (min_out + max_out) // 2
        if max_in == min_in: return min_out if not invert else max_out
        value = max(min_in, min(max_in, value)) # Clamp input value
        scaled = (value - min_in) / (max_in - min_in) * (max_out - min_out) + min_out
        if invert: scaled = (max_out + min_out) - scaled
        return int(round(max(min_out, min(max_out, scaled))))

    def _apply_smoothing(self, current_value, smooth_key):
        # (Function remains the same)
        if self.smoothing_factor <= 0: return current_value
        if self.smoothing_factor >= 1:
             if smooth_key not in self._smoothed_values: self._smoothed_values[smooth_key] = current_value
             return self._smoothed_values[smooth_key] # No smoothing, just store last value
        try: current_value_float = float(current_value)
        except (ValueError, TypeError): return current_value # Return original if not numeric
        alpha = 1.0 - self.smoothing_factor # Smoothing coefficient for new value
        previous_smoothed = self._smoothed_values.get(smooth_key, current_value_float) # Start with current if no history
        # Ensure previous value is numeric
        if not isinstance(previous_smoothed, (int, float)): previous_smoothed = current_value_float
        smoothed = alpha * current_value_float + (1.0 - alpha) * previous_smoothed
        self._smoothed_values[smooth_key] = smoothed
        return smoothed

    def _get_value_and_smooth(self, hand_data, mapping_params, smooth_key):
        # (Function remains largely the same, including centroid inset logic)
        source_name = mapping_params.get('source')
        raw_value = hand_data.get(source_name)
        if raw_value is None:
            # print(f"Warning: Source '{source_name}' not found in hand_data for key {smooth_key}")
            return None # Return None if source doesn't exist in data

        # --- Get Base min/max Input from Config or Defaults ---
        default_min_input, default_max_input = 0.0, 1.0
        # These defaults are only used if 'min/max_input' NOT in mapping_params
        # AND it's not a centroid source (which overrides these later)
        if source_name in ['centroid_x', 'wrist_x']: default_min_input, default_max_input = 0, self.camera_width
        elif source_name in ['centroid_y', 'wrist_y']: default_min_input, default_max_input = 0, self.camera_height
        elif source_name == 'wrist_z': default_min_input, default_max_input = -0.5, 0.5 # Example range
        elif source_name in ['hand_pitch', 'hand_roll', 'hand_yaw']: default_min_input, default_max_input = -np.pi / 2, np.pi / 2
        elif '_angle_curl' in source_name: default_min_input, default_max_input = 1.0, np.pi # Radians (Approx 57 to 180 deg)
        elif 'spread' in source_name: default_min_input, default_max_input = 0.0, np.pi / 4 # Radians (Approx 0 to 45 deg)
        elif 'pinch' in source_name: default_min_input, default_max_input = 0.01, 0.3 # Normalized distance

        min_input = mapping_params.get('min_input', default_min_input)
        max_input = mapping_params.get('max_input', default_max_input)

        # --- Centroid Inset Logic ---
        centroid_inset_ratio = 0.125 # 12.5% inset from each side
        if source_name == 'centroid_x':
            inset_x = self.camera_width * centroid_inset_ratio
            min_input = inset_x
            max_input = self.camera_width - inset_x
        elif source_name == 'centroid_y':
            inset_y = self.camera_height * centroid_inset_ratio
            min_input = inset_y
            max_input = self.camera_height - inset_y
        # --- End Centroid Inset Logic ---

        # Now scale using the potentially adjusted min/max_input
        scaled_value = self._scale_value(
            raw_value,
            min_input,
            max_input,
            invert=mapping_params.get('invert', False)
        )
        if scaled_value is None: return None # Propagate None if scaling failed

        # Apply smoothing to the scaled (0-127) value
        smoothed_value_float = self._apply_smoothing(float(scaled_value), smooth_key)
        midi_value = int(round(smoothed_value_float))

        return max(0, min(127, midi_value)) # Clamp final output just in case

    def calculate_midi(self, all_hands_data, frame_width, frame_height):
        """
        Calculates MIDI messages based on hand data and mappings.
        Assigns channels based on detection order or the 'force_channel1_hand' setting.
        Returns a dictionary: {(channel, cc): {'value': value, 'hand_index': index}}
        """
        # Store results here: {(channel, cc): {'value': value, 'hand_index': original_index}}
        midi_messages = {}

        # Update camera dimensions used for centroid mapping
        self.camera_width = frame_width
        self.camera_height = frame_height

        # Filter out hands that were not found
        detected_hands = [(idx, hd) for idx, hd in enumerate(all_hands_data) if hd.get('found', False)]
        num_detected_hands = len(detected_hands)

        # --- Determine which hand index corresponds to which channel ---
        hand_for_ch1_index = -1
        hand_for_ch2_index = -1
        hand_data_ch1 = None
        hand_data_ch2 = None

        if num_detected_hands == 1:
            # Single hand always gets Channel 1
            hand_for_ch1_index = detected_hands[0][0] # Get original index
            hand_data_ch1 = detected_hands[0][1]      # Get hand data
        elif num_detected_hands >= 2: # Handle 2 or more (though max_hands likely limits this)
            # Prioritize the first two detected hands for assignment
            hand1_idx, hand1_data = detected_hands[0]
            hand2_idx, hand2_data = detected_hands[1]
            hand1_label = hand1_data.get('handedness', 'Unknown')
            hand2_label = hand2_data.get('handedness', 'Unknown')

            # Assign based on force_channel1_hand setting
            if self.force_channel1_hand == "Left":
                if hand1_label == 'Left':
                    hand_for_ch1_index, hand_data_ch1 = hand1_idx, hand1_data
                    hand_for_ch2_index, hand_data_ch2 = hand2_idx, hand2_data
                elif hand2_label == 'Left':
                    hand_for_ch1_index, hand_data_ch1 = hand2_idx, hand2_data
                    hand_for_ch2_index, hand_data_ch2 = hand1_idx, hand1_data
                else: # No left hand found, use detection order
                    hand_for_ch1_index, hand_data_ch1 = hand1_idx, hand1_data
                    hand_for_ch2_index, hand_data_ch2 = hand2_idx, hand2_data

            elif self.force_channel1_hand == "Right":
                if hand1_label == 'Right':
                    hand_for_ch1_index, hand_data_ch1 = hand1_idx, hand1_data
                    hand_for_ch2_index, hand_data_ch2 = hand2_idx, hand2_data
                elif hand2_label == 'Right':
                    hand_for_ch1_index, hand_data_ch1 = hand2_idx, hand2_data
                    hand_for_ch2_index, hand_data_ch2 = hand1_idx, hand1_data
                else: # No right hand found, use detection order
                    hand_for_ch1_index, hand_data_ch1 = hand1_idx, hand1_data
                    hand_for_ch2_index, hand_data_ch2 = hand2_idx, hand2_data

            else: # "None" or invalid setting - use detection order
                hand_for_ch1_index, hand_data_ch1 = hand1_idx, hand1_data
                hand_for_ch2_index, hand_data_ch2 = hand2_idx, hand2_data
        else:
            # 0 hands detected, do nothing
            pass

        # --- Iterate through configured mappings ---
        can_do_implicit = self.max_hands > 1 and hand_data_ch1 is not None and hand_data_ch2 is not None

        for mapping_index, mapping in enumerate(self.mappings):
            if not isinstance(mapping, dict): continue

            source_name = mapping.get('source')
            mapping_channel = mapping.get('channel')
            mapping_cc = mapping.get('cc')

            if not source_name or mapping_channel is None or mapping_cc is None: continue

            # --- Process for Channel 1 (using hand_data_ch1 if available) ---
            if mapping_channel == 1 and hand_data_ch1 is not None:
                # Use the original index of the hand assigned to Ch1 for the smooth key
                smooth_key_ch1 = f"hand{hand_for_ch1_index}_map{mapping_index}_ch1_{source_name}"
                midi_value_ch1 = self._get_value_and_smooth(hand_data_ch1, mapping, smooth_key_ch1)
                if midi_value_ch1 is not None:
                    # Store value AND the original index of the hand that generated it
                    midi_messages[(1, mapping_cc)] = {'value': midi_value_ch1, 'hand_index': hand_for_ch1_index}

                # --- Check if IMPLICIT Ch 2 needed ---
                # Implicit mapping uses data from hand_data_ch2, but the *mapping* comes from the Ch1 config
                if can_do_implicit and source_name not in self.explicit_ch2_sources:
                    implicit_channel = 2
                    implicit_cc = mapping_cc # Use the original CC from Ch1 mapping
                    # Use the original index of the hand assigned to Ch2 for the smooth key
                    smooth_key_implicit = f"hand{hand_for_ch2_index}_map{mapping_index}_ch1_implicit_ch2_{source_name}"

                    midi_value_implicit = self._get_value_and_smooth(hand_data_ch2, mapping, smooth_key_implicit)
                    if midi_value_implicit is not None:
                         # Add/overwrite implicit mapping. Explicit Ch2 handled below may overwrite this.
                         midi_messages[(implicit_channel, implicit_cc)] = {'value': midi_value_implicit, 'hand_index': hand_for_ch2_index}

            # --- Process EXPLICIT Channel 2 Mappings (using hand_data_ch2 if available) ---
            elif mapping_channel == 2 and hand_data_ch2 is not None:
                 # Use the original index of the hand assigned to Ch2 for the smooth key
                smooth_key_ch2_expl = f"hand{hand_for_ch2_index}_map{mapping_index}_ch2_{source_name}"
                midi_value_ch2_expl = self._get_value_and_smooth(hand_data_ch2, mapping, smooth_key_ch2_expl)
                if midi_value_ch2_expl is not None:
                    # Explicit mapping always overwrites any implicit mapping for the same channel/cc
                    midi_messages[(2, mapping_cc)] = {'value': midi_value_ch2_expl, 'hand_index': hand_for_ch2_index}

        return midi_messages

    def update_mappings(self, new_mappings):
        """Updates the mappings used by the mapper."""
        self.mappings = new_mappings
        # Recalculate explicit Ch2 sources based on the new mappings
        self._update_explicit_ch2_sources()
        print("MidiMapper mappings updated.")
        print(f"  - New Explicit Ch2 sources: {self.explicit_ch2_sources or 'None'}")
        # Clear smoothed values as ranges might have changed drastically
        self._smoothed_values.clear()
        print("  - Smoothed values cleared.")