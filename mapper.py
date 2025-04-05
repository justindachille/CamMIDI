# mapper.py
import numpy as np
import warnings

class MidiMapper:
    def __init__(self, config):
        """Initializes the MIDI mapper."""
        self.config = config
        self.mappings = config.get('mappings', [])
        self.smoothing_factor = config.get('midi', {}).get('smoothing_factor', 0.5)
        # Removed implicit_cc_offset
        self.max_hands = config.get('mediapipe', {}).get('max_num_hands', 1)

        self.camera_width = config.get('camera', {}).get('width', 640)
        self.camera_height = config.get('camera', {}).get('height', 480)

        self._smoothed_values = {} # Store {smooth_key: smoothed_value}

        # Pre-calculate sources explicitly defined for channel 2
        self.explicit_ch2_sources = set()
        if self.max_hands > 1:
             for mapping in self.mappings:
                 if isinstance(mapping, dict) and mapping.get('channel') == 2 and mapping.get('source'):
                     self.explicit_ch2_sources.add(mapping['source'])

        print("MidiMapper initialized.")
        if self.max_hands > 1:
            print(f" Implicit Channel 2 mapping enabled (uses same CC as Channel 1).")
            print(f"  - Explicit Ch2 sources (will override implicit): {self.explicit_ch2_sources or 'None'}")

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
             return self._smoothed_values[smooth_key]
        try: current_value_float = float(current_value)
        except (ValueError, TypeError): return current_value
        alpha = 1.0 - self.smoothing_factor
        previous_smoothed = self._smoothed_values.get(smooth_key, current_value_float)
        if not isinstance(previous_smoothed, (int, float)): previous_smoothed = current_value_float
        smoothed = alpha * current_value_float + (1.0 - alpha) * previous_smoothed
        self._smoothed_values[smooth_key] = smoothed
        return smoothed

    def _get_value_and_smooth(self, hand_data, mapping_params, smooth_key):
        # *** Modification START ***
        source_name = mapping_params.get('source')
        raw_value = hand_data.get(source_name)
        if raw_value is None: return None

        # --- Get Base min/max Input from Config or Defaults ---
        default_min_input, default_max_input = 0.0, 1.0
        # These defaults are only used if 'min/max_input' NOT in mapping_params
        # AND it's not a centroid source (which overrides these later)
        if source_name in ['centroid_x', 'wrist_x']: default_min_input, default_max_input = 0, self.camera_width
        elif source_name in ['centroid_y', 'wrist_y']: default_min_input, default_max_input = 0, self.camera_height
        elif source_name == 'wrist_z': default_min_input, default_max_input = -0.5, 0.5
        elif source_name in ['hand_pitch', 'hand_roll', 'hand_yaw']: default_min_input, default_max_input = -np.pi / 2, np.pi / 2
        elif '_angle_curl' in source_name: default_min_input, default_max_input = np.pi / 2.1, np.pi * 0.98 # Curl defaults might need review based on angle calc changes
        elif 'spread' in source_name: default_min_input, default_max_input = 0.0, np.pi / 4
        elif 'pinch' in source_name: default_min_input, default_max_input = 0.01, 0.3

        min_input = mapping_params.get('min_input', default_min_input)
        max_input = mapping_params.get('max_input', default_max_input)

        # --- Centroid Inset Logic ---
        # Override min/max_input specifically for centroid sources based on camera dimensions
        # Use 12.5% inset from each side (75% usable area in the middle)
        centroid_inset_ratio = 0.125
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

        # Apply smoothing to the scaled (0-127) value
        smoothed_value_float = self._apply_smoothing(float(scaled_value), smooth_key)
        midi_value = int(round(smoothed_value_float))
        # *** Modification END ***

        return max(0, min(127, midi_value)) # Clamp final output just in case

    def calculate_midi(self, all_hands_data, frame_width, frame_height):
        """
        Calculates MIDI messages. Implicitly duplicates channel 1 mappings onto
        channel 2 (using the same CC) if no explicit channel 2 mapping exists
        for that source.
        """
        midi_messages = {} # {(channel, cc): value}

        # Update camera dimensions used for centroid mapping
        self.camera_width = frame_width
        self.camera_height = frame_height

        num_detected_hands = sum(1 for hd in all_hands_data if hd.get('found', False))
        can_do_implicit = self.max_hands > 1 and num_detected_hands > 1

        # --- Iterate through configured mappings ---
        for mapping_index, mapping in enumerate(self.mappings):
            if not isinstance(mapping, dict): continue

            source_name = mapping.get('source')
            mapping_channel = mapping.get('channel')
            mapping_cc = mapping.get('cc')

            if not source_name or mapping_channel is None or mapping_cc is None: continue

            # --- Process for Hand 0 (Channel 1) ---
            if mapping_channel == 1 and len(all_hands_data) > 0 and all_hands_data[0].get('found'):
                hand_data_ch1 = all_hands_data[0]
                smooth_key_ch1 = f"map_{mapping_index}_ch1_{source_name}" # More specific smooth key
                midi_value_ch1 = self._get_value_and_smooth(hand_data_ch1, mapping, smooth_key_ch1)
                if midi_value_ch1 is not None:
                    midi_messages[(1, mapping_cc)] = midi_value_ch1

                # --- Check if IMPLICIT Ch 2 needed for this Ch 1 mapping ---
                if can_do_implicit and source_name not in self.explicit_ch2_sources:
                    # We need data from the second hand (index 1)
                    if len(all_hands_data) > 1 and all_hands_data[1].get('found'):
                        hand_data_ch2 = all_hands_data[1]
                        # Use the SAME CC number, but channel 2
                        implicit_channel = 2
                        implicit_cc = mapping_cc # Use the original CC
                        # Use same mapping params (min/max/invert) as Ch1, but diff smooth key
                        smooth_key_implicit = f"map_{mapping_index}_ch1_implicit_ch2_{source_name}"

                        midi_value_implicit = self._get_value_and_smooth(hand_data_ch2, mapping, smooth_key_implicit)
                        if midi_value_implicit is not None:
                            # Add to messages. Overwriting is allowed if multiple sources implicitly map to the same CC.
                            # Explicit Ch2 mappings handled below will take ultimate precedence if they use this CC.
                             midi_messages[(implicit_channel, implicit_cc)] = midi_value_implicit


            # --- Process EXPLICIT Channel 2 Mappings ---
            # This ensures explicit mappings always override any implicit ones that might target the same CC
            elif mapping_channel == 2 and len(all_hands_data) > 1 and all_hands_data[1].get('found'):
                hand_data_ch2 = all_hands_data[1]
                smooth_key_ch2_expl = f"map_{mapping_index}_ch2_{source_name}" # More specific smooth key
                midi_value_ch2_expl = self._get_value_and_smooth(hand_data_ch2, mapping, smooth_key_ch2_expl)
                if midi_value_ch2_expl is not None:
                    # Explicit always wins for this CC on channel 2
                    midi_messages[(2, mapping_cc)] = midi_value_ch2_expl

        return midi_messages

    # Add a method to update the internal mappings if needed (e.g., after calibration)
    def update_mappings(self, new_mappings):
        """Updates the mappings used by the mapper."""
        self.mappings = new_mappings
        # Recalculate explicit Ch2 sources based on the new mappings
        self.explicit_ch2_sources = set()
        if self.max_hands > 1:
             for mapping in self.mappings:
                 if isinstance(mapping, dict) and mapping.get('channel') == 2 and mapping.get('source'):
                     self.explicit_ch2_sources.add(mapping['source'])
        print("MidiMapper mappings updated.")
        # Clear smoothed values as ranges might have changed drastically
        self._smoothed_values.clear()