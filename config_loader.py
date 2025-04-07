# config_loader.py
import yaml
import os
import numpy as np
import copy

DEFAULT_CONFIG = {
    'camera': {
        'index': 0,
        'width': 800,
        'height': 460
    },
    'mediapipe': {
        'static_image_mode': False,
        'max_num_hands': 2,
        'min_detection_confidence': 0.6,
        'min_tracking_confidence': 0.6,
        'centroid_landmark_id': 0 # 0 = Wrist, 9 = Middle Finger MCP
    },
    'midi': {
        'port_name': "IAC Driver Bus 1",
        'smoothing_factor': 0.6,
        'force_channel1_hand': 'Right',
    },
    'mappings': [
        # --- Basic Position ---
        {
            'source': 'centroid_x', # Or 'wrist_x'
            'channel': 1, 'cc': 1, # Example: Filter Cutoff
            'invert': False,
             # min/max_input now handled dynamically in mapper.py for centroid
        },
        {
            'source': 'centroid_y', # Or 'wrist_y'
            'channel': 1, 'cc': 2, # Example: Filter Resonance
            'invert': True # Screen Y down -> MIDI up
             # min/max_input now handled dynamically in mapper.py for centroid
        },
        # --- Depth (Z-axis) ---
        # {
        #     'source': 'wrist_z', # Raw Z from MediaPipe (smaller=closer)
        #     'channel': 1, 'cc': 1, # Example: Mod Wheel
        #     'invert': True,       # Closer -> Higher MIDI value
        #     # IMPORTANT: Calibrate min/max_input for your setup!
        #     'min_input': -0.6,    # Example: Closest expected Z
        #     'max_input': 0.2      # Example: Farthest expected Z
        # },
        # --- Orientation (Radians) ---
        {
            'source': 'hand_pitch', # Radians, angle with XY plane
            'channel': 1, 'cc': 3,
            'invert': False,
            'min_input': -np.pi / 2, # Approx -90 deg (pointing down)
            'max_input': np.pi / 2   # Approx +90 deg (pointing up)
        },
        {
            'source': 'hand_yaw', # Radians, rotation around vertical axis
            'channel': 1, 'cc': 4,
            'invert': False,
            'min_input': -np.pi / 2, # Approx -90 deg (pointing left)
            'max_input': np.pi / 2   # Approx +90 deg (pointing right)
        },
        {
            'source': 'hand_roll', # Radians, rotation around pointing axis
            'channel': 1, 'cc': 5, # Example: Pan
            'invert': False,
            'min_input': -np.pi / 2, # Approx -90 deg (palm down, fingers up)
            'max_input': np.pi / 2   # Approx +90 deg (palm up, fingers down)
        },
        # --- Finger Curls (Angle Based, Radians) ---
        # Straight finger = Pi (~3.14 rad), 90 degree bend = Pi/2 (~1.57 rad)
        # Smaller angle = more curl. Requires invert: True to map curl -> high MIDI.
        # *** CALIBRATE min/max_input (in radians) FOR YOUR HAND! ***
        {
            'source': 'index_angle_curl',
            'channel': 1, 'cc': 6,
            'invert': True,        # Curled (small angle) -> High MIDI
            'min_input': 1.5,      # Example: Approx 90deg bend (for MIDI 127)
            'max_input': 3.1       # Example: Approx Pi (straight for MIDI 0)
        },
        { 'source': 'middle_angle_curl', 'channel': 1, 'cc': 7, 'invert': True, 'min_input': 1.5, 'max_input': 3.1 },
        { 'source': 'ring_angle_curl',   'channel': 1, 'cc': 8, 'invert': True, 'min_input': 1.5, 'max_input': 3.1 },
        # **** Pinky curl sensitivity adjusted ****
        { 'source': 'pinky_angle_curl',  'channel': 1, 'cc': 9, 'invert': True, 'min_input': 1.7, 'max_input': 3.1 }, # Less sensitive (was 1.5)
        { 'source': 'thumb_angle_curl',  'channel': 1, 'cc': 10, 'invert': True, 'min_input': 1.8, 'max_input': 3.0 },
        # --- Finger Spreads (Radians, angle between MCPs at wrist) ---
        # *** CALIBRATE using buttons ***
        {
            'source': 'index_middle_spread',
            'channel': 1, 'cc': 11,
            'invert': False,       # Wider spread -> High MIDI
            'min_input': 0.0,      # Default: Fingers touching
            'max_input': np.pi / 5 # Default: Approx 36 deg spread
        },
        { 'source': 'middle_ring_spread', 'channel': 1, 'cc': 12, 'invert': False, 'min_input': 0.0, 'max_input': np.pi / 6 },
        { 'source': 'ring_pinky_spread', 'channel': 1, 'cc': 13, 'invert': False, 'min_input': 0.0, 'max_input': np.pi / 6 },
        # --- Pinch Distances (Normalized Distance Tip-to-Tip) ---
        # *** CALIBRATE using buttons ***
        {
           'source': 'thumb_index_pinch', # Normalized distance Tip 4 to Tip 8
           'channel': 1, 'cc': 14, # Example: Mod Wheel (replaces Z example)
           'invert': True,       # Small distance (pinch) -> High MIDI
           'min_input': 0.02,    # Example: Closest pinch distance
           'max_input': 0.25     # Example: Farthest pinch distance
        },
        { 'source': 'thumb_middle_pinch', 'channel': 1, 'cc': 15, 'invert': True, 'min_input': 0.03, 'max_input': 0.3 },
        { 'source': 'thumb_ring_pinch', 'channel': 1, 'cc': 16, 'invert': True, 'min_input': 0.05, 'max_input': 0.35 },
        { 'source': 'thumb_pinky_pinch', 'channel': 1, 'cc': 17, 'invert': True, 'min_input': 0.07, 'max_input': 0.4 },
        {
            'source': 'head_yaw', # Face Yaw (-pi/2 to pi/2 approx)
            'channel': 3, 'cc': 30,
            'invert': False,
            'min_input': -0.8, 'max_input': 0.8 # Calibrate based on observation
        },
        {
            'source': 'head_pitch', # Face Pitch (-pi/2 to pi/2 approx)
            'channel': 3, 'cc': 31,
            'invert': False, # Nodding up -> higher value? Or invert if needed
            'min_input': -0.6, 'max_input': 0.6 # Calibrate
        },
        {
            'source': 'head_roll', # Face Roll (-pi/2 to pi/2 approx)
            'channel': 3, 'cc': 32,
            'invert': False,
            'min_input': -0.5, 'max_input': 0.5 # Calibrate
        },
        {
            'source': 'mar', # Mouth Aspect Ratio (0=closed, ~1=open)
            'channel': 3, 'cc': 33,
            'invert': False, # Open mouth -> higher value
            'min_input': 0.0, 'max_input': 0.8 # Calibrate based on observation
        },
        {
            'source': 'jaw_openness', # Normalized Jaw Openness
            'channel': 3, 'cc': 34,
            'invert': False,
            'min_input': 0.0, 'max_input': 0.2 # Calibrate
        },
        # Average Eye Aspect Ratio (Blinking)
        # Might need custom logic in mapper or pre-processing if desired
        # Example mapping individual eyes:
        {
            'source': 'left_ear', # Left Eye Aspect Ratio (~0.35=open, <0.2=closed)
            'channel': 3, 'cc': 35,
            'invert': True, # Blink (small EAR) -> higher value?
            'min_input': 0.15, 'max_input': 0.40 # Calibrate
        },
         {
            'source': 'right_ear', # Right Eye Aspect Ratio
            'channel': 3, 'cc': 36,
            'invert': True,
            'min_input': 0.15, 'max_input': 0.40 # Calibrate
        },
        # Eyebrow Height examples
        {
             'source': 'left_eyebrow_height', # Normalized height
             'channel': 3, 'cc': 37,
             'invert': False, # Raise eyebrow -> higher value
             'min_input': 0.05, 'max_input': 0.25 # Calibrate
        },
        {
             'source': 'right_eyebrow_height', # Normalized height
             'channel': 3, 'cc': 38,
             'invert': False,
             'min_input': 0.05, 'max_input': 0.25 # Calibrate
        },
        # --- EXAMPLE MAPPINGS FOR HAND 2 (Channel 2) ---
        # If you want to control different parameters with the second hand,
        # duplicate the desired mappings above and change 'channel' to 2.
        # The CC numbers below would continue incrementing if uncommented.
        # {
        #     'source': 'centroid_x', # Second hand X
        #     'channel': 2, 'cc': 18, # Example: Delay Time
        #     'invert': False,
        # },
        # {
        #     'source': 'centroid_y', # Second hand Y
        #     'channel': 2, 'cc': 19, # Example: Delay Feedback
        #     'invert': True
        # },
        # {
        #     'source': 'index_angle_curl', # Second hand Index Curl
        #     'channel': 2, 'cc': 20, # Example: LFO Rate
        #     'invert': True,
        #     'min_input': 1.5,
        #     'max_input': 3.1
        # },
    ],
    'display': {
        'show_window': True,
        'draw_landmarks': True,
        'draw_face_tesselation': True,
        'draw_connections': True,
        'show_fps': True,
        'flip_horizontal': True
    }
}

def load_config(config_path='config.yaml', force_defaults=False):
    """Loads configuration from YAML file, providing defaults."""
    config = copy.deepcopy(DEFAULT_CONFIG)

    if not force_defaults and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f: user_config = yaml.safe_load(f)
            if user_config:
                print(f"Info: Loading configuration from {config_path}")
                # Update existing keys recursively
                def update_dict(target, source):
                    for key, value in source.items():
                        if key == 'mappings' and isinstance(value, list):
                            target[key] = value # Replace mappings list entirely
                        elif isinstance(value, dict) and key in target and isinstance(target[key], dict):
                            update_dict(target[key], value)
                        else:
                            target[key] = value

                update_dict(config, user_config)

        except Exception as e:
            print(f"Warning: Could not load or parse {config_path}. Using defaults. Error: {e}")
            config = copy.deepcopy(DEFAULT_CONFIG) # Reset to defaults on error
    elif not force_defaults and not os.path.exists(config_path):
        print("Info: config.yaml not found. Using defaults. Creating file.")
        save_config(DEFAULT_CONFIG, config_path) # Save defaults immediately
        # config is already default config here
    elif force_defaults:
        print("Info: Using forced default configuration.")
        # config is already default config here


    # --- Post-Load Validation & Default Filling ---
    # Ensure top-level keys exist
    for key, default_value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = copy.deepcopy(default_value)
        # Ensure sub-dictionaries have required keys (except mappings)
        elif isinstance(default_value, dict) and key != 'mappings':
            current_dict = config.get(key, {})
            if not isinstance(current_dict, dict):
                config[key] = copy.deepcopy(default_value) # Overwrite if type changed
            else:
                for sub_key, default_sub_value in default_value.items():
                    if sub_key not in current_dict:
                        config[key][sub_key] = copy.deepcopy(default_sub_value)

    # Validate mappings list separately
    if 'mappings' not in config or not isinstance(config['mappings'], list):
        config['mappings'] = copy.deepcopy(DEFAULT_CONFIG['mappings'])

    # Convert numpy types just in case they sneak in (e.g. from old saves)
    for mapping in config.get('mappings', []):
        if isinstance(mapping, dict):
            for map_key, map_value in mapping.items():
                if isinstance(map_value, np.floating): mapping[map_key] = float(map_value)
                elif isinstance(map_value, np.integer): mapping[map_key] = int(map_value)

    # Ensure backward compatibility: if only 'mediapipe' confidence exists, apply to 'face_mesh'
    mp_conf = config.get('mediapipe', {})
    fm_conf = config.get('face_mesh', {})
    if 'min_detection_confidence' not in fm_conf and 'min_detection_confidence' in mp_conf:
        fm_conf['min_detection_confidence'] = mp_conf['min_detection_confidence']
    if 'min_tracking_confidence' not in fm_conf and 'min_tracking_confidence' in mp_conf:
        fm_conf['min_tracking_confidence'] = mp_conf['min_tracking_confidence']
    config['face_mesh'] = fm_conf # Ensure face_mesh dict exists

    print("Configuration loaded.")
    # print(yaml.dump(config, sort_keys=False)) # Debug: Print final config
    return config


# --- save_config function ---
def save_config(config, config_path='config.yaml'):
     """Saves the current configuration to YAML file."""
     try:
         # Add comments before saving
         comment = """# CamMIDI Configuration
#
# Calibration: Use UI buttons for Hand Pinch/Spread min/max ranges.
# Manually edit face metric ranges (head_*, ear, mar, etc.) based on observed values.
# Press 'Save Config' in the UI to save changes to this file.
#
# See DEFAULT_CONFIG in config_loader.py for all possible parameters.
# Hand metrics ('centroid_*', 'hand_*', '*_curl', '*_spread', '*_pinch')
# Face metrics ('head_*', '*_ear', 'mar', 'jaw_openness', '*_eyebrow_height')
#
# MIDI:
# - port_name: Your virtual MIDI port.
# - smoothing_factor: 0.0 (none) to 0.99 (max). Default: 0.6
# - force_channel1_hand: Assigns Hand Ch1 based on handedness if two hands present.
#
# Mappings:
# - Define 'source' (metric name), 'channel', 'cc', 'invert' (optional),
#   'min_input'/'max_input' (REQUIRED for non-centroid sources).
# - Hand mappings can use implicit Channel 2 logic (see docs/mapper.py).
# - Face mappings typically use a dedicated channel (e.g., 3) or reuse 1/2.
#
"""
         # Convert numpy types to standard Python types for YAML saving
         config_to_save = copy.deepcopy(config)
         # Convert numpy types in mappings
         if 'mappings' in config_to_save and isinstance(config_to_save['mappings'], list):
             for mapping in config_to_save['mappings']:
                 if isinstance(mapping, dict):
                    for key, value in mapping.items():
                        if isinstance(value, np.integer): mapping[key] = int(value)
                        elif isinstance(value, np.floating):
                            if np.isinf(value) or np.isnan(value):
                                print(f"Warning: Invalid value ({value}) found for '{key}' in mapping '{mapping.get('source', 'Unknown')}'. Replacing with 0.")
                                mapping[key] = 0.0
                            else:
                                mapping[key] = float(value)
                        elif isinstance(value, float) and (np.isinf(value) or np.isnan(value)):
                            print(f"Warning: Invalid float value ({value}) found for '{key}' in mapping '{mapping.get('source', 'Unknown')}'. Replacing with 0.")
                            mapping[key] = 0.0
         with open(config_path, 'w') as f:
             f.write(comment)
             yaml.dump(config_to_save, f, default_flow_style=False, sort_keys=False)
         print(f"Configuration saved to {config_path}")
     except Exception as e:
         print(f"Error saving configuration to {config_path}: {e}")


if __name__ == '__main__':
    # Example usage: Load and print config
    print("--- Loading Default Config (Simulated) ---")
    cfg_default = load_config(force_defaults=True)
    print(yaml.dump(cfg_default, sort_keys=False))

    print("\n--- Loading Config (from file if exists, else default) ---")
    dummy_path = 'test_config_load.yaml'
    if os.path.exists(dummy_path): os.remove(dummy_path)
    cfg_normal = load_config(config_path=dummy_path)
    # print(yaml.dump(cfg_normal, sort_keys=False)) # Print loaded/created config

    # Example of accessing new values
    print(f"\nFace Max Num Faces: {cfg_normal.get('face_mesh', {}).get('max_num_faces')}")
    print(f"Hand Max Num Hands: {cfg_normal.get('mediapipe', {}).get('max_num_hands')}")

    # Test saving after modification
    if 'face_mesh' in cfg_normal:
        cfg_normal['face_mesh']['max_num_faces'] = 2 # Example change
        cfg_normal['mappings'].append({'source': 'face_test', 'channel': 3, 'cc': 99, 'min_input':0, 'max_input':1})
        save_config(cfg_normal, dummy_path)
        print(f"\n--- Loading Config ({dummy_path} after modification) ---")
        cfg_reloaded = load_config(config_path=dummy_path)
        print(f"Face Max Num Faces (Reloaded): {cfg_reloaded.get('face_mesh', {}).get('max_num_faces')}")
        # Clean up test file
        if os.path.exists(dummy_path): os.remove(dummy_path)