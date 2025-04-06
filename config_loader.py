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
                for key, value in user_config.items():
                    if key == 'mappings' and isinstance(value, list): config[key] = value
                    elif isinstance(value, dict) and key in config and isinstance(config[key], dict): config[key].update(value)
                    elif key in config: config[key] = value
                    else: config[key] = value
        except Exception as e:
            print(f"Warning: Could not load or parse {config_path}. Using defaults. Error: {e}")
            config = copy.deepcopy(DEFAULT_CONFIG)
    elif not force_defaults and not os.path.exists(config_path):
        print("Info: config.yaml not found. Using defaults. Creating file.")
        save_config(DEFAULT_CONFIG, config_path)
        config = copy.deepcopy(DEFAULT_CONFIG)
    elif force_defaults:
        print("Info: Using forced default configuration.")

    # --- Post-Load Validation ---
    for key, default_value in DEFAULT_CONFIG.items():
        if key not in config: config[key] = copy.deepcopy(default_value)
        elif isinstance(default_value, dict) and key != 'mappings':
            current_dict = config.get(key, {})
            if not isinstance(current_dict, dict): config[key] = copy.deepcopy(default_value)
            else:
                for sub_key, default_sub_value in default_value.items():
                    if sub_key not in current_dict: config[key][sub_key] = copy.deepcopy(default_sub_value)

    if 'mappings' not in config or not isinstance(config['mappings'], list):
        config['mappings'] = copy.deepcopy(DEFAULT_CONFIG['mappings'])

    # Convert numpy types
    for mapping in config.get('mappings', []):
        if isinstance(mapping, dict):
            for map_key, map_value in mapping.items():
                if isinstance(map_value, np.floating): mapping[map_key] = float(map_value)
                elif isinstance(map_value, np.integer): mapping[map_key] = int(map_value)

    print("Configuration loaded.")
    return config

def save_config(config, config_path='config.yaml'):
     """Saves the current configuration to YAML file."""
     try:
         # Add comments before saving
         comment = """# CamMIDI Configuration
#
# Calibration: Use the buttons in the UI ('Squeeze', 'Spread', 'Pinch Min/Max')
# to set the 'min_input' and 'max_input' ranges for spread and pinch gestures.
# Press 'Save Config' in the UI to make these changes permanent in this file.
#
# Centroid X/Y: The usable input range is automatically inset from the screen edges
# by 12.5% on each side (configurable inset in mapper.py if needed). The 'min_input'
# and 'max_input' fields for 'centroid_x'/'centroid_y' in this file are ignored.
#
# For sources using distances ('curl', 'pinch', 'wrist_z'):
# - These are sensitive to hand size and camera distance.
# - Values are often normalized relative to hand size (e.g., wrist-to-middle-MCP distance).
# - **You MUST calibrate 'min_input' and 'max_input' for your setup!**
#   (Use UI buttons for pinch/spread, manually edit for curl/z based on console/display values).
# - 'invert: True' is common for curls/pinches (small distance = high MIDI value).
#
# For sources using angles ('pitch', 'yaw', 'roll', 'spread'):
# - Values are typically in radians. Pi = 3.14159..., Pi/2 = 1.5708...
# - Ranges like -pi/2 to +pi/2 (-90 to +90 deg) or 0 to pi/4 (0 to 45 deg) are common starting points.
# - Pitch/Yaw/Roll default range uses min_input=-pi/2, max_input=+pi/2.
# - Calibrate 'min_input' / 'max_input' based on your comfortable range of motion (use UI for spread).
#
# MIDI:
# - port_name: Your virtual MIDI port.
# - smoothing_factor: 0.0 (none) to 0.99 (max). Default: 0.6
# - force_channel1_hand: Assigns Channel 1 based on handedness if two hands are present.
#   Options: "None" (default, first detected hand is Ch1),
#            "Left" (detected Left hand is Ch1, Right/Unknown is Ch2),
#            "Right" (detected Right hand is Ch1, Left/Unknown is Ch2).
#   A single detected hand always uses Channel 1 regardless of this setting.
#
# Mappings:
# - Processed PER HAND based on channel assignment logic (see force_channel1_hand).
# - channel: 1 -> Hand assigned to Channel 1
# - channel: 2 -> Hand assigned to Channel 2
# - Implicit Channel 2 Mapping:
#   If 'max_num_hands' > 1, any mapping for 'channel: 1'
#   will be IMPLICITLY duplicated for 'channel: 2' using the second hand's data
#   with the SAME CC number, UNLESS a mapping explicitly defines 'channel: 2'
#   for the SAME 'source'. Explicit mappings always override implicit ones
#   for a given source/channel/cc combination.
# - Add 'comment' field for notes (optional).
#
"""
         # Convert numpy types to standard Python types for YAML saving
         config_to_save = copy.deepcopy(config)
         if 'mappings' in config_to_save and isinstance(config_to_save['mappings'], list):
             for mapping in config_to_save['mappings']:
                 if isinstance(mapping, dict):
                     for key, value in mapping.items():
                         if isinstance(value, np.integer):
                             mapping[key] = int(value)
                         elif isinstance(value, np.floating):
                             mapping[key] = float(value)

         with open(config_path, 'w') as f:
             f.write(comment)
             yaml.dump(config_to_save, f, default_flow_style=False, sort_keys=False)
         print(f"Configuration saved to {config_path}")
     except Exception as e:
         print(f"Error saving configuration to {config_path}: {e}")


if __name__ == '__main__':
    # Example usage: Load and print config
    # Test default loading
    print("--- Loading Default Config (Simulated) ---")
    cfg_default = load_config(force_defaults=True)
    print(yaml.dump(cfg_default, sort_keys=False))

    # Test loading from file (if exists)
    print("\n--- Loading Config (from file if exists, else default) ---")
    # Ensure a dummy file doesn't exist for this test or rename it
    dummy_path = 'test_config_load.yaml'
    if os.path.exists(dummy_path): os.remove(dummy_path)
    cfg_normal = load_config(config_path=dummy_path) # Will create file with defaults
    # print(yaml.dump(cfg_normal, sort_keys=False))

    # Example of accessing a value
    print(f"\nSmoothing Factor (Normal Load): {cfg_normal.get('midi', {}).get('smoothing_factor')}")
    print(f"Max Num Hands (Normal Load): {cfg_normal.get('mediapipe', {}).get('max_num_hands')}")
    print(f"Force Channel 1 Hand (Normal Load): {cfg_normal.get('midi', {}).get('force_channel1_hand')}")

    # Test saving after modification
    if 'midi' in cfg_normal:
        cfg_normal['midi']['force_channel1_hand'] = "Left"
        cfg_normal['mappings'].append({'source': 'test', 'channel': 1, 'cc': 99})
        save_config(cfg_normal, dummy_path)
        print(f"\n--- Loading Config ({dummy_path} after modification) ---")
        cfg_reloaded = load_config(config_path=dummy_path)
        print(f"Force Channel 1 Hand (Reloaded): {cfg_reloaded.get('midi', {}).get('force_channel1_hand')}")
        # print(yaml.dump(cfg_reloaded, sort_keys=False))
        if os.path.exists(dummy_path): os.remove(dummy_path) # Clean up test file