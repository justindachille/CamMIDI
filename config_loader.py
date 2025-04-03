import yaml
import os
import numpy as np # Needed for default range examples

DEFAULT_CONFIG = {
    'camera': {
        'index': 0,
        'width': 640,
        'height': 480
    },
    'mediapipe': {
        'static_image_mode': False,
        'max_num_hands': 1,
        'min_detection_confidence': 0.6,
        'min_tracking_confidence': 0.6,
        'centroid_landmark_id': 0 # 0 = Wrist, 9 = Middle Finger MCP
    },
    'midi': {
        'port_name': "IAC Driver Bus 1", # CHANGE THIS
        'smoothing_factor': 0.6
    },
    'mappings': [
        # --- Basic Position ---
        {
            'source': 'centroid_x', # Or 'wrist_x'
            'channel': 1, 'cc': 74, # Example: Filter Cutoff
            'invert': False,
             # min/max_input default to camera width/height if not specified
        },
        {
            'source': 'centroid_y', # Or 'wrist_y'
            'channel': 1, 'cc': 71, # Example: Filter Resonance
            'invert': True # Screen Y down -> MIDI up
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
            'channel': 1, 'cc': 2,
            'invert': False,
            'min_input': 0, # Approx 0 deg
            'max_input': np.pi   # Approx +180 deg
        },
        {
            'source': 'hand_yaw', # Radians, rotation around vertical axis
            'channel': 1, 'cc': 3,
            'invert': False,
            'min_input': -np.pi / 2, # Approx -90 deg
            'max_input': np.pi / 2   # Approx +90 deg
        },
        {
            'source': 'hand_roll', # Radians, rotation around pointing axis
            'channel': 1, 'cc': 10, # Example: Pan
            'invert': False,
            'min_input': -np.pi / 2, # Approx -90 deg
            'max_input': np.pi / 2   # Approx +90 deg
        },
        # --- Finger Curls (Angle Based, Radians) ---
        # Straight finger = Pi (~3.14 rad), 90 degree bend = Pi/2 (~1.57 rad)
        # Smaller angle = more curl. Requires invert: True to map curl -> high MIDI.
        # *** CALIBRATE min/max_input (in radians) FOR YOUR HAND! ***
        {
            'source': 'index_angle_curl',
            'channel': 1, 'cc': 4,
            'invert': True,        # Curled (small angle) -> High MIDI
            # Calibrate: Find YOUR min (e.g. bent 90deg) & max (straight) angle in radians
            'min_input': 1.5,      # Example: Approx Pi/2 (most bent for MIDI 127)
            'max_input': 3.1       # Example: Approx Pi (straight for MIDI 0)
        },
        { 'source': 'middle_angle_curl', 'channel': 1, 'cc': 5, 'invert': True, 'min_input': 1.5, 'max_input': 3.1 },
        { 'source': 'ring_angle_curl',   'channel': 1, 'cc': 6, 'invert': True, 'min_input': 1.5, 'max_input': 3.1 },
        { 'source': 'pinky_angle_curl',  'channel': 1, 'cc': 9, 'invert': True, 'min_input': 1.5, 'max_input': 3.1 },
        { 'source': 'thumb_angle_curl',  'channel': 1, 'cc': 11, 'invert': True, 'min_input': 1.8, 'max_input': 3.0 }, 
        # --- Finger Spreads (Radians, angle between MCPs at wrist) ---
        # {
        #     'source': 'index_middle_spread',
        #     'channel': 1, 'cc': 12,
        #     'invert': False,       # Wider spread -> High MIDI
        #     'min_input': 0.0,      # Fingers touching
        #     'max_input': np.pi / 5 # Approx 36 deg spread
        # },
        # { 'source': 'middle_ring_spread', 'channel': 1, 'cc': 13, 'invert': False, 'min_input': 0.0, 'max_input': np.pi / 6 },
        # { 'source': 'ring_pinky_spread', 'channel': 1, 'cc': 14, 'invert': False, 'min_input': 0.0, 'max_input': np.pi / 6 },
        # --- Pinch Distances (Normalized Distance Tip-to-Tip) ---
        {
           'source': 'thumb_index_pinch', # Normalized distance Tip 4 to Tip 8
           'channel': 1, 'cc': 1, # Example: Mod Wheel (replaces Z example)
           'invert': True,       # Small distance (pinch) -> High MIDI
           # IMPORTANT: Calibrate min/max_input!
           'min_input': 0.02,    # Example: Closest pinch distance
           'max_input': 0.25     # Example: Farthest pinch distance
        },
        # { 'source': 'thumb_middle_pinch', 'channel': 1, 'cc': 16, 'invert': True, 'min_input': 0.03, 'max_input': 0.3 },
        # { 'source': 'thumb_ring_pinch', 'channel': 1, 'cc': 17, 'invert': True, 'min_input': 0.05, 'max_input': 0.35 },
        # { 'source': 'thumb_pinky_pinch', 'channel': 1, 'cc': 18, 'invert': True, 'min_input': 0.07, 'max_input': 0.4 },

    ],
    'display': {
        'show_window': True,
        'draw_landmarks': True,
        'draw_connections': True,
        'show_fps': True,
        'flip_horizontal': True
    }
}

def load_config(config_path='config.yaml'):
    """Loads configuration from YAML file, providing defaults."""
    config = DEFAULT_CONFIG.copy() # Start with defaults

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            if user_config:
                # Deep merge user config into defaults
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                         # Special handling for mappings list: overwrite instead of merge if user provides it
                         if key == 'mappings':
                              config[key] = value
                         else:
                              # Merge other dictionaries
                              config[key].update(value)
                    else:
                         # Overwrite top-level keys or lists (like mappings if not handled above)
                         config[key] = value
        except Exception as e:
            print(f"Warning: Could not load or parse {config_path}. Using default settings. Error: {e}")
    else:
        print("Info: config.yaml not found. Using default settings. A default config.yaml will be created.")
        # Save the default config with examples if file doesn't exist
        save_config(config, config_path) # Use the full DEFAULT_CONFIG


    # Ensure essential keys exist even if user file is malformed (simple check)
    for key, value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = value
        elif isinstance(value, dict):
             # Ensure sub-keys exist within nested dicts
             if key != 'mappings': # Don't force default mappings if user provided some
                for sub_key, sub_value in value.items():
                    if sub_key not in config.get(key, {}): # Check if key exists first
                         if key not in config: config[key] = {} # Create dict if missing
                         config[key][sub_key] = sub_value

    # Ensure 'mappings' exists as a list, even if empty
    if 'mappings' not in config or not isinstance(config['mappings'], list):
        config['mappings'] = [] # Default to empty list if missing or wrong type

    print("Configuration loaded.")
    # print(f"Using config: {yaml.dump(config)}") # Pretty print for debugging
    return config

def save_config(config, config_path='config.yaml'):
     """Saves the current configuration to YAML file."""
     try:
         # Add comments before saving the default config for the first time
         comment = """# CamMIDI Configuration
#
# For sources using distances ('curl', 'pinch', 'wrist_z'):
# - These are sensitive to hand size and camera distance.
# - Values are often normalized relative to hand size (e.g., wrist-to-middle-MCP distance).
# - **You MUST calibrate 'min_input' and 'max_input' for your setup!**
#   Observe the raw values printed in the console or displayed on screen
#   (you might need to temporarily add prints in mapper.py) to find your working range.
# - 'invert: True' is common for curls/pinches (small distance = high MIDI value).
#
# For sources using angles ('pitch', 'yaw', 'roll', 'spread'):
# - Values are typically in radians.
# - Ranges like -pi/2 to +pi/2 (-90 to +90 deg) or 0 to pi/4 (0 to 45 deg) are common starting points.
# - Calibrate 'min_input' / 'max_input' based on your comfortable range of motion.
#
# MIDI:
# - Set 'port_name' to your virtual MIDI port (e.g., "IAC Driver Bus 1" on macOS, "loopMIDI Port" on Windows).
#
"""
         with open(config_path, 'w') as f:
             f.write(comment)
             # Use numpy in dump? No, better convert np constants in DEFAULT_CONFIG
             # Tweak DEFAULT_CONFIG to use float approximations for saving
             config_to_save = config.copy() # Work on a copy
             for mapping in config_to_save.get('mappings',[]):
                  if 'min_input' in mapping and isinstance(mapping['min_input'], float) and np.isinf(mapping['min_input']):
                      mapping['min_input'] = -1.5708 # approx -pi/2
                  if 'max_input' in mapping and isinstance(mapping['max_input'], float) and np.isinf(mapping['max_input']):
                      mapping['max_input'] = 1.5708 # approx pi/2

             yaml.dump(config_to_save, f, default_flow_style=False, sort_keys=False)
         print(f"Configuration saved to {config_path}")
     except Exception as e:
         print(f"Error saving configuration to {config_path}: {e}")


if __name__ == '__main__':
    # Example usage: Load and print config
    cfg = load_config()
    # print(yaml.dump(cfg)) # Use yaml dump for readability
    # Example of accessing a value
    # print(f"Smoothing Factor: {cfg['midi']['smoothing_factor']}")
    # print(f"First mapping source: {cfg['mappings'][0]['source'] if cfg['mappings'] else 'None'}")
    # Example of saving (useful if you modify config programmatically)
    # save_config(cfg)