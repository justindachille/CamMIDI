import yaml
import os

DEFAULT_CONFIG = {
    'camera': {
        'index': 0,
        'width': 640, # Request width (driver might ignore)
        'height': 480 # Request height (driver might ignore)
    },
    'mediapipe': {
        'static_image_mode': False,
        'max_num_hands': 1, # Start with one hand for simplicity
        'min_detection_confidence': 0.6,
        'min_tracking_confidence': 0.6,
        'centroid_landmark_id': 0 # 0 = Wrist, 9 = Middle Finger MCP
    },
    'midi': {
        'port_name': "IAC Driver Bus 1", # CHANGE THIS for your system (macOS default)
                                         # Windows might use "loopMIDI Port"
        'smoothing_factor': 0.6 # 0.0 (none) to < 1.0 (heavy smoothing). Lower = faster response, more jitter.
    },
    'mappings': [
        # --- Define your MIDI mappings here ---
        {
            'source': 'centroid_x', # Use 'centroid_x', 'centroid_y' for now
            'channel': 1,
            'cc': 74, # Example: Filter Cutoff
            'invert': False,
             # Input range will be determined by camera width/height automatically
        },
        {
            'source': 'centroid_y',
            'channel': 1,
            'cc': 71, # Example: Filter Resonance
            'invert': True # Screen Y often increases downwards, MIDI usually low->high
        },
        # { # Example for future: Pinch distance (needs implementation in tracker/mapper)
        #   'source': 'pinch_distance', # Custom source name
        #   'type': 'pinch', # Tells mapper to use specific logic
        #   'finger1_id': 4, # Thumb tip
        #   'finger2_id': 8, # Index finger tip
        #   'channel': 1,
        #   'cc': 1, # Mod Wheel
        #   'invert': True, # Small distance = High MIDI value
        #   'min_input': 0.01, # Minimum expected distance (normalized)
        #   'max_input': 0.3   # Maximum expected distance (normalized)
        # }
    ],
    'display': {
        'show_window': True,
        'draw_landmarks': True,
        'draw_connections': True,
        'show_fps': True,
        'flip_horizontal': True # Mirror view for intuitive control
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
                # Deep merge user config into defaults (simple version)
                # For nested dicts like 'camera', 'midi', etc.
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                        config[key].update(value)
                    else:
                        config[key] = value # Overwrite lists or top-level keys
        except Exception as e:
            print(f"Warning: Could not load or parse {config_path}. Using default settings. Error: {e}")
    else:
        print("Info: config.yaml not found. Using default settings. A default config.yaml will be created.")
        save_config(config, config_path) # Create one if it doesn't exist


    # Ensure essential keys exist even if user file is malformed
    for key, value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = value
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                 if sub_key not in config[key]:
                     config[key][sub_key] = sub_value

    print("Configuration loaded.")
    # print(f"Using config: {config}") # Uncomment for debugging
    return config

def save_config(config, config_path='config.yaml'):
     """Saves the current configuration to YAML file."""
     try:
         with open(config_path, 'w') as f:
             yaml.dump(config, f, default_flow_style=False, sort_keys=False)
         print(f"Configuration saved to {config_path}")
     except Exception as e:
         print(f"Error saving configuration to {config_path}: {e}")


if __name__ == '__main__':
    # Example usage: Load and print config
    cfg = load_config()
    # print(cfg)
    # Example of accessing a value
    # print(f"MIDI Port: {cfg['midi']['port_name']}")
    # print(f"Centroid Landmark ID: {cfg['mediapipe']['centroid_landmark_id']}")
    # Example of saving (useful if you modify config programmatically)
    # save_config(cfg)