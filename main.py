# main.py
import cv2
import time
import numpy as np
import collections
import argparse
import warnings

# Import config_loader functions directly
from config_loader import load_config, save_config
from tracker import HandTracker
from mapper import MidiMapper
from midi_sender import MidiSender

# --- Global Variables for GUI / Calibration ---
buttons = {}
hovered_button_key = None
calibration_action = None # Stores the key of the button clicked
window_name = "CamMIDI Output"
last_calibrated_message = ""
message_display_time = 0

def draw_text_outlined(img, text, pos, font, scale, fg_color, bg_color, fg_thickness, bg_thickness_add=2):
    """Draws text with a background outline."""
    x, y = pos
    cv2.putText(img, text, (x, y), font, scale, bg_color, fg_thickness + bg_thickness_add, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, fg_color, fg_thickness, cv2.LINE_AA)

def define_buttons(frame_width, frame_height):
    """Defines button locations and text based on frame size."""
    global buttons
    buttons.clear() # Clear previous definitions

    button_h = 28 # Slightly smaller height
    button_w = 145 # Slightly wider
    spacing = 4
    bottom_margin = spacing + 5 # Space from bottom edge
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45 # Slightly smaller font
    fg_color = (255, 255, 255)
    bg_color = (100, 100, 100)
    hover_color = (130, 130, 130) # Color when hovered
    action_color = (160, 160, 160) # Color briefly when clicked

    # --- ROW 1: Pinches (Index, Middle) ---
    row1_y = frame_height - bottom_margin - 2 * button_h - spacing # Top row of buttons
    start_x = spacing

    # T-Index
    buttons['calib_ti_pinch_min'] = {
        'rect': (start_x, row1_y, button_w, button_h),
        'text': 'Set T-Idx MIN', 'action': 'calib_ti_pinch_min', 'help': 'Click when Thumb & Index tips are CLOSEST',
        'font': font, 'scale': font_scale, 'fg': fg_color, 'bg': bg_color, 'hover': hover_color, 'action_c': action_color
    }
    start_x += button_w + spacing
    buttons['calib_ti_pinch_max'] = {
        'rect': (start_x, row1_y, button_w, button_h),
        'text': 'Set T-Idx MAX', 'action': 'calib_ti_pinch_max', 'help': 'Click when Thumb & Index tips are FARTHEST apart',
        'font': font, 'scale': font_scale, 'fg': fg_color, 'bg': bg_color, 'hover': hover_color, 'action_c': action_color
    }
    start_x += button_w + spacing + 10 # Gap

    # T-Middle
    buttons['calib_tm_pinch_min'] = {
        'rect': (start_x, row1_y, button_w, button_h),
        'text': 'Set T-Mid MIN', 'action': 'calib_tm_pinch_min', 'help': 'Click when Thumb & Middle tips are CLOSEST',
        'font': font, 'scale': font_scale, 'fg': fg_color, 'bg': bg_color, 'hover': hover_color, 'action_c': action_color
    }
    start_x += button_w + spacing
    buttons['calib_tm_pinch_max'] = {
        'rect': (start_x, row1_y, button_w, button_h),
        'text': 'Set T-Mid MAX', 'action': 'calib_tm_pinch_max', 'help': 'Click when Thumb & Middle tips are FARTHEST apart',
        'font': font, 'scale': font_scale, 'fg': fg_color, 'bg': bg_color, 'hover': hover_color, 'action_c': action_color
    }

    # --- ROW 2: Pinches (Ring, Pinky) ---
    row2_y = frame_height - bottom_margin - button_h # Middle row
    start_x = spacing

    # T-Ring
    buttons['calib_tr_pinch_min'] = {
        'rect': (start_x, row2_y, button_w, button_h),
        'text': 'Set T-Ring MIN', 'action': 'calib_tr_pinch_min', 'help': 'Click when Thumb & Ring tips are CLOSEST',
        'font': font, 'scale': font_scale, 'fg': fg_color, 'bg': bg_color, 'hover': hover_color, 'action_c': action_color
    }
    start_x += button_w + spacing
    buttons['calib_tr_pinch_max'] = {
        'rect': (start_x, row2_y, button_w, button_h),
        'text': 'Set T-Ring MAX', 'action': 'calib_tr_pinch_max', 'help': 'Click when Thumb & Ring tips are FARTHEST apart',
        'font': font, 'scale': font_scale, 'fg': fg_color, 'bg': bg_color, 'hover': hover_color, 'action_c': action_color
    }
    start_x += button_w + spacing + 10 # Gap

    # T-Pinky
    buttons['calib_tp_pinch_min'] = {
        'rect': (start_x, row2_y, button_w, button_h),
        'text': 'Set T-Pink MIN', 'action': 'calib_tp_pinch_min', 'help': 'Click when Thumb & Pinky tips are CLOSEST',
        'font': font, 'scale': font_scale, 'fg': fg_color, 'bg': bg_color, 'hover': hover_color, 'action_c': action_color
    }
    start_x += button_w + spacing
    buttons['calib_tp_pinch_max'] = {
        'rect': (start_x, row2_y, button_w, button_h),
        'text': 'Set T-Pink MAX', 'action': 'calib_tp_pinch_max', 'help': 'Click when Thumb & Pinky tips are FARTHEST apart',
        'font': font, 'scale': font_scale, 'fg': fg_color, 'bg': bg_color, 'hover': hover_color, 'action_c': action_color
    }

    # --- ROW 3: Spread & Save ---
    # Let's place Spread on the left of row 3, Save on the right
    row3_y = frame_height - bottom_margin # Bottom row - but we need space for help text maybe put row 1/2 lower
    row2_y = frame_height - bottom_margin - button_h
    row1_y = frame_height - bottom_margin - 2*button_h - spacing

    # Recalculate Y for rows 1 and 2 to make space below row 2
    row3_y = frame_height - bottom_margin - button_h # New bottom row Y
    row2_y = row3_y - button_h - spacing
    row1_y = row2_y - button_h - spacing
    help_text_y = frame_height - bottom_margin + 2 # Y position for help text

    # Redefine buttons with corrected Y
    # Row 1
    buttons['calib_ti_pinch_min']['rect'] = (buttons['calib_ti_pinch_min']['rect'][0], row1_y, button_w, button_h)
    buttons['calib_ti_pinch_max']['rect'] = (buttons['calib_ti_pinch_max']['rect'][0], row1_y, button_w, button_h)
    buttons['calib_tm_pinch_min']['rect'] = (buttons['calib_tm_pinch_min']['rect'][0], row1_y, button_w, button_h)
    buttons['calib_tm_pinch_max']['rect'] = (buttons['calib_tm_pinch_max']['rect'][0], row1_y, button_w, button_h)
    # Row 2
    buttons['calib_tr_pinch_min']['rect'] = (buttons['calib_tr_pinch_min']['rect'][0], row2_y, button_w, button_h)
    buttons['calib_tr_pinch_max']['rect'] = (buttons['calib_tr_pinch_max']['rect'][0], row2_y, button_w, button_h)
    buttons['calib_tp_pinch_min']['rect'] = (buttons['calib_tp_pinch_min']['rect'][0], row2_y, button_w, button_h)
    buttons['calib_tp_pinch_max']['rect'] = (buttons['calib_tp_pinch_max']['rect'][0], row2_y, button_w, button_h)

    # Row 3 - Spread
    start_x = spacing
    buttons['calib_spread_min'] = {
        'rect': (start_x, row3_y, button_w, button_h),
        'text': 'Set Spread MIN', 'action': 'calib_spread_min', 'help': 'Click when fingers are SQUEEZED together',
        'font': font, 'scale': font_scale, 'fg': fg_color, 'bg': bg_color, 'hover': hover_color, 'action_c': action_color
    }
    start_x += button_w + spacing
    buttons['calib_spread_max'] = {
        'rect': (start_x, row3_y, button_w, button_h),
        'text': 'Set Spread MAX', 'action': 'calib_spread_max', 'help': 'Click when fingers are SPREAD APART maximally',
        'font': font, 'scale': font_scale, 'fg': fg_color, 'bg': bg_color, 'hover': hover_color, 'action_c': action_color
    }

    # Row 3 - Save (Far Right)
    save_w = 100 # Smaller width for save
    save_x = frame_width - save_w - spacing
    buttons['save_config'] = {
        'rect': (save_x, row3_y, save_w, button_h),
        'text': 'Save Config', 'action': 'save_config', 'help': 'Save current calibrated min/max values to config.yaml',
        'font': font, 'scale': font_scale, 'fg': (0, 255, 0), 'bg': (0, 80, 0), 'hover': (0, 120, 0), 'action_c': (0,160,0)
    }

    # Store help text position
    buttons['help_text_pos'] = (spacing, help_text_y)


def draw_buttons(frame):
    """Draws all defined buttons and hover help text on the frame."""
    global hovered_button_key
    active_help_text = None

    for key, btn in buttons.items():
        if 'rect' not in btn: continue # Skip help_text_pos entry

        x, y, w, h = btn['rect']
        bg = btn['bg']
        # Check if this button is hovered
        is_hovered = (key == hovered_button_key)
        if is_hovered:
            bg = btn['hover']
            active_help_text = btn.get('help') # Get help text if hovered

        # Briefly change color if it was the last action clicked
        if key == calibration_action: # Use the global action flag
             bg = btn['action_c']

        # Draw background rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1)
        # Draw border
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

        # Calculate text position for centering
        text_size = cv2.getTextSize(btn['text'], btn['font'], btn['scale'], 1)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2

        # Draw text
        cv2.putText(frame, btn['text'], (text_x, text_y), btn['font'], btn['scale'], btn['fg'], 1, cv2.LINE_AA)

    # Draw the active help text (if any)
    if active_help_text and 'help_text_pos' in buttons:
        help_x, help_y = buttons['help_text_pos']
        draw_text_outlined(frame, f"Info: {active_help_text}", (help_x, help_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), (0,0,0), 1)


def mouse_callback(event, x, y, flags, param):
    """Handles mouse events (clicks and movement for hover)."""
    global calibration_action, hovered_button_key
    
    # Update hover state on movement
    if event == cv2.EVENT_MOUSEMOVE:
        new_hover_key = None
        for key, btn in buttons.items():
            if 'rect' not in btn: continue
            bx, by, bw, bh = btn['rect']
            if bx <= x <= bx + bw and by <= y <= by + bh:
                new_hover_key = key
                break
        # Only update if hover state changed to reduce redraw flicker potentially
        if new_hover_key != hovered_button_key:
            hovered_button_key = new_hover_key

    # Handle clicks
    elif event == cv2.EVENT_LBUTTONDOWN:
        # Reset hover state on click start (optional)
        # hovered_button_key = None
        clicked_on_button = False
        for key, btn in buttons.items():
            if 'rect' not in btn: continue
            bx, by, bw, bh = btn['rect']
            if bx <= x <= bx + bw and by <= y <= by + bh:
                calibration_action = btn['action'] # Set the action flag
                print(f"Button '{btn['text']}' clicked. Action: {calibration_action}")
                clicked_on_button = True
                break
        # If click was not on a button, clear the action (prevents accidental repeats)
        # if not clicked_on_button:
        #      calibration_action = None # Maybe not needed if action is reset in main loop

def update_config_value(config, source_name, param_name, value, hand_index=0):
    """Updates min_input or max_input for a specific source in the config, ensuring min < max."""
    global last_calibrated_message, message_display_time
    found_mapping = False
    updated = False # Flag to check if update actually happened

    for mapping in config.get('mappings', []):
        if isinstance(mapping, dict) and mapping.get('source') == source_name:
            found_mapping = True
            current_val = mapping.get(param_name)

            # --- Check for min/max inversion and auto-correct ---
            other_param = 'max_input' if param_name == 'min_input' else 'min_input'
            other_val = mapping.get(other_param) # Get the *other* boundary

            corrected_value = value
            swap_occurred = False

            if param_name == 'min_input':
                # Setting MIN: if new min >= current max, set max to be slightly larger than new min
                if other_val is not None and value >= other_val:
                    mapping['max_input'] = value + 0.01 # Ensure max is always > min
                    print(f"Warning: New min ({value:.4f}) >= current max ({other_val:.4f}). Auto-adjusting max.")
                    swap_occurred = True # Technically an adjustment, not a swap
                mapping[param_name] = value # Set the new min
                updated = True

            elif param_name == 'max_input':
                # Setting MAX: if new max <= current min, set min to be slightly smaller than new max
                if other_val is not None and value <= other_val:
                    mapping['min_input'] = value - 0.01 if value > 0 else 0.0 # Ensure min is always < max (and non-negative)
                    print(f"Warning: New max ({value:.4f}) <= current min ({other_val:.4f}). Auto-adjusting min.")
                    swap_occurred = True # Adjustment
                mapping[param_name] = value # Set the new max
                updated = True

            # --- Format Message ---
            if updated:
                final_min = mapping.get('min_input', 'N/A')
                final_max = mapping.get('max_input', 'N/A')
                try: final_min_f = f"{final_min:.3f}"
                except: final_min_f = str(final_min)
                try: final_max_f = f"{final_max:.3f}"
                except: final_max_f = str(final_max)

                msg = f"Calibrated {source_name}: Range [{final_min_f}, {final_max_f}]"
                if swap_occurred: msg += " (Adjusted)"
                print(msg)
                last_calibrated_message = msg
                message_display_time = time.time()

            break # Assume only one mapping per source

    if not found_mapping:
        msg = f"Warning: No mapping found for source '{source_name}'. Cannot calibrate."
        print(msg)
        last_calibrated_message = msg
        message_display_time = time.time()

    # Return flag indicating if an update was made to trigger mapper refresh
    return updated

# --- Main Function ---
def main():
    global calibration_action, last_calibrated_message, message_display_time

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="CamMIDI: Control MIDI with Hand Tracking.")
    parser.add_argument('--use-defaults', action='store_true', help="Force use of default configuration, ignore config.yaml file.")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the configuration file.")
    args = parser.parse_args()
    config_path = args.config

    # --- Initialization ---
    config = load_config(config_path=config_path, force_defaults=args.use_defaults)
    if args.use_defaults: print("Info: --use-defaults flag detected. Ignoring config.yaml.")

    tracker = HandTracker(config)
    mapper = MidiMapper(config) # Mapper reads force_channel1_hand config now
    midi_sender = MidiSender(config)

    # --- Camera Setup ---
    cam_idx = config.get('camera', {}).get('index', 0)
    req_w = config.get('camera', {}).get('width', 800) # Use config defaults
    req_h = config.get('camera', {}).get('height', 460)
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Error: Could not open camera index {cam_idx}.")
        if midi_sender: midi_sender.close()
        if tracker: tracker.close()
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened. Requested: {req_w}x{req_h}, Actual: {actual_width}x{actual_height}")

    display_enabled = config.get('display', {}).get('show_window', True)
    if display_enabled:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, actual_width, actual_height)
        define_buttons(actual_width, actual_height) # Define buttons based on actual size
        cv2.setMouseCallback(window_name, mouse_callback) # Set the mouse callback

    pTime = 0

    # --- Startup Info ---
    print("\nStarting CamMIDI loop. Press 'Q' in the output window to quit.")
    print("Use UI buttons to calibrate Pinch/Spread min/max values.")
    print("Press 'Save Config' button in UI to save calibrated values.")
    print("Ensure your DAW is configured to receive MIDI from:", config.get('midi', {}).get('port_name', 'N/A'))
    max_h = config.get('mediapipe', {}).get('max_num_hands', 1)
    force_ch1 = config.get('midi', {}).get('force_channel1_hand', 'None')
    print(f"Tracking up to {max_h} hands.")
    print(f" Channel 1 Assignment: {force_ch1} hand preference (default: first detected)")
    if max_h > 1:
        print(f" Implicit Ch2 mapping: ON (uses SAME CC as Channel 1). Explicit Ch2 mappings override.")
    print("---")
    print("CALIBRATION NOTE: Adjust 'min_input'/'max_input' in config.yaml (or use UI buttons)!")
    print("---")

    # --- Main Loop ---
    try:
        action_display_start_time = 0
        last_action_for_draw = None # Used for button feedback visualization
        while True:
            # (Frame capture)
            success, frame = cap.read()
            if not success or frame is None:
                 print("Warning: Failed to capture frame.")
                 time.sleep(0.1) # Avoid busy-looping on error
                 continue

            # 1. Track Hands
            # Important: process_frame might flip the frame based on config
            processed_frame, all_hands_data = tracker.process_frame(frame)

            # --- Handle Calibration Actions ---
            if calibration_action:
                current_action = calibration_action
                last_action_for_draw = current_action # Store for visual feedback
                action_display_start_time = time.time()
                calibration_action = None # Reset flag immediately
                config_updated = False

                # Find the first *detected* hand for calibration data
                # Calibration always uses the first physically detected hand (index 0 if found)
                first_hand_data = None
                first_hand_index = -1
                for idx, hd in enumerate(all_hands_data):
                    if hd.get('found', False):
                        first_hand_data = hd
                        first_hand_index = idx
                        break # Found the first one

                if first_hand_data is None:
                    msg = "Calibration failed: No hand detected."
                    print(msg)
                    last_calibrated_message = msg
                    message_display_time = time.time()
                else:
                     # --- Perform Calibration based on current_action ---
                     # (Spread actions using first detected hand's data)
                     if current_action == 'calib_spread_min':
                         # Calibrate all spread sources based on the first detected hand's current state
                         config_updated |= update_config_value(config, 'index_middle_spread', 'min_input', first_hand_data.get('index_middle_spread', 0.0))
                         config_updated |= update_config_value(config, 'middle_ring_spread', 'min_input', first_hand_data.get('middle_ring_spread', 0.0))
                         config_updated |= update_config_value(config, 'ring_pinky_spread', 'min_input', first_hand_data.get('ring_pinky_spread', 0.0))
                     elif current_action == 'calib_spread_max':
                         config_updated |= update_config_value(config, 'index_middle_spread', 'max_input', first_hand_data.get('index_middle_spread', np.pi/4))
                         config_updated |= update_config_value(config, 'middle_ring_spread', 'max_input', first_hand_data.get('middle_ring_spread', np.pi/5))
                         config_updated |= update_config_value(config, 'ring_pinky_spread', 'max_input', first_hand_data.get('ring_pinky_spread', np.pi/5))
                     # (Pinch actions - T-Idx, T-Mid, T-Ring, T-Pinky)
                     elif current_action == 'calib_ti_pinch_min': config_updated |= update_config_value(config, 'thumb_index_pinch', 'min_input', first_hand_data.get('thumb_index_pinch', 0.01))
                     elif current_action == 'calib_ti_pinch_max': config_updated |= update_config_value(config, 'thumb_index_pinch', 'max_input', first_hand_data.get('thumb_index_pinch', 0.3))
                     elif current_action == 'calib_tm_pinch_min': config_updated |= update_config_value(config, 'thumb_middle_pinch', 'min_input', first_hand_data.get('thumb_middle_pinch', 0.02))
                     elif current_action == 'calib_tm_pinch_max': config_updated |= update_config_value(config, 'thumb_middle_pinch', 'max_input', first_hand_data.get('thumb_middle_pinch', 0.35))
                     elif current_action == 'calib_tr_pinch_min': config_updated |= update_config_value(config, 'thumb_ring_pinch', 'min_input', first_hand_data.get('thumb_ring_pinch', 0.03))
                     elif current_action == 'calib_tr_pinch_max': config_updated |= update_config_value(config, 'thumb_ring_pinch', 'max_input', first_hand_data.get('thumb_ring_pinch', 0.4))
                     elif current_action == 'calib_tp_pinch_min': config_updated |= update_config_value(config, 'thumb_pinky_pinch', 'min_input', first_hand_data.get('thumb_pinky_pinch', 0.05))
                     elif current_action == 'calib_tp_pinch_max': config_updated |= update_config_value(config, 'thumb_pinky_pinch', 'max_input', first_hand_data.get('thumb_pinky_pinch', 0.45))
                     # (Save action)
                     elif current_action == 'save_config':
                         save_config(config, config_path)
                         last_calibrated_message = f"Configuration saved to {config_path}"
                         message_display_time = time.time()
                         config_updated = False # No mapper update needed for save

                     # Update mapper if config values (min/max input) changed
                     if config_updated:
                          print("Config updated by calibration, refreshing mapper.")
                          # Pass only the mappings list to the update method
                          mapper.update_mappings(config.get('mappings', []))


            # 2. Map MIDI (using potentially updated mappings)
            # Pass the *actual* frame dimensions to the mapper for centroid calculation
            # Returns: {(channel, cc): {'value': value, 'hand_index': index}}
            midi_map = mapper.calculate_midi(all_hands_data, actual_width, actual_height)

            # 3. Send MIDI
            if midi_sender and midi_sender.port_opened:
                 # Adapt the loop to extract the value from the data dictionary
                 for (ch, cc), data in midi_map.items():
                      val = data['value'] # Extract value
                      midi_sender.send_cc(ch, cc, val) # Send it

            # 4. Visual Feedback
            if display_enabled:
                display_frame = processed_frame.copy()
                # Draw tracker visuals (landmarks, etc.) AFTER potential frame flip in tracker
                display_frame = tracker.draw_visuals(display_frame, all_hands_data)

                # FPS
                # ... (FPS drawing code remains the same) ...
                cTime = time.time(); delta_time = cTime - pTime
                fps = 1 / delta_time if delta_time > 0 else 0; pTime = cTime
                if config.get('display', {}).get('show_fps', True):
                    draw_text_outlined(display_frame, f"FPS: {int(fps)}", (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), (0,0,0), 2)


                # --- MIDI Debug Display (Updated Logic) ---
                messages_left = []  # Channel 2 messages (drawn on left)
                messages_right = [] # Channel 1 messages (drawn on right)

                # Build source lookup (remains the same logic as before)
                source_lookup = {}
                explicit_ch2_sources = mapper.explicit_ch2_sources # Get from mapper instance
                max_h_conf = config.get('mediapipe', {}).get('max_num_hands', 1) # Get max_hands from config
                for mapping in config.get('mappings', []):
                    if not isinstance(mapping, dict): continue
                    m_ch = mapping.get('channel')
                    m_cc = mapping.get('cc')
                    m_src = mapping.get('source')
                    if m_ch is None or m_cc is None or m_src is None: continue

                    # Store primary mapping (Ch1 or explicit Ch2)
                    source_lookup[(m_ch, m_cc)] = (m_src, False) # (source_name, is_implicit)

                    # If it's a Ch1 mapping and implicit Ch2 is possible, add potential implicit entry
                    if m_ch == 1 and max_h_conf > 1 and m_src not in explicit_ch2_sources:
                        # Check if an explicit Ch2 mapping ALREADY exists for the *same CC*
                        has_explicit_conflict_cc = any(
                            m2.get('channel') == 2 and m2.get('cc') == m_cc
                            for m2 in config.get('mappings', []) if isinstance(m2, dict)
                        )
                        if not has_explicit_conflict_cc:
                             # Add implicit mapping for Ch2, using same CC as Ch1
                             source_lookup[(2, m_cc)] = (m_src, True) # Mark as implicit

                # --- Iterate through the MIDI map generated by mapper ---
                for (ch, cc), data in midi_map.items():
                    val = data['value']
                    hand_index = data['hand_index'] # Get the *original detection index* of the hand

                    # Get corresponding hand data and handedness for display label
                    if 0 <= hand_index < len(all_hands_data):
                        hand_data = all_hands_data[hand_index]
                        detected_handedness = hand_data.get('handedness', 'Unknown')
                        # Use H1/H2 based on the *original detection index* for the label
                        hand_label = f"H{hand_index + 1}({detected_handedness[0]})"
                    else:
                        hand_label = f"H?({ch})" # Fallback if hand_index is invalid (shouldn't happen)

                    # Lookup source name based on the *output channel and CC*
                    source_info = source_lookup.get((ch, cc))
                    source_name = f"Src?({ch},{cc})"
                    mapping_type = ""
                    if source_info:
                        source_name, is_implicit = source_info
                        if is_implicit: mapping_type = "[Impl]"
                        elif ch == 2: mapping_type = "[Expl]" # Explicit Ch2 mapping

                    # Format the text string: HandLabel[Type] Source -> Ch CC: Value
                    text = f"{hand_label}{mapping_type} {source_name} -> Ch{ch} CC{cc}: {val}"

                    # Assign to display list based on the *output MIDI channel* (ch)
                    if ch == 1:
                        messages_right.append(text) # Channel 1 messages go on the right
                    elif ch == 2:
                        messages_left.append(text) # Channel 2 messages go on the left

                # Draw messages from the lists
                # ... (drawing logic remains the same, uses messages_left/messages_right) ...
                y_offset_left, y_offset_right = 60, 60
                font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                fg_col, bg_col = (255, 255, 255), (0, 0, 0)
                margin = 10
                for text in messages_left: # Draw left-aligned (Channel 2)
                     draw_text_outlined(display_frame, text, (margin, y_offset_left), font, scale, fg_col, bg_col, thick)
                     y_offset_left += 18
                for text in messages_right: # Draw right-aligned (Channel 1)
                    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
                    tx = max(0, actual_width - tw - margin) # Align to right
                    draw_text_outlined(display_frame, text, (tx, y_offset_right), font, scale, fg_col, bg_col, thick)
                    y_offset_right += 18
                # --- End MIDI Debug Display ---

                # --- Draw Buttons & Help Text ---
                # Clear button action highlight after a short delay
                if last_action_for_draw and (time.time() - action_display_start_time > 0.2):
                    last_action_for_draw = None # Stop highlighting

                # Pass the action flag (stored in last_action_for_draw) to draw_buttons
                # draw_buttons function needs slight modification to accept this
                # Let's assume draw_buttons checks against a global or passed state
                # No, draw_buttons already uses the global `calibration_action` for *brief* highlight
                # We need to update draw_buttons or use the `last_action_for_draw` state here
                # Reverting draw_buttons to just use hover effect for simplicity now.
                # The brief flash on click is handled by the global `calibration_action` logic reset above.
                draw_buttons(display_frame) # Hover effect handled by mouse callback updating hovered_button_key

                # --- Draw Calibration Status Message ---
                # ... (remains the same) ...
                font_status, scale_status, thick_status = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                if last_calibrated_message and (time.time() - message_display_time < 3.0):
                     (tw, th), _ = cv2.getTextSize(last_calibrated_message, font_status, scale_status, thick_status)
                     tx = (actual_width - tw) // 2
                     # Position below the lowest possible button + some margin, or below left debug text?
                     # Let's try below buttons. Find max button Y + height.
                     max_button_y = 0
                     for btn in buttons.values():
                         if 'rect' in btn:
                             max_button_y = max(max_button_y, btn['rect'][1] + btn['rect'][3])
                     ty = max_button_y + 20 # Position below buttons

                     draw_text_outlined(display_frame, last_calibrated_message, (tx, ty), font_status, scale_status, (0, 255, 255), bg_col, thick_status)
                elif last_calibrated_message and (time.time() - message_display_time >= 3.0):
                    last_calibrated_message = "" # Clear message after timeout


                # --- Show Frame ---
                cv2.imshow(window_name, display_frame)

            # Exit Check & Save Shortcut
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('s'):
                 print("Save triggered by keyboard ('s')")
                 save_config(config, config_path)
                 last_calibrated_message = f"Config saved to {config_path} (Key 's')"
                 message_display_time = time.time()


    except KeyboardInterrupt: print("Keyboard interrupt received.")
    finally:
        # Cleanup
        print("Shutting down...")
        if cap and cap.isOpened(): cap.release()
        if display_enabled: cv2.destroyAllWindows()
        if tracker: tracker.close()
        if midi_sender: midi_sender.close()
        print("CamMIDI stopped.")

if __name__ == "__main__":
    main()