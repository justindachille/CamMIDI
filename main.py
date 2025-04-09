# main.py
import cv2
import time
import numpy as np
import collections
import argparse
import warnings

from config_loader import load_config, save_config
from hand_tracker import HandTracker
from face_tracker import FaceTracker
from mapper import MidiMapper
from midi_sender import MidiSender

# --- Global Variables for GUI / Calibration ---
buttons = {}
hovered_button_key = None
calibration_action = None # Stores the key of the button clicked
window_name = "CamMIDI Output"
last_calibrated_message = ""
message_display_time = 0
last_action_key_clicked = None
last_action_time = 0

is_auto_calibrating = False
auto_calib_ranges = {} # Stores {'source_name': {'min': val, 'max': val}}
calibration_epsilon = 0.01

# --- GUI Functions (draw_text_outlined, define_buttons, draw_buttons, mouse_callback) ---
# These remain largely the same, define_buttons might need slight adjustment
# if adding face calibration buttons later.
def draw_text_outlined(img, text, pos, font, scale, fg_color, bg_color, fg_thickness, bg_thickness_add=2):
    """Draws text with a background outline."""
    x, y = pos
    cv2.putText(img, text, (x, y), font, scale, bg_color, fg_thickness + bg_thickness_add, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, fg_color, fg_thickness, cv2.LINE_AA)

def define_buttons(frame_width, frame_height):
    """Defines button locations and text based on frame size."""
    global buttons, is_auto_calibrating # Need is_auto_calibrating to set button text
    buttons.clear() # Clear previous definitions

    button_h = 28
    button_w = 145
    spacing = 4
    bottom_margin = spacing + 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    fg_color = (255, 255, 255)
    bg_color = (100, 100, 100)
    hover_color = (130, 130, 130)
    action_color = (160, 160, 160)

    # Calculate Y positions to allow space for help text below
    help_text_height_allowance = 20 # Space for one line of help text
    row4_y = frame_height - bottom_margin - button_h # New row for Auto-Calibrate
    row3_y = row4_y - button_h - spacing
    row2_y = row3_y - button_h - spacing
    row1_y = row2_y - button_h - spacing
    help_text_y = frame_height - bottom_margin + 2 # Y position below last row

    # --- ROW 1: Pinches (Index, Middle) --- (Keep as is)
    start_x = spacing
    buttons['calib_ti_pinch_min'] = {'rect': (start_x, row1_y, button_w, button_h), 'text': 'Set T-Idx MIN', 'action': 'calib_ti_pinch_min', 'help': 'Click when Thumb & Index tips are CLOSEST', 'font': font, 'scale': font_scale, 'fg': fg_color, 'bg': bg_color, 'hover': hover_color, 'action_c': action_color}
    start_x += button_w + spacing
    buttons['calib_ti_pinch_max'] = {'rect': (start_x, row1_y, button_w, button_h), 'text': 'Set T-Idx MAX', 'action': 'calib_ti_pinch_max', 'help': 'Click when Thumb & Index tips are FARTHEST apart', 'font': font, 'scale': font_scale, 'fg': fg_color, 'bg': bg_color, 'hover': hover_color, 'action_c': action_color}
    start_x += button_w + spacing + 10 # Gap
    buttons['calib_tm_pinch_min'] = {'rect': (start_x, row1_y, button_w, button_h), 'text': 'Set T-Mid MIN', 'action': 'calib_tm_pinch_min', 'help': 'Click when Thumb & Middle tips are CLOSEST', 'font': font, 'scale': font_scale, 'fg': fg_color, 'bg': bg_color, 'hover': hover_color, 'action_c': action_color}
    start_x += button_w + spacing
    buttons['calib_tm_pinch_max'] = {'rect': (start_x, row1_y, button_w, button_h), 'text': 'Set T-Mid MAX', 'action': 'calib_tm_pinch_max', 'help': 'Click when Thumb & Middle tips are FARTHEST apart', 'font': font, 'scale': font_scale, 'fg': fg_color, 'bg': bg_color, 'hover': hover_color, 'action_c': action_color}

    # --- ROW 2: Pinches (Ring, Pinky) --- (Keep as is)
    start_x = spacing
    buttons['calib_tr_pinch_min'] = {'rect': (start_x, row2_y, button_w, button_h), 'text': 'Set T-Ring MIN', 'action': 'calib_tr_pinch_min', 'help': 'Click when Thumb & Ring tips are CLOSEST', 'font': font, 'scale': font_scale, 'fg': fg_color, 'bg': bg_color, 'hover': hover_color, 'action_c': action_color}
    start_x += button_w + spacing
    buttons['calib_tr_pinch_max'] = {'rect': (start_x, row2_y, button_w, button_h), 'text': 'Set T-Ring MAX', 'action': 'calib_tr_pinch_max', 'help': 'Click when Thumb & Ring tips are FARTHEST apart', 'font': font, 'scale': font_scale, 'fg': fg_color, 'bg': bg_color, 'hover': hover_color, 'action_c': action_color}
    start_x += button_w + spacing + 10 # Gap
    buttons['calib_tp_pinch_min'] = {'rect': (start_x, row2_y, button_w, button_h), 'text': 'Set T-Pink MIN', 'action': 'calib_tp_pinch_min', 'help': 'Click when Thumb & Pinky tips are CLOSEST', 'font': font, 'scale': font_scale, 'fg': fg_color, 'bg': bg_color, 'hover': hover_color, 'action_c': action_color}
    start_x += button_w + spacing
    buttons['calib_tp_pinch_max'] = {'rect': (start_x, row2_y, button_w, button_h), 'text': 'Set T-Pink MAX', 'action': 'calib_tp_pinch_max', 'help': 'Click when Thumb & Pinky tips are FARTHEST apart', 'font': font, 'scale': font_scale, 'fg': fg_color, 'bg': bg_color, 'hover': hover_color, 'action_c': action_color}

    # --- ROW 3: Spread --- (Keep as is)
    start_x = spacing
    buttons['calib_spread_min'] = {'rect': (start_x, row3_y, button_w, button_h), 'text': 'Set Spread MIN', 'action': 'calib_spread_min', 'help': 'Click when fingers are SQUEEZED together', 'font': font, 'scale': font_scale, 'fg': fg_color, 'bg': bg_color, 'hover': hover_color, 'action_c': action_color}
    start_x += button_w + spacing
    buttons['calib_spread_max'] = {'rect': (start_x, row3_y, button_w, button_h), 'text': 'Set Spread MAX', 'action': 'calib_spread_max', 'help': 'Click when fingers are SPREAD APART maximally', 'font': font, 'scale': font_scale, 'fg': fg_color, 'bg': bg_color, 'hover': hover_color, 'action_c': action_color}

    # --- ROW 4: Auto-Calibrate & Save ---
    start_x = spacing
    # Dynamic text/color for Auto-Calibrate button
    auto_calib_text = "Stop Calibrating" if is_auto_calibrating else "Auto-Calibrate ALL"
    auto_calib_fg = (255, 150, 0) if is_auto_calibrating else (0, 200, 255)
    auto_calib_bg = (80, 40, 0) if is_auto_calibrating else (0, 60, 80)
    auto_calib_hover = (100, 60, 0) if is_auto_calibrating else (0, 80, 100)
    auto_calib_action_c = (120, 80, 0) if is_auto_calibrating else (0, 100, 120)
    auto_calib_help = "Click to STOP auto range detection" if is_auto_calibrating else "Click to START auto detecting min/max for ALL controls"
    auto_button_w = button_w * 2 + spacing # Make it wider
    buttons['toggle_auto_calibrate'] = {'rect': (start_x, row4_y, auto_button_w, button_h), 'text': auto_calib_text, 'action': 'toggle_auto_calibrate', 'help': auto_calib_help, 'font': font, 'scale': font_scale, 'fg': auto_calib_fg, 'bg': auto_calib_bg, 'hover': auto_calib_hover, 'action_c': auto_calib_action_c}

    # Save Button (Far Right)
    save_w = 100
    save_x = frame_width - save_w - spacing
    buttons['save_config'] = {'rect': (save_x, row4_y, save_w, button_h), 'text': 'Save Config', 'action': 'save_config', 'help': 'Save current calibrated values to config.yaml', 'font': font, 'scale': font_scale, 'fg': (0, 255, 0), 'bg': (0, 80, 0), 'hover': (0, 120, 0), 'action_c': (0,160,0)}

    buttons['help_text_pos'] = (spacing, help_text_y)

def draw_buttons(frame, highlight_duration=0.2):
    """Draws buttons, handling hover and brief click highlight."""
    global hovered_button_key, last_action_key_clicked, last_action_time
    active_help_text = None
    current_time = time.time()

    show_action_highlight = (last_action_key_clicked is not None and
                             (current_time - last_action_time <= highlight_duration))

    for key, btn in buttons.items():
        if 'rect' not in btn: continue

        x, y, w, h = btn['rect']
        bg = btn['bg'] # Default background

        # Prioritize Action Highlight -> Hover -> Default
        if show_action_highlight and key == last_action_key_clicked:
            bg = btn['action_c']
        elif key == hovered_button_key:
            bg = btn['hover']
            active_help_text = btn.get('help')

        cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

        text_size = cv2.getTextSize(btn['text'], btn['font'], btn['scale'], 1)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(frame, btn['text'], (text_x, text_y), btn['font'], btn['scale'], btn['fg'], 1, cv2.LINE_AA)

    if active_help_text and 'help_text_pos' in buttons:
        help_x, help_y = buttons['help_text_pos']
        draw_text_outlined(frame, f"Info: {active_help_text}", (help_x, help_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), (0,0,0), 1)

def mouse_callback(event, x, y, flags, param):
    """Handles mouse events (clicks and movement for hover)."""
    global calibration_action, hovered_button_key, last_action_key_clicked, last_action_time
    global is_auto_calibrating

    if event == cv2.EVENT_MOUSEMOVE:
        new_hover_key = None
        for key, btn in buttons.items():
            if 'rect' not in btn: continue
            bx, by, bw, bh = btn['rect']
            if bx <= x <= bx + bw and by <= y <= by + bh:
                new_hover_key = key
                break
        # Update global hover state only if it changed
        if new_hover_key != hovered_button_key:
            hovered_button_key = new_hover_key

    elif event == cv2.EVENT_LBUTTONDOWN:
        clicked_button_key = None
        for key, btn in buttons.items():
            if 'rect' not in btn: continue
            bx, by, bw, bh = btn['rect']
            if bx <= x <= bx + bw and by <= y <= by + bh:
                # Special handling for Auto-Calibrate Toggle
                if btn['action'] == 'toggle_auto_calibrate':
                    is_auto_calibrating = not is_auto_calibrating # Toggle the state directly
                    calibration_action = btn['action']
                else:
                    calibration_action = btn['action']

                last_action_key_clicked = key
                last_action_time = time.time()
                hovered_button_key = None # Reset hover state
                print(f"Button '{btn['text']}' clicked. Action pending: {calibration_action}")
                clicked_button_key = key
                break
        # If click was outside any button, clear pending action and highlight trigger
        if clicked_button_key is None:
            calibration_action = None
            last_action_key_clicked = None


# --- update_config_value (Only for Hand Calibration Now) ---
def update_config_value(config, source_name, param_name, value):
    """Updates min_input or max_input for a specific HAND source in the config."""
    global last_calibrated_message, message_display_time
    found_mapping = False
    updated = False

    for mapping in config.get('mappings', []):
        if isinstance(mapping, dict) and mapping.get('source') == source_name:
            # Basic check to avoid applying to face sources accidentally
            is_potential_face_source = source_name.startswith('head_') or source_name.endswith('_ear') or source_name == 'mar' or source_name == 'jaw_openness' or 'eyebrow' in source_name
            if is_potential_face_source:
                print(f"Info: Calibration button pressed, but source '{source_name}' looks like a face metric. Skipping UI calibration.")
                continue # Skip face sources for button calibration

            found_mapping = True
            current_val = mapping.get(param_name)

            # Check for min/max inversion and auto-correct
            other_param = 'max_input' if param_name == 'min_input' else 'min_input'
            other_val = mapping.get(other_param)

            adjustment_info = ""
            if param_name == 'min_input':
                if other_val is not None and value >= other_val:
                    mapping['max_input'] = value + calibration_epsilon
                    adjustment_info = f" (Adjusted max to {mapping['max_input']:.3f})"
                mapping[param_name] = value
                updated = True

            elif param_name == 'max_input':
                if other_val is not None and value <= other_val:
                    mapping['min_input'] = value - calibration_epsilon if value > calibration_epsilon else 0.0
                    adjustment_info = f" (Adjusted min to {mapping['min_input']:.3f})"
                mapping[param_name] = value
                updated = True

            if updated:
                final_min = mapping.get('min_input', 'N/A')
                final_max = mapping.get('max_input', 'N/A')
                try: final_min_f = f"{final_min:.3f}"
                except: final_min_f = str(final_min)
                try: final_max_f = f"{final_max:.3f}"
                except: final_max_f = str(final_max)

                msg = f"Calibrated {source_name}: Range [{final_min_f}, {final_max_f}]{adjustment_info}"
                print(msg)
                last_calibrated_message = msg
                message_display_time = time.time()

            break # Assume only one mapping per source for calibration buttons

    if not found_mapping and not is_potential_face_source: # Avoid warning for skipped face sources
        msg = f"Warning: No mapping found for source '{source_name}'. Cannot calibrate via UI."
        print(msg)
        last_calibrated_message = msg
        message_display_time = time.time()

    return updated

def apply_auto_calibration_to_config(config, calib_ranges):
    """Applies the observed min/max ranges to the config mappings."""
    global last_calibrated_message, message_display_time
    config_changed = False
    print("\n--- Applying Auto-Calibration Results ---")
    if not calib_ranges:
        print("No calibration data collected.")
        return False

    num_updated = 0
    for source_name, ranges in calib_ranges.items():
        min_val = ranges.get('min')
        max_val = ranges.get('max')

        if min_val is None or max_val is None:
            print(f" - Skipping {source_name}: Incomplete range data.")
            continue

        # Ensure min <= max, add epsilon if they are equal
        if min_val > max_val:
             min_val, max_val = max_val, min_val # Swap them
             print(f" - Info {source_name}: Min > Max observed, swapping.")
        if abs(min_val - max_val) < 1e-6: # Use float comparison
            max_val += calibration_epsilon
            print(f" - Info {source_name}: Min == Max observed, adding epsilon to max.")

        found_mapping = False
        for mapping in config.get('mappings', []):
            if isinstance(mapping, dict) and mapping.get('source') == source_name:
                # Update only if different (prevent unnecessary saves)
                current_min = mapping.get('min_input')
                current_max = mapping.get('max_input')
                if not np.isclose(current_min, min_val, atol=1e-5) or \
                   not np.isclose(current_max, max_val, atol=1e-5):
                    mapping['min_input'] = float(min_val) # Ensure float type
                    mapping['max_input'] = float(max_val) # Ensure float type
                    print(f" - Updated {source_name}: min={min_val:.3f}, max={max_val:.3f}")
                    config_changed = True
                    num_updated += 1
                else:
                    pass
                found_mapping = True
                # Don't break, update all mappings using this source if duplicated

    if config_changed:
        msg = f"Auto-calibration applied. Updated {num_updated} source range(s). Ready to save."
        print(msg)
        last_calibrated_message = msg
        message_display_time = time.time()
    else:
        msg = "Auto-calibration finished. No changes detected from previous config."
        print(msg)
        last_calibrated_message = msg
        message_display_time = time.time()


    print("----------------------------------------")
    return config_changed


# --- NEW update_auto_calibration_ranges ---
def update_auto_calibration_ranges(config, all_hands_data, face_data):
    """Updates the min/max ranges while auto-calibration is active."""
    global auto_calib_ranges

    # Use first detected hand's data for hand metrics (simplifies logic)
    first_hand_data = next((hd for hd in all_hands_data if hd.get('found')), None)

    # Combine sources for iteration
    data_sources = []
    if first_hand_data: data_sources.append(first_hand_data)
    if face_data and face_data.get('found'): data_sources.append(face_data)

    # Define sources to skip auto-calibration (usually fixed ranges or dynamic in mapper)
    skip_sources = {'centroid_x', 'centroid_y'}

    # Iterate through all mappings to know *which* sources to track
    for mapping in config.get('mappings', []):
        if not isinstance(mapping, dict): continue
        source_name = mapping.get('source')
        if not source_name or source_name in skip_sources: continue
        # Check if min/max are explicitly defined; if so, maybe skip?
        # For now, let's always track ranges during auto-calib, even if defined.
        # if mapping.get('min_input') is not None and mapping.get('max_input') is not None:
        #     continue # Skip sources with already defined ranges?

        raw_value = None
        if first_hand_data and source_name in first_hand_data:
            raw_value = first_hand_data.get(source_name)
        elif face_data and source_name in face_data:
            raw_value = face_data.get(source_name)

        if raw_value is not None:
            try:
                current_val = float(raw_value) # Ensure numeric

                if source_name not in auto_calib_ranges:
                    # First time seeing this source in this calibration run
                    auto_calib_ranges[source_name] = {'min': current_val, 'max': current_val}
                else:
                    # Update existing min/max
                    if current_val < auto_calib_ranges[source_name]['min']:
                        auto_calib_ranges[source_name]['min'] = current_val
                    if current_val > auto_calib_ranges[source_name]['max']:
                        auto_calib_ranges[source_name]['max'] = current_val
            except (ValueError, TypeError):
                 pass # Ignore non-numeric values


def main():
    global calibration_action, last_calibrated_message, message_display_time
    global last_action_key_clicked, last_action_time
    global is_auto_calibrating, auto_calib_ranges

    parser = argparse.ArgumentParser(description="CamMIDI: Control MIDI with Hand and Face Tracking.")
    parser.add_argument('--use-defaults', action='store_true', help="Force use of default configuration.")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the configuration file.")
    args = parser.parse_args()
    config_path = args.config

    # --- Initialization ---
    config = load_config(config_path=config_path, force_defaults=args.use_defaults)

    hand_tracker = HandTracker(config)
    face_tracker = FaceTracker(config) # Initialize Face Tracker
    mapper = MidiMapper(config)
    midi_sender = MidiSender(config)

    # --- Camera Setup ---
    cam_idx = config.get('camera', {}).get('index', 0)
    req_w = config.get('camera', {}).get('width', 800)
    req_h = config.get('camera', {}).get('height', 460)
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Error: Could not open camera index {cam_idx}.")
        # Ensure cleanup even if camera fails
        if midi_sender: midi_sender.close()
        if hand_tracker: hand_tracker.close()
        if face_tracker: face_tracker.close()
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened. Requested: {req_w}x{req_h}, Actual: {actual_width}x{actual_height}")

    display_enabled = config.get('display', {}).get('show_window', True)
    if display_enabled:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        define_buttons(actual_width, actual_height)
        cv2.setMouseCallback(window_name, mouse_callback)

    pTime = 0

    # --- Startup Info --- (Keep as is)
    print("\nStarting CamMIDI loop. Press 'Q' in the output window to quit.")
    print("Use UI buttons OR 'Auto-Calibrate' to set min/max ranges.")
    print("Manually edit config.yaml for fine-tuning or unsupported metrics.")
    print("Press 'Save Config' button or 's' key to save calibrated values.")
    print("MIDI Output Port:", config.get('midi', {}).get('port_name', 'N/A'))
    max_h = config.get('mediapipe', {}).get('max_num_hands', 1)
    max_f = config.get('face_mesh', {}).get('max_num_faces', 1)
    force_ch1 = config.get('midi', {}).get('force_channel1_hand', 'None')
    print(f"Tracking up to {max_h} hands and {max_f} face(s).")
    print(f" Hand Channel 1 Assignment: {force_ch1} preference.")
    if max_h > 1: print(" Implicit Hand Ch2 mapping: ON.")
    print("---")

    # --- Main Loop ---
    try:
        action_display_start_time = 0
        last_action_key_for_draw = None

        while True:
            success, frame = cap.read()
            if not success or frame is None:
                 print("Warning: Failed to capture frame.")
                 time.sleep(0.1)
                 continue

            # 1. Track Hands and Face
            # Pass frame.copy() to avoid trackers modifying the same frame buffer if they drew on it
            hand_processed_frame, all_hands_data = hand_tracker.process_frame(frame.copy())
            face_processed_frame, face_data = face_tracker.process_frame(frame.copy())
            processed_frame = face_processed_frame # Use face frame as base for drawing

            # --- Auto-Calibration Mode Logic ---
            if is_auto_calibrating:
                update_auto_calibration_ranges(config, all_hands_data, face_data)

            # --- Handle Button Clicks / Actions ---
            current_action = None
            if calibration_action: # Check if mouse callback set an action
                current_action = calibration_action
                if current_action != 'toggle_auto_calibrate':
                    last_action_key_for_draw = current_action # Used for non-persistent highlight
                    action_display_start_time = time.time()
                calibration_action = None # Reset flag immediately

                config_updated_by_action = False

                # --- Toggle Auto-Calibrate Action ---
                if current_action == 'toggle_auto_calibrate':
                    if not is_auto_calibrating: # State was toggled OFF by mouse click
                         print("Stopping Auto-Calibration.")
                         config_updated_by_action = apply_auto_calibration_to_config(config, auto_calib_ranges)
                    else: # State was toggled ON by mouse click
                         print("Starting Auto-Calibration... Move through desired ranges!")
                         auto_calib_ranges.clear() # Clear previous results
                         last_calibrated_message = "AUTO-CALIBRATING... Move freely!"
                         message_display_time = time.time()

                    # Redefine buttons to update text ("Start"/"Stop")
                    if display_enabled: define_buttons(actual_width, actual_height)

                # --- Handle Manual Calibration Actions ---
                elif not is_auto_calibrating: # Only allow manual if NOT auto-calibrating
                    first_hand_data = next((hd for hd in all_hands_data if hd.get('found')), None)
                    if first_hand_data is None and current_action != 'save_config':
                        msg = "Manual Calibration failed: No hand detected."
                        print(msg)
                        last_calibrated_message = msg
                        message_display_time = time.time()
                    else:
                        # Perform HAND Calibration based on current_action
                        if current_action == 'calib_spread_min':
                             config_updated_by_action |= update_config_value(config, 'index_middle_spread', 'min_input', first_hand_data.get('index_middle_spread', 0.0))
                             config_updated_by_action |= update_config_value(config, 'middle_ring_spread', 'min_input', first_hand_data.get('middle_ring_spread', 0.0))
                             config_updated_by_action |= update_config_value(config, 'ring_pinky_spread', 'min_input', first_hand_data.get('ring_pinky_spread', 0.0))
                        elif current_action == 'calib_spread_max':
                             config_updated_by_action |= update_config_value(config, 'index_middle_spread', 'max_input', first_hand_data.get('index_middle_spread', np.pi/4))
                             config_updated_by_action |= update_config_value(config, 'middle_ring_spread', 'max_input', first_hand_data.get('middle_ring_spread', np.pi/5))
                             config_updated_by_action |= update_config_value(config, 'ring_pinky_spread', 'max_input', first_hand_data.get('ring_pinky_spread', np.pi/5))
                        elif current_action == 'calib_ti_pinch_min': config_updated_by_action |= update_config_value(config, 'thumb_index_pinch', 'min_input', first_hand_data.get('thumb_index_pinch', 0.01))
                        elif current_action == 'calib_ti_pinch_max': config_updated_by_action |= update_config_value(config, 'thumb_index_pinch', 'max_input', first_hand_data.get('thumb_index_pinch', 0.3))
                        elif current_action == 'calib_tm_pinch_min': config_updated_by_action |= update_config_value(config, 'thumb_middle_pinch', 'min_input', first_hand_data.get('thumb_middle_pinch', 0.02))
                        elif current_action == 'calib_tm_pinch_max': config_updated_by_action |= update_config_value(config, 'thumb_middle_pinch', 'max_input', first_hand_data.get('thumb_middle_pinch', 0.35))
                        elif current_action == 'calib_tr_pinch_min': config_updated_by_action |= update_config_value(config, 'thumb_ring_pinch', 'min_input', first_hand_data.get('thumb_ring_pinch', 0.03))
                        elif current_action == 'calib_tr_pinch_max': config_updated_by_action |= update_config_value(config, 'thumb_ring_pinch', 'max_input', first_hand_data.get('thumb_ring_pinch', 0.4))
                        elif current_action == 'calib_tp_pinch_min': config_updated_by_action |= update_config_value(config, 'thumb_pinky_pinch', 'min_input', first_hand_data.get('thumb_pinky_pinch', 0.05))
                        elif current_action == 'calib_tp_pinch_max': config_updated_by_action |= update_config_value(config, 'thumb_pinky_pinch', 'max_input', first_hand_data.get('thumb_pinky_pinch', 0.45))
                        elif current_action == 'save_config':
                             save_config(config, config_path)
                             last_calibrated_message = f"Configuration saved to {config_path}"
                             message_display_time = time.time()
                             config_updated_by_action = False # No mapper update needed for save
                else:
                    # User clicked a manual calibration button while auto-calibrating
                    msg = "Please stop Auto-Calibration before manual adjustments."
                    print(msg)
                    last_calibrated_message = msg
                    message_display_time = time.time()

                # Update mapper if config values (min/max input) changed from ANY action
                if config_updated_by_action:
                    print("Config updated by action, refreshing mapper.")
                    mapper.update_mappings(config.get('mappings', []))

            # Make button highlight brief
            if last_action_key_for_draw and (time.time() - action_display_start_time > 0.15): # Short highlight
                 last_action_key_for_draw = None

            # 2. Map MIDI
            midi_map = mapper.calculate_midi(all_hands_data, face_data, actual_width, actual_height)

            # 3. Send MIDI
            if midi_sender and midi_sender.port_opened:
                 for (ch, cc), data in midi_map.items():
                      val = data['value']
                      midi_sender.send_cc(ch, cc, val)

            # 4. Visual Feedback
            if display_enabled:
                display_frame = processed_frame
                # Draw hand visuals on top
                display_frame = hand_tracker.draw_visuals(display_frame, all_hands_data)
                # Draw face visuals (ensure it's called after potentially flipped frame)
                display_frame = face_tracker.draw_visuals(display_frame, face_data)

                # FPS calculation
                cTime = time.time(); delta_time = cTime - pTime
                fps = 1 / delta_time if delta_time > 0 else 0; pTime = cTime
                if config.get('display', {}).get('show_fps', True):
                    draw_text_outlined(display_frame, f"FPS: {int(fps)}", (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), (0,0,0), 2)


                # --- MIDI Debug Display --- (Keep as is)
                messages_ch1, messages_ch2, messages_ch3_plus = [], [], []
                source_lookup = { (m['channel'], m['cc']): m['source']
                                  for m in config.get('mappings', []) if isinstance(m, dict) and m.get('source')}
                for (ch, cc), data in midi_map.items():
                    val = data['value']
                    source_type = data['source_type']
                    source_index = data['source_index'] # 0 for face, 0 or 1 for hand
                    source_name = source_lookup.get((ch, cc), f"Src?({ch},{cc})")
                    label = ""
                    if source_type == 'hand':
                        handedness = "H?"
                        if 0 <= source_index < len(all_hands_data): handedness = all_hands_data[source_index].get('handedness', 'H?')[0]
                        label = f"H{source_index+1}({handedness})"
                    elif source_type == 'face': label = "Face"
                    text = f"{label} {source_name} -> Ch{ch} CC{cc}: {val}"
                    if ch == 1: messages_ch1.append(text)
                    elif ch == 2: messages_ch2.append(text)
                    else: messages_ch3_plus.append(f"Ch{ch}: {text}")
                y_offset_ch1, y_offset_ch2, y_offset_ch3 = 60, 60, 60 + (len(messages_ch2) * 18) + 10
                font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                fg_col, bg_col = (255, 255, 255), (0, 0, 0)
                margin = 10
                for text in messages_ch1:
                    (tw, th), _ = cv2.getTextSize(text, font, scale, thick); tx = max(0, actual_width - tw - margin)
                    draw_text_outlined(display_frame, text, (tx, y_offset_ch1), font, scale, fg_col, bg_col, thick); y_offset_ch1 += 18
                for text in messages_ch2: draw_text_outlined(display_frame, text, (margin, y_offset_ch2), font, scale, fg_col, bg_col, thick); y_offset_ch2 += 18
                if not messages_ch2: y_offset_ch3 = 60 # Adjust if no Ch2 messages
                for text in messages_ch3_plus: draw_text_outlined(display_frame, text, (margin, y_offset_ch3), font, scale, fg_col, bg_col, thick); y_offset_ch3 += 18

                # --- Draw Buttons & Help Text ---
                draw_buttons(display_frame, highlight_duration=0.15) # Use the function defined earlier

                # --- Draw Calibration Status Message / Auto-Calibrate Indicator ---
                font_status, scale_status, thick_status = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                status_message = ""
                status_color = (0, 255, 255) # Default cyan

                if is_auto_calibrating:
                     status_message = "AUTO-CALIBRATING... Move through ranges! Click button again to STOP."
                     status_color = (255, 150, 0) # Orange/Yellow
                elif last_calibrated_message and (time.time() - message_display_time < 4.0): # Show longer
                    status_message = last_calibrated_message
                elif last_calibrated_message and (time.time() - message_display_time >= 4.0):
                    last_calibrated_message = "" # Clear message after timeout

                if status_message:
                     (tw, th), _ = cv2.getTextSize(status_message, font_status, scale_status, thick_status)
                     tx = (actual_width - tw) // 2
                     # Position below buttons or debug text
                     max_button_y = 0
                     for btn_key, btn_data in buttons.items():
                         if 'rect' in btn_data: max_button_y = max(max_button_y, btn_data['rect'][1] + btn_data['rect'][3])
                     ty = max(y_offset_ch3, max_button_y) + 25 # Adjusted position

                     draw_text_outlined(display_frame, status_message, (tx, ty), font_status, scale_status, status_color, bg_col, thick_status)


                # --- Show Frame ---
                cv2.imshow(window_name, display_frame)

            # Exit Check & Save Shortcut
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('s'): # Keyboard shortcut to save config
                 if is_auto_calibrating:
                      print("Cannot save while Auto-Calibrating. Stop calibration first.")
                      last_calibrated_message = "Stop Auto-Calibration before saving!"
                      message_display_time = time.time()
                 else:
                      print("Save triggered by keyboard ('s')")
                      save_config(config, config_path)
                      last_calibrated_message = f"Config saved to {config_path} (Key 's')"
                      message_display_time = time.time()

    except KeyboardInterrupt: print("\nKeyboard interrupt received.")
    finally:
        # Cleanup
        print("Shutting down...")
        if cap and cap.isOpened(): cap.release()
        if display_enabled: cv2.destroyAllWindows()
        if hand_tracker: hand_tracker.close()
        if face_tracker: face_tracker.close()
        if midi_sender: midi_sender.close()
        print("CamMIDI stopped.")

if __name__ == "__main__":
    main()