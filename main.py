import cv2
import time
import numpy as np

from config_loader import load_config
from tracker import HandTracker
from mapper import MidiMapper
from midi_sender import MidiSender

def main():
    # --- Initialization ---
    config = load_config()
    tracker = HandTracker(config)
    mapper = MidiMapper(config)
    midi_sender = MidiSender(config)

    # Camera Setup
    cap = cv2.VideoCapture(config['camera']['index'])
    if not cap.isOpened():
        print(f"Error: Could not open camera index {config['camera']['index']}.")
        midi_sender.close() # Close MIDI port if camera fails
        tracker.close()
        return

    # Attempt to set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['camera']['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera']['height'])
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened. Requested: {config['camera']['width']}x{config['camera']['height']}, Actual: {actual_width}x{actual_height}")


    if config['display']['show_window']:
        cv2.namedWindow("CamMIDI Output", cv2.WINDOW_NORMAL)
        # Resize based on actual capture size initially
        cv2.resizeWindow("CamMIDI Output", actual_width, actual_height)


    pTime = 0 # Previous time for FPS calculation

    print("\nStarting CamMIDI loop. Press 'Q' in the output window to quit.")
    print("Ensure your DAW is configured to receive MIDI from:", config['midi']['port_name'])
    print("---")
    print("CALIBRATION NOTE: For distance/angle sources (curl, pinch, orientation, z),")
    print("you MUST adjust 'min_input' and 'max_input' in config.yaml for your setup!")
    print("---")


    # --- Main Loop ---
    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                print("Error: Failed to grab frame from camera.")
                time.sleep(0.5) # Avoid busy-looping on error
                continue

            loop_start_time = time.time() # For detailed timing if needed

            # 1. Track Hands & Calculate Metrics
            processed_frame, tracking_data = tracker.process_frame(frame)

            # 2. Map Tracking Data to MIDI
            midi_map = mapper.calculate_midi(tracking_data, actual_width, actual_height)

            # 3. Send MIDI Messages
            if midi_sender.port_opened:
                 for (ch, cc), val in midi_map.items():
                      midi_sender.send_cc(ch, cc, val)

            # 4. Visual Feedback
            if config['display']['show_window']:
                display_frame = processed_frame.copy() # Work on a copy

                # Draw tracking visuals (landmarks, etc.)
                display_frame = tracker.draw_visuals(display_frame, tracking_data)

                # Calculate and display FPS
                cTime = time.time()
                fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
                pTime = cTime

                if config['display']['show_fps']:
                    cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Display mapped MIDI values on screen
                y_offset = 60
                for (ch, cc), val in midi_map.items():
                     # Find corresponding mapping details for display
                     label = f"Ch{ch} CC{cc}"
                     source_name = "Unknown"
                     for m in config.get('mappings', []):
                         if m['channel'] == ch and m['cc'] == cc:
                              source_name = m['source']
                              label = f"{source_name} -> Ch{ch} CC{cc}"
                              break # Simple: take the first match if multiple map to same CC

                     text = f"{label}: {val}"
                     cv2.putText(display_frame, text, (10, y_offset),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1) # Smaller font
                     y_offset += 18 # Smaller spacing


                cv2.imshow("CamMIDI Output", display_frame)

            # --- Check for Exit Key ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exit key pressed.")
                break

    except KeyboardInterrupt:
         print("Keyboard interrupt received.")
    finally:
        # --- Cleanup ---
        print("Shutting down...")
        cap.release()
        if config['display']['show_window']:
            cv2.destroyAllWindows()
        tracker.close()
        midi_sender.close()
        print("CamMIDI stopped.")


if __name__ == "__main__":
    main()