import rtmidi
import time

class MidiSender:
    def __init__(self, config):
        """Initializes the MIDI output."""
        self.port_name = config['midi']['port_name']
        self.midi_out = rtmidi.MidiOut()
        self.available_ports = self.midi_out.get_ports()
        self.port_opened = False
        self._last_sent_values = {} # Store {(channel, cc): value}

        print("Available MIDI output ports:", self.available_ports)

        try:
            port_index = -1
            for i, name in enumerate(self.available_ports):
                 # Simple substring matching, might need refinement
                 if self.port_name in name:
                      port_index = i
                      break

            if port_index != -1:
                 self.midi_out.open_port(port_index)
                 self.port_opened = True
                 print(f"Successfully opened MIDI port: {self.available_ports[port_index]}")
            else:
                 print(f"Error: MIDI port '{self.port_name}' not found.")
                 print("Please ensure a virtual MIDI port (like IAC Driver on Mac or loopMIDI on Windows) is running and configured.")
                 # Fallback: Try opening the first available port if any exist
                 if self.available_ports:
                     try:
                        print(f"Attempting to open first available port: {self.available_ports[0]}")
                        self.midi_out.open_port(0)
                        self.port_opened = True
                        print(f"Successfully opened MIDI port: {self.available_ports[0]}")
                        print(f"Update config.yaml with 'port_name: \"{self.available_ports[0]}\"' for future use.")
                     except Exception as e_fallback:
                         print(f"Error opening fallback port: {e_fallback}")
                 else:
                      print("No MIDI output ports available.")


        except Exception as e:
            print(f"Error initializing MIDI: {e}")


    def send_cc(self, channel, cc_number, value):
        """Sends a MIDI Control Change message if the value has changed."""
        if not self.port_opened:
            # print("Warning: MIDI port not open. Cannot send message.")
            return

        # Validate MIDI values
        channel = int(channel)
        cc_number = int(cc_number)
        value = int(round(value)) # Ensure integer

        if not (1 <= channel <= 16):
            print(f"Warning: Invalid MIDI channel {channel}. Must be 1-16.")
            return
        if not (0 <= cc_number <= 127):
            print(f"Warning: Invalid MIDI CC number {cc_number}. Must be 0-127.")
            return
        if not (0 <= value <= 127):
             value = max(0, min(127, value)) # Clamp value to valid range
            # print(f"Warning: Clamping MIDI value {value} to 0-127 range.")
            # return # Option: skip sending if value needs clamping, or just clamp and send


        message_key = (channel, cc_number)
        last_value = self._last_sent_values.get(message_key, -1) # Use -1 to ensure first message sends

        if value != last_value:
            # MIDI CC message: 0xBn (Control Change on channel n), controller number, value
            # Channel in rtmidi is 0-indexed, config is 1-indexed
            status_byte = 0xB0 | (channel - 1)
            message = [status_byte, cc_number, value]
            try:
                self.midi_out.send_message(message)
                self._last_sent_values[message_key] = value
                # print(f"Sent MIDI: Ch={channel}, CC={cc_number}, Val={value}") # Debugging
            except Exception as e:
                print(f"Error sending MIDI message: {e}")


    def close(self):
        """Closes the MIDI port."""
        if self.midi_out.is_port_open():
            # Optional: Send CC 0 (all notes off) or reset specific CCs to 0/64?
            # for (ch, cc), _ in self._last_sent_values.items():
            #     reset_val = 0 # Or 64 for things like pan/volume
            #     status_byte = 0xB0 | (ch - 1)
            #     message = [status_byte, cc, reset_val]
            #     self.midi_out.send_message(message)
            #     time.sleep(0.001) # Small delay
            self.midi_out.close_port()
            self.port_opened = False
            print("MIDI port closed.")
        del self.midi_out # Release rtmidi object