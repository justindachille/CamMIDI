# CamMIDI: Real-time Hand and Face Tracking for Expressive MIDI Control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![CamMIDI Demo showing hand and face tracking controlling MIDI parameters](assets/looped.gif)

## Overview

CamMIDI leverages your webcam to translate real-time hand gestures and facial expressions into MIDI Control Change (CC) messages. It provides an intuitive and highly customizable way for musicians, artists, and developers to create expressive control surfaces without specialized hardware. Turn subtle movements like finger pinches, head tilts, or eyebrow raises into powerful modulation sources for synthesizers, DAWs, VJ software, game engines, and more.

## Core Features

*   **Multi-Modal Tracking:** Simultaneously tracks hands (up to 2) and faces using Google's MediaPipe framework.
*   **Rich Data Extraction:** Captures a wide range of metrics:
    *   **Hands:** 3D position (X/Y/Z), orientation (Pitch/Yaw/Roll), individual finger curls, finger spreads, and pinch distances.
    *   **Face:** 3D head pose (Pitch/Yaw/Roll), Eye Aspect Ratio (EAR) for blinks, Mouth Aspect Ratio (MAR), jaw openness, and eyebrow height.
*   **Flexible MIDI Mapping:** Easily configure which gesture/expression controls which MIDI CC message on specific channels via a simple `config.yaml` file. Supports inversion and input range scaling.
*   **Real-time Performance:** Optimized for low-latency processing to ensure responsive control.
*   **Intuitive Calibration:**
    *   **Manual:** Fine-tune specific hand pinch/spread ranges using interactive GUI buttons.
    *   **Auto-Calibrate:** Dynamically detects the minimum and maximum values for all mapped parameters as you move, allowing for rapid setup.
*   **Visual Feedback:** An OpenCV window displays the camera feed, tracked landmarks, active MIDI mappings, and calibration controls.
*   **Cross-Platform:** Runs on Windows, macOS, and Linux (requires appropriate Python and MIDI setup).

## Technology Stack

*   **Python 3.x**
*   **OpenCV:** Camera interaction, image processing, GUI elements.
*   **MediaPipe:** Core machine learning models for hand and face landmark detection.
*   **python-rtmidi:** Cross-platform real-time MIDI output.
*   **PyYAML:** Loading and saving user configuration.
*   **NumPy:** Efficient numerical calculations for vector math and scaling.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/justindachille/CamMIDI.git
    cd CamMIDI
    ```
2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS / Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up a Virtual MIDI Port:** CamMIDI needs a software MIDI port to send data to other applications.
    *   **macOS:** Use the built-in "IAC Driver". Open "Audio MIDI Setup", go to the MIDI Studio window (Window > Show MIDI Studio), double-click "IAC Driver", and ensure "Device is online" is checked. Note the port name (usually "IAC Driver Bus 1").
    *   **Windows:** Install a virtual MIDI loopback driver like [loopMIDI](https://www.tobias-erichsen.de/software/loopmidi.html). Create a port and note its name.
    *   **Linux:** Use ALSA's `snd-virmidi` or Jack MIDI. Configuration varies.
5.  **Configure `config.yaml`:** Open `config.yaml` and set `midi.port_name` to the exact name of your virtual MIDI port created in the previous step. Adjust camera index or resolution if needed.

## Usage

1.  **Connect:** Ensure your target application (DAW, synth, VJ software) is set up to receive MIDI input from the virtual MIDI port you configured.
2.  **Run:** Execute the main script from your terminal:
    ```bash
    python main.py
    ```
    *   Use `python main.py --use-defaults` to ignore `config.yaml` and run with default settings.
    *   Use `python main.py --config path/to/your_config.yaml` to specify a different config file.
3.  **Calibrate:**
    *   Use the **"Auto-Calibrate ALL"** button: Click it, perform the full range of desired motions for *all* mapped controls for several seconds, then click "Stop Calibrating".
    *   Alternatively, use the **manual buttons** (e.g., "Set T-Idx MIN/MAX") for fine-tuning specific hand gestures.
4.  **Save:** Click the **"Save Config"** button (or press 's') to store your calibrated ranges in `config.yaml` for future use.
5.  **Control:** Your movements should now be sending MIDI CC messages according to your mappings.
6.  **Quit:** Press 'q' in the output window.

## Configuration (`config.yaml`)

The `config.yaml` file allows extensive customization:

*   `camera`: Select camera index and resolution.
*   `mediapipe` / `face_mesh`: Adjust tracking model sensitivity and parameters.
*   `midi`: Set the output port name, smoothing factor, and hand channel assignment logic.
*   `display`: Toggle visual elements like landmarks, connections, FPS counter.
*   `mappings`: **The core configuration.** Define rules for mapping tracking `source` data (e.g., `index_angle_curl`, `head_yaw`) to MIDI `channel` and `cc` numbers. Specify `min_input`, `max_input` (crucial for scaling), and optional `invert`.

*Calibration is essential for mapping raw tracking values to the 0-127 MIDI range effectively.*

## Future Enhancements

*   Support for MIDI Note On/Off and Pitch Bend messages.
*   More sophisticated gesture recognition (e.g., static poses, dynamic gestures).
*   Configuration validation using schemas.
*   Executable packaging for easier distribution.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
