Keyword Detection Example - "left", "right", "go" Commands
===========================================================

This example demonstrates real-time keyword spotting for navigation
commands using a custom-trained TensorFlow Lite model.

Keywords:
  - left  : Turn robot left
  - right : Turn robot right
  - go    : Move forward
  - unknown : Other words
  - silence : No speech detected

Model Requirements:
-------------------
Place your trained Edge TPU model at:
  models/model_int8_edgetpu.tflite

To train the model, see: train_keyword_complete.ipynb

Building:
---------
From coralmicro root directory:

  bash build.sh

The binary will be at:
  build/examples/classify_keywords/classify_keywords

Flashing to Board:
------------------
1. Connect Coral Dev Board Micro via USB
2. Put board in bootloader mode (hold BOOT, press RESET)
3. Flash:

  python3 scripts/flashtool.py \
    --build_dir build \
    --elf_path build/examples/classify_keywords/classify_keywords

Usage:
------
1. The board will start listening immediately
2. Speak keywords clearly into the microphone
3. Detection prints:
   - Detected keyword
   - Confidence percentage
   - All class scores
   - Inference timing

4. Serial console output (115200 baud):

  ========================================
  Keyword Detector: left, right, go
  ========================================

  Model loaded successfully (81920 bytes)
  Edge TPU initialized
  Tensors allocated

  Model Info:
    Input shape: [1, 1960]
    Input type: INT8
    Output shape: [1, 5]
    Output type: INT8
    Expected input size: 1960 (49 time steps x 40 mel bins)
    Number of classes: 5

  Audio frontend initialized
  Audio service started

  Listening for keywords: left, right, go...
  Speak clearly into the microphone.

  ==========================================
  DETECTED: left (confidence: 87.45%)
  ==========================================
  All scores:
    left: 87.45%
    right: 5.49%
    go: 3.92%
    unknown: 2.35%
    silence: 0.78%

  Preprocess: 15ms, Inference: 8ms, Total: 23ms

Configuration:
--------------
Edit classify_keywords.cc to adjust:
  - kThreshold : Detection confidence threshold (default: 0.5)
  - kTopK      : Number of results to show (default: 3)
  - kModelName : Path to model file

Audio Processing:
-----------------
- Sample rate: 16 kHz
- Window size: 30ms
- Window stride: 20ms
- Feature bins: 40 (mel-frequency)
- Duration: 1 second per inference
- Input: 49 time steps x 40 freq bins = 1960 features

Performance:
------------
Expected timing on Coral Dev Board Micro:
  - Preprocessing: ~15ms (STFT + Mel conversion)
  - Inference: ~8ms (Edge TPU accelerated)
  - Total: ~23ms per classification

Tips:
-----
1. Speak clearly and at normal volume
2. Position microphone 10-30cm from mouth
3. Reduce background noise for better accuracy
4. Wait 1 second between commands (inference rate limit)
5. Confidence threshold of 50% filters most false positives

Troubleshooting:
----------------
Q: Model not found error
A: Copy model_int8_edgetpu.tflite to models/ directory

Q: Low accuracy
A: Retrain model with more epochs (30-40)
   Check microphone positioning
   Reduce background noise

Q: Edge TPU initialization failed
A: Ensure board has Edge TPU support
   Check USB connection

Q: Wrong input size error
A: Model expects 1960 features (49x40)
   Verify spectrogram parameters match training

Integration:
------------
To use detected keywords in your application:

1. Modify the run() function to call your control code:

   if (probability > kThreshold && max_index >= 0 && max_index <= 2) {
     switch (max_index) {
       case 0:  // left
         robot.turnLeft();
         break;
       case 1:  // right
         robot.turnRight();
         break;
       case 2:  // go
         robot.moveForward();
         break;
     }
   }

2. Add debouncing to prevent repeated commands:

   static uint32_t last_command_time = 0;
   uint32_t current_time = TimerMillis();
   if (current_time - last_command_time > 2000) {
     // Execute command
     last_command_time = current_time;
   }

License:
--------
Copyright 2022 Google LLC
Licensed under Apache License 2.0
See LICENSE file for details
