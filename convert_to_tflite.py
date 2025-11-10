#!/usr/bin/env python3
"""
Standalone script to convert Keras model to TFLite.
This runs in a separate process to avoid kernel crashes.
"""
import os
import sys
import tensorflow as tf
import gc

def convert_to_float_tflite(model_path, output_path):
    """Convert Keras model to Float32 TFLite"""
    print("=" * 60)
    print("Converting to Float32 TFLite model")
    print("=" * 60)

    # Load the model
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Convert
    print("Converting...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"\n✓ Float32 model saved: {output_path}")
    print(f"✓ Size: {size_kb:.2f} KB")
    print("=" * 60)

    return size_kb

def convert_to_int8_tflite(model_path, dataset_path, output_path):
    """Convert Keras model to INT8 TFLite with quantization"""
    print("\n" + "=" * 60)
    print("Converting to INT8 quantized TFLite model")
    print("=" * 60)

    # Load the model
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Load calibration data
    print(f"Loading calibration data from: {dataset_path}")
    import numpy as np
    calibration_data = np.load(dataset_path)
    print(f"Calibration data shape: {calibration_data.shape}")
    print(f"Using {len(calibration_data)} samples for calibration")

    # Representative dataset generator (simpler approach)
    def representative_dataset_gen():
        for i in range(len(calibration_data)):
            if i % 10 == 0:
                print(f"  Calibration sample {i}/{len(calibration_data)}")
            # Yield single sample expanded to batch dimension
            yield [tf.expand_dims(calibration_data[i], axis=0)]

    # Convert with optimizations (dynamic range quantization)
    print("\nSetting up quantization...")
    print("Using dynamic range quantization (Edge TPU Compiler will perform full INT8)")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen

    print("\nConverting model (this may take 1-2 minutes)...")
    tflite_quant_model = converter.convert()

    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_quant_model)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"\n✓ INT8 model saved: {output_path}")
    print(f"✓ Size: {size_kb:.2f} KB")
    print("=" * 60)

    return size_kb

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Float32: python convert_to_tflite.py float <model.keras> <output.tflite>")
        print("  INT8:    python convert_to_tflite.py int8 <model.keras> <calibration.npy> <output.tflite>")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "float":
        model_path = sys.argv[2]
        output_path = sys.argv[3]
        convert_to_float_tflite(model_path, output_path)
    elif mode == "int8":
        model_path = sys.argv[2]
        dataset_path = sys.argv[3]
        output_path = sys.argv[4]
        convert_to_int8_tflite(model_path, dataset_path, output_path)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
