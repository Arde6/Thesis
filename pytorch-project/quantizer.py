"""
quantizer.py
------------

Quantizes tensorflow models and saves them as tflite.
Can do fp32, fp16 and int8 dynamic and static 
"""

import tensorflow as tf
import h5py
import numpy as np
import argparse
import os

def make_representative_dataset(h5_path: str, num_samples: int = 100):
    """
    Builds a representative dataset generator from a processed .h5 file.
    Scaler is already baked into the model, so raw scaled features are used as-is.
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Representative dataset file not found: {h5_path}")
        
    with h5py.File(h5_path, 'r') as f:
        features = f['features'][:num_samples]  # shape: (N, feature_dim)

    features = features.astype(np.float32)

    def representative_data_gen():
        for sample in features:
            yield [np.expand_dims(sample, axis=0)]  # shape: (1, feature_dim)

    return representative_data_gen

def main():
    parser = argparse.ArgumentParser(description="Quantize a TensorFlow SavedModel to TFLite.")
    parser.add_argument("--input_model", type=str, default="models/saved_model", help="Path to the SavedModel directory.")
    parser.add_argument("--output_tflite", type=str, required=True, help="Path where the output .tflite model will be saved.")
    parser.add_argument("--data_h5", type=str, default="processed_data.h5", help="Path to the .h5 file for representative dataset (required for int8_static).")
    parser.add_argument("--mode", type=str, choices=["fp32", "fp16", "int8_dynamic", "int8_static"], required=True, 
                        help="Quantization mode: fp32 (none), fp16, int8_dynamic, or int8_static.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to use for representative dataset.")

    args = parser.parse_args()

    print(f"Loading model from: {args.input_model}")
    converter = tf.lite.TFLiteConverter.from_saved_model(args.input_model)

    if args.mode == "fp32":
        print("Converting to TFLite (FP32, no quantization)...")
        # No extra settings needed for FP32
        pass

    elif args.mode == "fp16":
        print("Converting to TFLite (FP16 quantization)...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    elif args.mode == "int8_dynamic":
        print("Converting to TFLite (INT8 Dynamic Range quantization)...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    elif args.mode == "int8_static":
        print("Converting to TFLite (INT8 Static quantization)...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = make_representative_dataset(args.data_h5, args.num_samples)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_tflite)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_tflite, "wb") as f:
        f.write(tflite_model)
    
    print(f"Successfully saved TFLite model to: {args.output_tflite}")

if __name__ == "__main__":
    main()
