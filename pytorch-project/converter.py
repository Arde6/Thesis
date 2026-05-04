"""
convert_to_tf.py
----------------
Converts a PyTorch feedforward neural network (.pth) to a TensorFlow SavedModel.

Strategy: PyTorch → ONNX → TensorFlow SavedModel (via onnx2tf)
- Requires Python 3.12
- No manual architecture reconstruction needed — the full graph is captured
  from a trace, BatchNorm and all.
- Optionally embeds a z-score scaler as a preprocessing step in the graph.
"""

import os
import glob
import re
import tempfile

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnx2tf
import tensorflow as tf


# ---------------------------------------------------------------------------
# 1.  Scaler loading
# ---------------------------------------------------------------------------

def load_scaler(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load mean and scale arrays from a .npz or HDF5 file.
    Returns (mean, scale) as float32 numpy arrays.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler file not found: {path}")

    if path.endswith(".npz"):
        data = np.load(path)
        return data["mean"].astype(np.float32), data["scale"].astype(np.float32)

    try:
        import h5py
        with h5py.File(path, "r") as f:
            if "scaler_mean" in f and "scaler_scale" in f:
                return (
                    f["scaler_mean"][:].astype(np.float32),
                    f["scaler_scale"][:].astype(np.float32),
                )
    except Exception:
        pass

    raise RuntimeError(
        f"Unsupported scaler file or missing keys in: {path}\n"
        "Expected a .npz with keys 'mean'/'scale', or an HDF5 file "
        "with datasets 'scaler_mean'/'scaler_scale'."
    )


# ---------------------------------------------------------------------------
# 2.  Architecture inference
# ---------------------------------------------------------------------------

def infer_arch_from_state_dict(state_dict: dict) -> tuple[int, list[int], int]:
    """
    Infer (input_size, hidden_sizes, output_size) from Linear weight shapes.
    Only 2-D tensors are considered (BatchNorm weights are 1-D and skipped).
    """
    linear_keys: list[tuple[int, str]] = []

    for i, (k, v) in enumerate(state_dict.items()):
        if not isinstance(v, torch.Tensor) or v.ndim != 2:
            continue
        m = re.search(r"(\d+)(?=\.weight$)", k)
        idx = int(m.group(1)) if m else i
        linear_keys.append((idx, k))

    if not linear_keys:
        raise RuntimeError(
            "No 2-D weight tensors found. Check that the .pth file contains "
            "a valid state dict or a dict with a 'model_state_dict' key."
        )

    linear_keys.sort(key=lambda x: x[0])
    shapes = [tuple(state_dict[k].shape) for _, k in linear_keys]

    input_size   = int(shapes[0][1])
    hidden_sizes = [int(s[0]) for s in shapes[:-1]]
    output_size  = int(shapes[-1][0])
    return input_size, hidden_sizes, output_size


# ---------------------------------------------------------------------------
# 3.  PyTorch scaler wrapper
# ---------------------------------------------------------------------------

class ScalerWrapper(nn.Module):
    """
    Wraps any nn.Module and prepends z-score normalisation.
    Embedding the scaler here means it gets captured in the ONNX graph,
    so the SavedModel accepts raw (unscaled) inputs.
    """

    def __init__(self, model: nn.Module, mean: np.ndarray, scale: np.ndarray):
        super().__init__()
        self.register_buffer("mean",  torch.tensor(mean,  dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.scale)


# ---------------------------------------------------------------------------
# 4.  Load checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(model_path: str) -> tuple[nn.Module, int]:
    """
    Load the checkpoint and return the eval-mode model and input size.

    Supports three checkpoint formats:
      - Full model object saved directly with torch.save(model, ...)
      - Dict with a 'model' key containing the full model object
      - State dict (bare or under 'model_state_dict') — requires
        neural_net_onecyclelr.NeuralNetwork to be importable
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # Case 1: checkpoint is the full model object
    if isinstance(checkpoint, nn.Module):
        model = checkpoint.eval()

    # Case 2: dict containing the full model object
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        model = checkpoint["model"].eval()

    # Case 3: state dict only — reconstruct from local NeuralNetwork class
    else:
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        from lib.neural_net_onecyclelr import NeuralNetwork
        input_size, hidden_sizes, output_size = infer_arch_from_state_dict(state_dict)
        model = NeuralNetwork(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
        )
        model.load_state_dict(state_dict)
        model.eval()

    input_size, _, _ = infer_arch_from_state_dict(model.state_dict())
    print(
        f"Checkpoint loaded  |  input_size={input_size}  |  "
        f"BatchNorm={'yes' if any('running_mean' in k for k in model.state_dict()) else 'no'}"
    )
    return model, input_size


# ---------------------------------------------------------------------------
# 5.  Export to ONNX
# ---------------------------------------------------------------------------

def export_onnx(model: nn.Module, input_size: int, onnx_path: str) -> None:
    """
    Trace the model with a dummy input and export to ONNX opset 17.
    Dynamic batch axis is enabled so the SavedModel accepts any batch size.
    """
    dummy = torch.randn(1, input_size)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
    )
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX graph exported and validated  →  {onnx_path}")


# ---------------------------------------------------------------------------
# 6.  Convert ONNX → TensorFlow SavedModel
# ---------------------------------------------------------------------------

def convert_onnx_to_savedmodel(onnx_path: str, output_dir: str) -> None:
    """
    Use onnx2tf to convert the ONNX file to a TF SavedModel directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Converting ONNX → SavedModel  →  {output_dir}")
    
    # onnx2tf 2.4.0+ uses 'flatbuffer_direct' by default.
    # We must explicitly set flatbuffer_direct_output_saved_model=True 
    # to ensure a TensorFlow SavedModel is generated alongside the TFLite model.
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=output_dir,
        copy_onnx_input_output_names_to_tflite=True,
        flatbuffer_direct_output_saved_model=True,
        non_verbose=False,
    )


# ---------------------------------------------------------------------------
# 7.  Numerical verification
# ---------------------------------------------------------------------------

def verify_outputs(
    pt_model: nn.Module,
    savedmodel_dir: str,
    input_size: int,
    n_samples: int = 5,
    atol: float = 1e-4,
) -> bool:
    """
    Run random inputs through both the PyTorch model and the TF SavedModel
    and confirm outputs agree within tolerance.
    """
    print("\nVerifying numerical outputs …")

    # Force eval mode — critical for BatchNorm to use running stats
    pt_model.eval()

    # onnx2tf typically creates a 'saved_model' subfolder within the output directory
    actual_sm_path = savedmodel_dir
    potential_sm_path = os.path.join(savedmodel_dir, "saved_model")
    if os.path.isdir(potential_sm_path):
        actual_sm_path = potential_sm_path
    
    print(f"  Loading SavedModel from: {actual_sm_path}")
    tf_model  = tf.saved_model.load(actual_sm_path)
    infer_fn  = tf_model.signatures["serving_default"]
    
    # Detect input and output keys dynamically as onnx2tf may rename them
    input_key  = list(infer_fn.structured_input_signature[1].keys())[0]
    output_key = list(infer_fn.structured_outputs.keys())[0]
    print(f"  TF keys | input: '{input_key}' | output: '{output_key}'")

    all_ok = True
    rng    = np.random.default_rng(42)

    for i in range(n_samples):
        x_np = rng.standard_normal((1, input_size)).astype(np.float32)

        with torch.no_grad():
            pt_out = pt_model(torch.tensor(x_np)).numpy()

        # Run inference using the detected input key
        tf_out = infer_fn(**{input_key: tf.constant(x_np)})[output_key].numpy()

        max_diff = float(np.abs(pt_out - tf_out).max())
        status   = "✓" if max_diff < atol else "✗"
        print(f"  Sample {i + 1}: max |Δ| = {max_diff:.2e}  {status}")
        if max_diff >= atol:
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# 8.  Main entry point
# ---------------------------------------------------------------------------

def convert_to_savedmodel(
    model_path: str,
    output_dir: str,
    scaler_path: str | None = None,
    keep_onnx: bool = False,
) -> None:
    """
    Full conversion pipeline:
      1. Load PyTorch checkpoint
      2. Optionally wrap with z-score scaler (baked into the graph)
      3. Export to ONNX
      4. Convert ONNX → TF SavedModel via onnx2tf
      5. Verify numerical equivalence
    """
    print(f"── Step 1/4  Load checkpoint ────────────────────────────")
    model, input_size = load_checkpoint(model_path)

    if scaler_path:
        print(f"\n── Step 2/4  Embed scaler ───────────────────────────────")
        mean, scale = load_scaler(scaler_path)
        print(f"  mean shape={mean.shape}  scale shape={scale.shape}")
        export_model = ScalerWrapper(model, mean, scale).eval()
        print("  Scaler embedded — SavedModel will accept raw inputs.")
    else:
        print("\n── Step 2/4  No scaler provided, skipping ───────────────")
        export_model = model.eval()

    if keep_onnx:
        onnx_path = os.path.join(output_dir, "model.onnx")
        os.makedirs(output_dir, exist_ok=True)
    else:
        _tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        onnx_path = _tmp.name
        _tmp.close()

    try:
        print(f"\n── Step 3/4  Export ONNX ────────────────────────────────")
        export_onnx(export_model, input_size, onnx_path)

        print(f"\n── Step 4/4  Convert to SavedModel ──────────────────────")
        convert_onnx_to_savedmodel(onnx_path, output_dir)
    finally:
        if not keep_onnx and os.path.exists(onnx_path):
            os.remove(onnx_path)

    ok = verify_outputs(export_model, output_dir, input_size)
    if ok:
        print("\n✓  All outputs match within tolerance.")
    else:
        print("\n⚠  Some outputs exceeded tolerance — check the diff above.")

    print(f"\nDone. SavedModel written to: {output_dir}")

    # Cleanup: remove default tflite files if only SavedModel was intended
    # (onnx2tf 2.4.0 generates these by default in the output_dir)
    default_tflites = glob.glob(os.path.join(output_dir, "*.tflite"))

    for f in default_tflites:
        try:
            os.remove(f)
            print(f"Removed default tflite: {f}")
        except OSError as e:
            print(f"Error removing {f}: {e}")


# ---------------------------------------------------------------------------
# 9.  CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    convert_to_savedmodel(
        model_path="models/trained_model.pth",
        scaler_path="processed_data.h5.scaler.npz",  # set to None to skip
        output_dir="models/saved_model",
        keep_onnx=False,
    )