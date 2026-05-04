"""
inference.py
------------

BME688 Live Inference using LiteRT (TFLite)
Mirrors the preprocessing in DataPreprocessor.read_bmespecimen_file()

Feature layout (13 values):
  - Indices 0–8:  resistance_gassensor from readings 1–9
  - Indices 9–12: resistance_gassensor, temperature, pressure,
                  relative_humidity from reading 10 (the last one)
"""

import numpy as np
import collections
import logging
import time
import json

# LiteRT
# Falls back to tflite_runtime or tensorflow.lite if LiteRT is not installed
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
WINDOW_SIZE   = 10   # number of BME688 readings per inference, this needs to match training window size

class BME688Classifier:
    """
    Collects live BME688 readings and runs TFLite inference once a full
    window of WINDOW_SIZE readings has been accumulated.

    Usage:
        clf = BME688Classifier("model.tflite", class_names=["air", "ethanol", ...])

        # Call this every time you get a new reading from the sensor:
        result = clf.update(resistance_gassensor, temperature, pressure, relative_humidity)

        # result is None while the window is filling up.
        # Once full it returns {"class_index": int, "class_name": str, "probabilities": np.ndarray}
        # The window then resets so the next 10 readings form a fresh prediction.
    """

    def __init__(self, model_path: str, class_names: list[str] | None = None):
        """
        Args:
            model_path:  Path to the .tflite model file.
            class_names: Optional list of label strings ordered by class index.
                         e.g. ["clean_air", "ethanol", "acetone"]
        """
        self.class_names = class_names
        self._window: collections.deque = collections.deque(maxlen=WINDOW_SIZE)

        # --- Load model ---
        self._interpreter = Interpreter(model_path=model_path)
        self._interpreter.allocate_tensors()

        input_details  = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()

        self._input_index  = input_details[0]["index"]
        self._output_index = output_details[0]["index"]
        self._input_dtype  = input_details[0]["dtype"]

        logger.info(
            f"Model loaded from '{model_path}'. "
            f"Input shape: {input_details[0]['shape']}, dtype: {self._input_dtype}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        resistance_gassensor: float,
        temperature: float,
        pressure: float,
        relative_humidity: float,
        cycle_step_index: int | None = None,
    ) -> dict | None:
        """
        Feed one BME688 reading into the classifier.

        Returns a result dict when a full window is ready, otherwise None.
        """
        # --- Alignment Logic ---
        if cycle_step_index is not None:
            # Always start a new window on step 0
            if cycle_step_index == 0:
                if len(self._window) > 0:
                    logger.warning("New cycle started before previous window finished. Resetting.")
                self._window.clear()

        # If we haven't seen a 0 yet, don't start collecting
        elif len(self._window) == 0:
            return None
        # Check for dropped messages (e.g., we have 3 items, next should be index 3)
        elif len(self._window) != cycle_step_index:
            logger.warning(f"Drop detected! Expected step {len(self._window)}, got {cycle_step_index}. Resetting.")
            self._window.clear()
            return None
        
        reading = [
            resistance_gassensor,
            temperature,
            pressure,
            relative_humidity,
        ]
        self._window.append(reading)

        if len(self._window) < WINDOW_SIZE:
            logger.debug(f"Window filling: {len(self._window)}/{WINDOW_SIZE}")
            return None

        # Build feature vector exactly as in training
        feature_vector = self._build_features()

        # Run inference
        result = self._predict(feature_vector)

        # Reset window for next batch
        self._window.clear()

        return result

    def reset(self):
        """Discard buffered readings and start a fresh window."""
        self._window.clear()
        logger.info("Window reset.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_features(self) -> np.ndarray:
        """
        Reconstruct the 13-element feature vector used during training.

        Training logic (from read_bmespecimen_file):
            group    = 10 readings  [r0, r1, ..., r8, r9]
            tenth    = group[-1]            # all 4 values
            first_values = [lst[0] for lst in group[:-1]]   # gas only (index 0)
            features = first_values + tenth  → 9 + 4 = 13 values
        """
        window = list(self._window)           # list of 10 x [gas, T, P, RH]

        # Indices 0–8: gas resistance from readings 0–8
        first_values = [reading[0] for reading in window[:-1]]   # 9 values

        # Indices 9–12: all 4 channels of the 10th reading
        tenth = window[-1].copy()                                  # 4 values

        features = np.array(first_values + tenth, dtype=np.float32)  # shape (13,)
        return features

    def _predict(self, feature_vector: np.ndarray) -> dict:
        """Run the TFLite model on one 13-element feature vector."""
        # Model expects shape (1, 13)
        input_data = feature_vector.reshape(1, -1).astype(self._input_dtype)

        self._interpreter.set_tensor(self._input_index, input_data)
        start_time = time.perf_counter_ns()
        self._interpreter.invoke()
        end_time = time.perf_counter_ns()
        inference_time = (end_time - start_time) / 1000


        output = self._interpreter.get_tensor(self._output_index)  # shape (1, n_classes)
        probabilities = output[0]

        class_index = int(np.argmax(probabilities))
        class_name = (
            self.class_names[class_index]
            if self.class_names and class_index < len(self.class_names)
            else str(class_index)
        )

        logger.info(
            f"Prediction: {class_name} (index {class_index}), "
            f"confidence: {probabilities[class_index]:.1f}, "
            f"time taken: {inference_time} microseconds"
        )

        return {
            "class_index":   class_index,
            "class_name":    class_name,
            "probabilities": probabilities,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import threading
    import queue
    import paho.mqtt.client as mqtt
    from bme688_simulator import BME688Simulator
 
    parser = argparse.ArgumentParser(
        description="Run BME688 TFLite inference using simulator or MQTT readings."
    )
    parser.add_argument(
        "--model-path",
        default="models/int8_dyn_model.tflite",
        help="Path to .tflite model file.",
    )
    parser.add_argument(
        "--class-names",
        default=None,
        help="Comma-separated class labels, e.g. clean_air,ethanol,acetone",
    )
    parser.add_argument(
        "--specimen-source",
        default="data/specimendata/",
        help="Simulator source: directory or single .bmespecimen file.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay (seconds) between simulator readings.",
    )
    parser.add_argument(
        "--mode",
        choices=["simulator", "mqtt"],
        default="simulator",
        help="Reading input mode.",
    )
    parser.add_argument(
        "--mqtt-host",
        default="localhost",
        help="MQTT broker host.",
    )
    parser.add_argument(
        "--mqtt-port",
        type=int,
        default=1883,
        help="MQTT broker port.",
    )
    parser.add_argument(
        "--mqtt-topic",
        default="bme688/readings",
        help="MQTT topic to subscribe to.",
    )
    args = parser.parse_args()

    model_path = args.model_path
    class_names = args.class_names.split(",") if args.class_names else None
    specimen_source = args.specimen_source
    delay = args.delay
    input_mode = args.mode
    mqtt_host = args.mqtt_host
    mqtt_port = args.mqtt_port
    mqtt_topic = args.mqtt_topic
 
    clf = BME688Classifier(model_path, class_names=class_names)
    reading_count = 0
    prediction_count = 0
    correct = 0

    if input_mode == "simulator":
        sim = BME688Simulator(specimen_source, delay=delay, noise_std=0.02)
        label_to_idx = {name: i for i, name in enumerate(sim.unique_labels())}

        print(f"\nInput mode        : simulator")
        print(f"Labels in dataset : {sim.unique_labels()}")
        print(f"Total readings    : {sim.total_readings}")
        print(f"Window size       : {WINDOW_SIZE} readings per inference")
        print(f"Playback delay    : {delay}s\n")

        for reading in sim:
            reading_count += 1

            result = clf.update(
                resistance_gassensor=reading.gas,
                temperature=reading.temperature,
                pressure=reading.pressure,
                relative_humidity=reading.humidity,
                cycle_step_index=reading.cycle_step_index
            )

            if result is not None:
                prediction_count += 1
                predicted = result["class_index"]
                actual = label_to_idx.get(reading.label, -1)  # label of window's last reading
                match = predicted == actual
                if match:
                    correct += 1

                print(f"=== Inference #{prediction_count} (reading {reading_count}) ===")
                print(f"  Predicted  : {predicted} (index {result['class_index']})")
                print(f"  Actual     : {actual}  {'✓' if match else '✗'}")
                print(f"  Confidence : {result['probabilities'][result['class_index']]:.1f}")
                print(f"  All probs  : {result['probabilities']}\n")

        if prediction_count:
            print(f"--- Summary ------------------------------")
            print(f"  Readings processed : {reading_count}")
            print(f"  Predictions made   : {prediction_count}")
            print(f"  Correct            : {correct}")
            print(f"  Accuracy           : {correct / prediction_count:.1f}")
        else:
            print("No predictions were made - check your specimen files and model.")

    elif input_mode == "mqtt":
        print(f"\nInput mode     : mqtt")
        print(f"Broker         : {mqtt_host}:{mqtt_port}")
        print(f"Topic          : {mqtt_topic}")
        print(f"Window size    : {WINDOW_SIZE} readings per inference")
        print("")

        reading_queue: queue.Queue[dict] = queue.Queue()
        stop_event = threading.Event()

        def on_connect(client, userdata, flags, reason_code, properties=None):
            if reason_code == 0:
                logger.info(f"Connected to MQTT broker {mqtt_host}:{mqtt_port}")
                client.subscribe(mqtt_topic)
                logger.info(f"Subscribed to topic '{mqtt_topic}'")
            else:
                logger.error(f"MQTT connect failed with code {reason_code}")

        def on_message(client, userdata, msg):
            try:
                payload = json.loads(msg.payload.decode("utf-8"))
                reading_queue.put(payload)
            except Exception as exc:
                logger.warning(f"Failed to parse MQTT payload: {exc}")

        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(mqtt_host, mqtt_port, 60)
        client.loop_start()

        try:
            while not stop_event.is_set():
                payload = reading_queue.get()
                reading_count += 1

                result = clf.update(
                    resistance_gassensor=float(payload["gas"]),
                    temperature=float(payload["temperature"]),
                    pressure=float(payload["pressure"]),
                    relative_humidity=float(payload["humidity"]),
                    cycle_step_index=int(payload["cycle_step_index"])
                )

                if result is not None:
                    prediction_count += 1
                    print(f"=== Inference #{prediction_count} (reading {reading_count}) ===")
                    print(f"  Predicted  : {result['class_name']} (index {result['class_index']})")
                    print(f"  Confidence : {result['probabilities'][result['class_index']]:.1f}")
                    print(f"  All probs  : {result['probabilities']}\n")
        except KeyboardInterrupt:
            print("\nStopping MQTT inference...")
        finally:
            client.loop_stop()
            client.disconnect()
            print(f"Readings processed : {reading_count}")
            print(f"Predictions made   : {prediction_count}")
