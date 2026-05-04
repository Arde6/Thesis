"""
bme688_simulator.py
-------------------

Simulates a bme688 sending readings.
Sends via MQTT or staright to code, see inference.py.
"""

import json
import os
import glob
import time
import random
import logging
import argparse
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Sequence, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    import paho.mqtt.client as mqtt  # type: ignore
except Exception:
    mqtt = None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SensorReading:
    """One BME688 reading frame."""
    gas: float           # resistance
    temperature: float   # °C
    pressure: float      # hPa
    humidity: float      # %RH
    # Metadata (for confirming model accuracy during inference)
    label: str = ""
    specimen_id: str = ""
    cycle_id: Optional[int] = None
    cycle_step_index: Optional[int] = None
    heater_temperature: Optional[float] = None


# ---------------------------------------------------------------------------
# Parsers (extensible)
# ---------------------------------------------------------------------------

class ReadingParser(Protocol):
    """Pluggable parser for a single file into SensorReading frames."""

    def can_parse(self, file_path: str) -> bool:
        ...

    def parse(self, file_path: str) -> List[SensorReading]:
        ...


class ParserRegistry:
    """Registry for file format parsers."""

    def __init__(self) -> None:
        self._parsers: List[ReadingParser] = []

    def register(self, parser: ReadingParser) -> None:
        self._parsers.append(parser)

    @property
    def parsers(self) -> Sequence[ReadingParser]:
        return tuple(self._parsers)

    def parser_for(self, file_path: str) -> Optional[ReadingParser]:
        for p in self._parsers:
            try:
                if p.can_parse(file_path):
                    return p
            except Exception as e:
                logger.debug("Parser %s failed can_parse(%s): %s", type(p).__name__, file_path, e)
        return None

    def parse_file(self, file_path: str) -> List[SensorReading]:
        parser = self.parser_for(file_path)
        if parser is None:
            raise ValueError(f"No parser registered for file: '{file_path}'")
        return parser.parse(file_path)


def _parse_bmespecimen(file_path: str) -> List[SensorReading]:
    """
    Parse a single .bmespecimen JSON file into an ordered list of
    SensorReading objects — one per specimenDataPoint row, in file order.
    Mirrors read_bmespecimen_file() from DataPreprocessor but keeps ALL
    readings (no grouping / windowing) so the simulator can replay them
    one by one.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    root = data.get("data", {})

    # --- specimen metadata ---
    specimen_data = root.get("specimenData", {})
    specimen_id   = str(specimen_data.get("id", ""))
    label         = specimen_data.get("label", "unknown")

    # --- heaterProfiles lookup: id → list[temperature] ---
    heater_profiles: dict = {}
    for hp in root.get("heaterProfiles", []):
        steps = hp.get("steps", [])
        temps = [s.get("temperature") for s in steps if "temperature" in s]
        heater_profiles[hp.get("id")] = temps

    # --- sensorConfigs lookup: sensorId → heaterProfileId ---
    sensor_to_heater: dict = {}
    for sc in root.get("sensorConfigs", []):
        if "sensorId" in sc and "heaterProfileId" in sc:
            sensor_to_heater[sc["sensorId"]] = sc["heaterProfileId"]

    # --- cycles lookup: cycle_id → sensorId ---
    cycle_to_sensor: dict = {}
    for cy in root.get("cycles", []):
        if "id" in cy and "sensorId" in cy:
            cycle_to_sensor[cy["id"]] = cy["sensorId"]

    # --- dataColumns key list ---
    column_keys = [
        dc.get("key")
        for dc in root.get("dataColumns", [])
        if "key" in dc
    ]

    # --- specimenDataPoints ---
    raw_points = root.get("specimenDataPoints", [])
    if not raw_points:
        logger.warning(f"No specimenDataPoints in {file_path}")
        return []

    # Normalise to list-of-lists
    if not isinstance(raw_points[0], list):
        raw_points = [raw_points]

    readings: List[SensorReading] = []

    for point in raw_points:
        # Map list values → column keys
        dp: dict = {key: point[i] for i, key in enumerate(column_keys) if i < len(point)}

        # Resolve sensor / heater chain
        cycle_id     = dp.get("cycle_id")
        sensor_id    = cycle_to_sensor.get(cycle_id)
        hp_id        = sensor_to_heater.get(sensor_id) if sensor_id is not None else None
        step_index   = dp.get("cycle_step_index")
        heater_temp: Optional[float] = None
        if hp_id is not None and step_index is not None:
            temps = heater_profiles.get(hp_id, [])
            if step_index < len(temps):
                heater_temp = temps[step_index]

        readings.append(SensorReading(
            gas               = dp.get("resistance_gassensor", 0.0),
            temperature       = dp.get("temperature", 0.0),
            pressure          = dp.get("pressure", 0.0),
            humidity          = dp.get("relative_humidity", 0.0),
            label             = label,
            specimen_id       = specimen_id,
            cycle_id          = cycle_id,
            cycle_step_index  = step_index,
            heater_temperature= heater_temp,
        ))

    return readings


class BmeSpecimenParser:
    """Parser for `.bmespecimen` JSON files."""

    def can_parse(self, file_path: str) -> bool:
        return file_path.lower().endswith(".bmespecimen")

    def parse(self, file_path: str) -> List[SensorReading]:
        return _parse_bmespecimen(file_path)


DEFAULT_PARSERS = ParserRegistry()
DEFAULT_PARSERS.register(BmeSpecimenParser())


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class BME688Simulator:
    """
    Simulates a live BME688 sensor by replaying readings from one or more
    files supported by registered parsers.

    Parameters
    ----------
    source : str
        Path to a directory of supported files OR a single file.
    delay : float
        Seconds to sleep between readings (0 = as fast as possible).
    shuffle_files : bool
        Randomise the order in which files are played back.
    loop : bool
        When all files are exhausted, restart from the beginning.
    random_seed : int | None
        Seed for file-order shuffling (None = non-deterministic).
    noise-std : float
        Random noise to readings.
    """

    def __init__(
        self,
        source: str,
        delay: float = 0.0,
        shuffle_files: bool = False,
        loop: bool = False,
        random_seed: Optional[int] = None,
        noise_std: float = 0.0,  # Standard deviation for noise (0.0 = off)
        parsers: ParserRegistry = DEFAULT_PARSERS,
    ):
        self.delay        = delay
        self.loop         = loop
        self.noise_std    = noise_std
        self._rng         = random.Random(random_seed)
        self._parsers     = parsers

        # Resolve file list
        if os.path.isfile(source):
            self._files = [source]
        elif os.path.isdir(source):
            candidates = sorted(glob.glob(os.path.join(source, "*")))
            self._files = [
                fp
                for fp in candidates
                if os.path.isfile(fp) and self._parsers.parser_for(fp) is not None
            ]
            if not self._files:
                known = ", ".join(type(p).__name__ for p in self._parsers.parsers) or "(none)"
                raise FileNotFoundError(
                    f"No supported files found in '{source}'. Registered parsers: {known}"
                )
        else:
            raise FileNotFoundError(f"Source not found: '{source}'")

        if shuffle_files:
            self._rng.shuffle(self._files)

        logger.info(f"Simulator initialised with {len(self._files)} file(s).")

        # Pre-parse all files into an in-memory reading list
        self._all_readings: List[SensorReading] = []
        for fp in self._files:
            try:
                parsed = self._parsers.parse_file(fp)
                logger.info(f"  Loaded {len(parsed):>6} readings from {os.path.basename(fp)}")
                self._all_readings.extend(parsed)
            except Exception as e:
                logger.warning(f"  Skipping {fp}: {e}")

        if not self._all_readings:
            raise ValueError("No readings could be parsed from the provided files.")

        logger.info(f"Total readings available: {len(self._all_readings)}")

        self._index = 0

    # ------------------------------------------------------------------
    # Iterator protocol — use in a for-loop or while True
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[SensorReading]:
        return self
    
    def _add_noise(self, value: float, percentage: float) -> float:
        """Applies Gaussian noise based on a percentage of the value."""
        if percentage <= 0:
            return value
        # Calculate standard deviation as a fraction of the reading
        sigma = value * percentage
        return self._rng.gauss(value, sigma)

    def __next__(self) -> SensorReading:
        if self._index >= len(self._all_readings):
            if self.loop:
                self._index = 0
                logger.info("Simulator looping back to start.")
            else:
                raise StopIteration

        reading = self._all_readings[self._index]
        self._index += 1

        if self.delay > 0:
            time.sleep(self.delay)
        
        if self.noise_std > 0:
            return SensorReading(
                gas         = self._add_noise(reading.gas, self.noise_std),
                temperature = self._add_noise(reading.temperature, self.noise_std * 0.1), # Temp is usually more stable
                pressure    = self._add_noise(reading.pressure, 0.0001), # Pressure rarely fluctuates wildly
                humidity    = self._add_noise(reading.humidity, self.noise_std * 0.5),
                label       = reading.label,
                specimen_id = reading.specimen_id,
                cycle_id    = reading.cycle_id,
                cycle_step_index   = reading.cycle_step_index,
                heater_temperature = reading.heater_temperature
            )

        return reading

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def reset(self):
        """Rewind to the first reading."""
        self._index = 0

    @property
    def total_readings(self) -> int:
        return len(self._all_readings)

    @property
    def readings_remaining(self) -> int:
        return max(0, len(self._all_readings) - self._index)

    @property
    def current_label(self) -> Optional[str]:
        """Label of the reading that will be returned on the next call."""
        if self._index < len(self._all_readings):
            return self._all_readings[self._index].label
        return None

    def unique_labels(self) -> List[str]:
        """All class labels present in the loaded data."""
        return sorted(set(r.label for r in self._all_readings))
    
    def get_label_mapping(self) -> dict:
        """Returns a dictionary mapping string labels to their integer indices."""
        return {name: i for i, name in enumerate(self.unique_labels())}

    @property
    def current_label_index(self) -> Optional[int]:
        """Returns the integer index of the current reading's label."""
        mapping = self.get_label_mapping()
        label = self.current_label
        return mapping.get(label) if label else None


# ---------------------------------------------------------------------------
# Payload formatting (extensible)
# ---------------------------------------------------------------------------

ReadingFormatter = Callable[[SensorReading], Dict[str, Any]]


class FormatterRegistry:
    """Registry for payload formatters (by name)."""

    def __init__(self) -> None:
        self._formatters: Dict[str, ReadingFormatter] = {}

    def register(self, name: str, formatter: ReadingFormatter) -> None:
        self._formatters[name] = formatter

    def get(self, name: str) -> ReadingFormatter:
        if name not in self._formatters:
            known = ", ".join(sorted(self._formatters.keys())) or "(none)"
            raise KeyError(f"Unknown formatter '{name}'. Known: {known}")
        return self._formatters[name]

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._formatters.keys()))


def format_json_flat(reading: SensorReading) -> Dict[str, Any]:
    """Default: flat JSON-friendly dict per reading."""
    return {
        "timestamp": time.time(),
        "gas": reading.gas,
        "temperature": reading.temperature,
        "pressure": reading.pressure,
        "humidity": reading.humidity,
        "label": reading.label,
        "specimen_id": reading.specimen_id,
        "cycle_id": reading.cycle_id,
        "cycle_step_index": reading.cycle_step_index,
        "heater_temperature": reading.heater_temperature,
    }


DEFAULT_FORMATTERS = FormatterRegistry()
DEFAULT_FORMATTERS.register("json_flat", format_json_flat)


# ---------------------------------------------------------------------------
# MQTT publishing
# ---------------------------------------------------------------------------

class MqttPublisher:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        client_id: Optional[str] = None,
        keepalive: int = 60,
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.keepalive = keepalive
        self._client = None

    def connect(self) -> None:
        if mqtt is None:
            raise RuntimeError(
                "MQTT support requires 'paho-mqtt'. Install it (e.g. `pip install paho-mqtt`)."
            )
        self._client = mqtt.Client(client_id=self.client_id or "", protocol=mqtt.MQTTv311)
        self._client.connect(self.host, self.port, keepalive=self.keepalive)
        self._client.loop_start()

    def publish(self, topic: str, payload: bytes, *, qos: int = 0, retain: bool = False) -> None:
        if self._client is None:
            raise RuntimeError("MQTT client not connected. Call connect() first.")
        info = self._client.publish(topic, payload=payload, qos=qos, retain=retain)
        # Ensure delivery for QoS 1/2; QoS 0 is fire-and-forget.
        if qos > 0:
            info.wait_for_publish()

    def close(self) -> None:
        if self._client is None:
            return
        try:
            self._client.loop_stop()
        finally:
            self._client.disconnect()
            self._client = None


def publish_simulated_readings(
    sim: BME688Simulator,
    publisher: MqttPublisher,
    *,
    topic: str,
    formatter: ReadingFormatter = format_json_flat,
    qos: int = 0,
    retain: bool = False,
    max_readings: Optional[int] = None,
    dry_run: bool = False,
) -> int:
    """Publish simulator readings to MQTT. Returns number of readings processed."""
    n = 0
    for reading in sim:
        payload_dict = formatter(reading)
        payload_bytes = json.dumps(payload_dict, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

        if dry_run:
            logger.info("DRY_RUN topic=%s payload=%s", topic, payload_bytes.decode("utf-8", errors="replace"))
        else:
            publisher.publish(topic, payload_bytes, qos=qos, retain=retain)

        n += 1
        if max_readings is not None and n >= max_readings:
            break
    return n


# ---------------------------------------------------------------------------
# CLI — Simulate a bme688 in network sending data via MQTT
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(description="Replay BME688 readings from supported files and publish to MQTT")
    parser.add_argument("source", help="Directory of supported files, or a single file")
    parser.add_argument("--delay",   type=float, default=0.05,   help="Seconds between readings (default 0.05)")
    parser.add_argument("--loop",    action="store_true",        help="Loop forever")
    parser.add_argument("--shuffle", action="store_true",        help="Shuffle file order")
    parser.add_argument("--max",     type=int,   default=None,   help="Stop after N readings")
    parser.add_argument("--noise-std", type=float, default=0.0, help="Relative Gaussian noise level (0.0 disables noise)")
    parser.add_argument("--mqtt-host", type=str, default="localhost", help="MQTT broker host (default localhost)")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port (default 1883)")
    parser.add_argument("--mqtt-topic", type=str, default="bme688/readings", help="MQTT topic to publish to")
    parser.add_argument("--mqtt-qos", type=int, default=0, choices=(0, 1, 2), help="MQTT QoS (0,1,2)")
    parser.add_argument("--mqtt-retain", action="store_true", help="Publish retained MQTT messages")
    parser.add_argument("--format", type=str, default="json_flat", help="Payload format name")
    parser.add_argument("--dry-run", action="store_true", help="Log payloads instead of publishing to MQTT")
    args = parser.parse_args()

    sim = BME688Simulator(
        source        = args.source,
        delay         = args.delay,
        shuffle_files = args.shuffle,
        loop          = args.loop,
        noise_std     = args.noise_std,
    )

    formatter = DEFAULT_FORMATTERS.get(args.format)
    publisher = MqttPublisher(args.mqtt_host, args.mqtt_port)
    if not args.dry_run:
        publisher.connect()
        logger.info("Connected to MQTT %s:%s topic=%s", args.mqtt_host, args.mqtt_port, args.mqtt_topic)

    try:
        count = publish_simulated_readings(
            sim,
            publisher,
            topic=args.mqtt_topic,
            formatter=formatter,
            qos=args.mqtt_qos,
            retain=args.mqtt_retain,
            max_readings=args.max,
            dry_run=args.dry_run,
        )
        logger.info("Processed %s reading(s).", count)
    finally:
        publisher.close()

if __name__ == "__main__":
    _cli()