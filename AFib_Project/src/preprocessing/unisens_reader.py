"""
=============================================================================
FILE: src/preprocessing/unisens_reader.py
PROJECT: HRV Feature Extraction and Context-Aware False Positive Analysis

PHASE 4 (part 1 of 2): READING THE RAW UNISENS FILES
=============================================================================

PURPOSE:
    CACHET-CADB does not ship CSV files of ECG. It ships RAW BINARY dumps
    straight from the sensor, in a format called Unisens. This module is the
    only place in the project that knows how to turn those bytes into numbers.

WHAT IS UNISENS?
    A recording folder looks like this:

        signal/P1/a2b3c4@cachet.dk1/0/
            unisens.xml     <- the METADATA. Describes every other file.
            ecg.bin         <- 60 MB of raw ECG
            acc.bin         <- accelerometer, 3 channels interleaved
            angularrate.bin <- gyroscope, 3 channels interleaved
            press.bin       <- barometric pressure
            hr_live.bin     <- device-computed heart rate, 1 value/minute
            ...

    The .bin files have NO header. They are just numbers packed end to end.
    On its own, ecg.bin is unreadable - you cannot know whether a pair of
    bytes is one 16-bit number or half of a 32-bit one. unisens.xml is what
    makes it readable. It tells us, per file:

        dataType      "int16"   -> each sample is 2 bytes, signed
        endianess     "LITTLE"  -> least significant byte comes first
        sampleRate    1024      -> samples per second
        baseline      2048      -> the ADC value that means "zero volts"
        lsbValue      0.00268   -> millivolts per ADC step
        channel       (count)   -> how many signals are interleaved

    So the conversion from bytes to millivolts is:

        raw_int = numpy.fromfile(path, dtype='<i2')       # honour endianness
        millivolts = (raw_int - baseline) * lsbValue

    THIS IS THE WHOLE TRICK. Everything else in this file is bookkeeping.

WHY WE PARSE THE XML INSTEAD OF HARDCODING:
    It would be shorter to write `fs = 1024` and move on. We do not, for two
    reasons. First, an earlier version of this project's config confidently
    said 256 Hz - a wrong constant that would have silently corrupted every
    heart-rate calculation by a factor of four. Reading the metadata makes
    that class of bug impossible. Second, if a single recording in the
    dataset was made with different settings, we find out instead of quietly
    producing garbage for that subject.

AUTHOR: [Your Name]
=============================================================================
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# -----------------------------------------------------------------------------
# Mapping from the Unisens dataType string to a numpy dtype.
# The "<" and ">" prefixes are numpy's notation for byte order:
#   "<i2" = little-endian signed 2-byte integer
#   ">i4" = big-endian   signed 4-byte integer
# Getting this wrong does not crash - it silently produces nonsense numbers,
# which is far worse. Hence the explicit table.
# -----------------------------------------------------------------------------
_DTYPE_MAP = {
    "int8": "i1",
    "uint8": "u1",
    "int16": "i2",
    "uint16": "u2",
    "int32": "i4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}

# Unisens XML uses a namespace. ElementTree reports tags as
# "{http://www.unisens.org/unisens2.0}signalEntry", so we strip it.
def _tag(element) -> str:
    """Return an element's tag name without its XML namespace prefix."""
    return element.tag.split("}")[-1]


@dataclass
class SignalEntry:
    """
    Everything unisens.xml tells us about ONE binary file.

    Think of this as the "label on the tin". It does not hold any signal data
    itself - just the instructions for reading it.
    """

    entry_id: str                 # filename, e.g. "ecg.bin"
    content_class: str            # "ecg", "acc", "press", ...
    sample_rate: float            # Hz
    dtype: str                    # numpy dtype string, e.g. "<i2"
    baseline: float               # ADC value representing zero
    lsb_value: float              # physical units per ADC step
    unit: str                     # "mV", "g", "Pa", ...
    channels: List[str] = field(default_factory=list)

    @property
    def n_channels(self) -> int:
        """How many signals are interleaved in this one file (>=1)."""
        return max(1, len(self.channels))


@dataclass
class UnisensRecording:
    """
    One recording PART - i.e. one folder containing a unisens.xml.

    A subject has sessions; a session is split into parts numbered
    0, 1, 2, ... and a final part named "<n>-last". Each part is an
    independent recording with its own sample-index origin. That last point
    matters enormously: the Start/End sample indices in annotation.csv are
    relative to THEIR OWN PART, not to the session as a whole.
    """

    path: str
    measurement_id: str
    duration_sec: float
    timestamp_start: str
    entries: Dict[str, SignalEntry]           # keyed by content_class
    attributes: Dict[str, str]                # age, gender, weight, ...

    # ---------------------------------------------------------------- helpers
    @property
    def ecg(self) -> Optional[SignalEntry]:
        return self.entries.get("ecg")

    @property
    def subject_age(self) -> Optional[float]:
        value = self.attributes.get("age")
        return float(value) if value else None

    @property
    def subject_gender(self) -> Optional[str]:
        return self.attributes.get("gender")


def read_unisens_metadata(part_dir: str) -> UnisensRecording:
    """
    Parse the unisens.xml in `part_dir` and return a UnisensRecording.

    PARAMETERS
    ----------
    part_dir : str
        Folder containing unisens.xml, e.g.
        ".../signal/P1/a2b3c4@cachet.dk1/0"

    RETURNS
    -------
    UnisensRecording

    RAISES
    ------
    FileNotFoundError
        If there is no unisens.xml. A signal folder without one is unusable,
        and we would rather stop loudly here than skip data silently.
    """
    xml_path = os.path.join(part_dir, "unisens.xml")
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"No unisens.xml in: {part_dir}")

    root = ET.parse(xml_path).getroot()

    # --- top-level attributes -------------------------------------------
    measurement_id = root.attrib.get("measurementId", "")
    duration_sec = float(root.attrib.get("duration", 0) or 0)
    timestamp_start = root.attrib.get("timestampStart", "")

    # --- customAttributes: age, gender, weight, height, sensor -----------
    attributes: Dict[str, str] = {}
    for element in root.iter():
        if _tag(element) == "customAttribute":
            attributes[element.attrib.get("key", "")] = element.attrib.get("value", "")

    # --- signalEntry blocks: one per .bin file ---------------------------
    entries: Dict[str, SignalEntry] = {}
    for element in root.iter():
        if _tag(element) != "signalEntry":
            continue

        attrib = element.attrib
        data_type = attrib.get("dataType", "int16").lower()
        if data_type not in _DTYPE_MAP:
            # Unknown format: skip rather than guess. Guessing byte layout is
            # how you get a plausible-looking but completely wrong signal.
            continue

        # Endianness lives on a nested <binFileFormat endianess="LITTLE"/>
        endianness = "LITTLE"
        for child in element:
            if _tag(child) == "binFileFormat":
                endianness = child.attrib.get("endianess", "LITTLE").upper()

        prefix = "<" if endianness == "LITTLE" else ">"
        numpy_dtype = prefix + _DTYPE_MAP[data_type]

        channels = [
            child.attrib.get("name", "")
            for child in element
            if _tag(child) == "channel"
        ]

        content_class = attrib.get("contentClass", attrib.get("id", "unknown"))

        entries[content_class] = SignalEntry(
            entry_id=attrib.get("id", ""),
            content_class=content_class,
            sample_rate=float(attrib.get("sampleRate", 0) or 0),
            dtype=numpy_dtype,
            baseline=float(attrib.get("baseline", 0) or 0),
            lsb_value=float(attrib.get("lsbValue", 1) or 1),
            unit=attrib.get("unit", ""),
            channels=channels,
        )

    return UnisensRecording(
        path=part_dir,
        measurement_id=measurement_id,
        duration_sec=duration_sec,
        timestamp_start=timestamp_start,
        entries=entries,
        attributes=attributes,
    )


def read_signal(
    recording: UnisensRecording,
    content_class: str = "ecg",
    start_sample: Optional[int] = None,
    n_samples: Optional[int] = None,
    physical: bool = True,
) -> np.ndarray:
    """
    Read a slice of one signal out of its .bin file.

    THE IMPORTANT PART - WHY WE SLICE INSTEAD OF LOADING EVERYTHING:
        ecg.bin for a single part is ~60 MB, and there are dozens of parts.
        Loading them all would need tens of gigabytes of RAM. But we only ever
        need the 10-second windows that carry a label - about 0.3% of the
        recorded data. numpy.fromfile(offset=..., count=...) seeks straight to
        the bytes we want, so memory use stays flat no matter how big the
        dataset grows.

    PARAMETERS
    ----------
    recording : UnisensRecording
        From read_unisens_metadata().
    content_class : str
        Which signal: "ecg", "acc", "movementAcceleration_live", ...
    start_sample : int, optional
        First sample to read (per channel). None = from the beginning.
    n_samples : int, optional
        How many samples to read (per channel). None = to the end.
    physical : bool
        True  -> convert to physical units (mV, g, ...) using baseline/lsbValue
        False -> return the raw ADC integers

    RETURNS
    -------
    np.ndarray
        Shape (n_samples,) for single-channel signals,
        shape (n_samples, n_channels) for interleaved ones such as acc.

    RAISES
    ------
    KeyError / FileNotFoundError
        If the requested signal is not present in this recording.
    """
    if content_class not in recording.entries:
        raise KeyError(
            f"Signal '{content_class}' not in {recording.path}. "
            f"Available: {sorted(recording.entries)}"
        )

    entry = recording.entries[content_class]
    bin_path = os.path.join(recording.path, entry.entry_id)
    if not os.path.isfile(bin_path):
        raise FileNotFoundError(f"Missing binary file: {bin_path}")

    dtype = np.dtype(entry.dtype)
    n_ch = entry.n_channels

    # Byte offset. Channels are INTERLEAVED - for a 3-channel accelerometer the
    # file reads x0,y0,z0,x1,y1,z1,... - so sample k begins at byte
    # k * n_channels * itemsize.
    offset_bytes = 0
    if start_sample:
        offset_bytes = int(start_sample) * n_ch * dtype.itemsize

    count = -1 if n_samples is None else int(n_samples) * n_ch

    data = np.fromfile(bin_path, dtype=dtype, count=count, offset=offset_bytes)

    if n_ch > 1:
        # Drop any trailing partial frame before reshaping, otherwise numpy
        # raises. A truncated final frame happens when a recording is cut off.
        usable = (len(data) // n_ch) * n_ch
        data = data[:usable].reshape(-1, n_ch)

    if physical:
        # (raw - baseline) * lsbValue  ->  millivolts / g / Pa
        # float32 is deliberate: it halves memory versus float64 and is far
        # more precision than a 16-bit ADC can justify.
        data = (data.astype(np.float32) - entry.baseline) * entry.lsb_value

    return data


def read_ecg_segment(
    recording: UnisensRecording,
    start_sample: int,
    end_sample: int,
) -> np.ndarray:
    """
    Convenience wrapper: read ECG between two sample indices, in millivolts.

    `start_sample` and `end_sample` are exactly the Start and End columns of
    annotation.csv, which are already expressed in ECG samples at the native
    rate - so no conversion is needed here.
    """
    return read_signal(
        recording,
        content_class="ecg",
        start_sample=start_sample,
        n_samples=end_sample - start_sample,
        physical=True,
    )


# -----------------------------------------------------------------------------
# SELF-TEST
# Run directly to verify the reader against one real recording:
#     python src/preprocessing/unisens_reader.py
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from config.config import CADB_SIGNAL_DIR

    demo = os.path.join(CADB_SIGNAL_DIR, "P1", "a2b3c4@cachet.dk1", "0")
    rec = read_unisens_metadata(demo)

    print("=" * 64)
    print("UNISENS READER SELF-TEST")
    print("=" * 64)
    print(f"Path        : {rec.path}")
    print(f"Measurement : {rec.measurement_id}")
    print(f"Duration    : {rec.duration_sec:.0f} s "
          f"({rec.duration_sec / 3600:.2f} hours)")
    print(f"Started     : {rec.timestamp_start}")
    print(f"Subject     : age={rec.subject_age}, gender={rec.subject_gender}, "
          f"sensor={rec.attributes.get('sensorType')}")
    print("-" * 64)
    print(f"{'contentClass':<28}{'rate(Hz)':>10}{'ch':>4}{'unit':>6}  dtype")
    for name, e in rec.entries.items():
        print(f"{name:<28}{e.sample_rate:>10.4f}{e.n_channels:>4}"
              f"{e.unit:>6}  {e.dtype}")
    print("-" * 64)

    ecg = read_signal(rec, "ecg", start_sample=0, n_samples=10 * 1024)
    print(f"ECG 10 s slice : shape={ecg.shape}, dtype={ecg.dtype}")
    print(f"  range        : {ecg.min():.3f} to {ecg.max():.3f} mV")
    print(f"  mean         : {ecg.mean():.4f} mV")

    acc = read_signal(rec, "acc", start_sample=0, n_samples=10 * 64)
    print(f"ACC 10 s slice : shape={acc.shape} (x, y, z)")
    print(f"  magnitude    : {np.linalg.norm(acc, axis=1).mean():.3f} g "
          f"(~1.0 g at rest = gravity)")
    print("=" * 64)
