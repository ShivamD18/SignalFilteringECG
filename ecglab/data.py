"""Access to the PhysioNet databases used by ECG Signal Lab.

MIT-BIH Arrhythmia Database (``mitdb``)
    48 half-hour, two-lead ambulatory ECG recordings digitised at 360 Hz, with
    every beat labelled and cross-checked by two or more cardiologists.
    Moody GB, Mark RG. IEEE Eng Med Biol 20(3):45-50, 2001.

MIT-BIH Noise Stress Test Database (``nstdb``)
    Real noise recorded from active volunteers with electrodes placed where
    the ECG is not visible: baseline wander (``bw``), muscle artifact (``ma``)
    and electrode motion (``em``).
    Moody GB, Muldrow WE, Mark RG. Computers in Cardiology 11:381-384, 1984.

Both are distributed by PhysioNet (Goldberger AL et al., Circulation
101(23):e215-e220, 2000) under the Open Data Commons Attribution License v1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MITDB_DIR = DATA_DIR / "mitdb"
NSTDB_DIR = DATA_DIR / "nstdb"

FS = 360.0  # MIT-BIH sampling rate (Hz)

# AAMI EC57 groups the MIT-BIH beat codes into five classes:
#   N  normal and bundle-branch-block beats, nodal/atrial escape beats
#   S  supraventricular ectopic beats (atrial / nodal premature)
#   V  ventricular ectopic beats (PVCs, ventricular escape)
#   F  fusion of ventricular and normal beats
#   Q  paced, fusion of paced and normal, unclassifiable
SYMBOL_TO_AAMI = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N", "B": "N",
    "A": "S", "a": "S", "J": "S", "S": "S", "n": "S",
    "V": "V", "E": "V", "r": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}
BEAT_SYMBOLS = frozenset(SYMBOL_TO_AAMI)
CLASSES = ("N", "S", "V", "F")  # Q is too rare outside paced records to learn
CLASS_NAMES = {
    "N": "Normal / bundle branch block",
    "S": "Supraventricular ectopic (e.g. PAC)",
    "V": "Ventricular ectopic (e.g. PVC)",
    "F": "Fusion of ventricular and normal",
    "Q": "Paced / unclassifiable",
}

ALL_RECORDS = (
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
)
# AAMI recommends excluding the four paced records.
PACED_RECORDS = ("102", "104", "107", "217")

# Inter-patient split of de Chazal, O'Dwyer & Reilly (IEEE TBME 51(7), 2004).
# Training and test sets contain *different patients*, which is the only honest
# way to estimate how a classifier will do on someone it has never seen.
DS1 = (
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
    "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230",
)
DS2 = (
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
    "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234",
)

NOISE_RECORDS = {"bw": "Baseline wander", "ma": "Muscle (EMG) artifact", "em": "Electrode motion"}


@dataclass
class Record:
    """One ECG lead plus its reference annotations."""

    name: str
    signal: np.ndarray            # millivolts
    fs: float
    lead: str
    beat_samples: np.ndarray      # sample index of every annotated beat
    beat_symbols: np.ndarray      # MIT-BIH beat code for each beat
    rhythm: list[tuple[int, str]] = field(default_factory=list)  # (onset sample, label)

    @property
    def beat_classes(self) -> np.ndarray:
        return np.array([SYMBOL_TO_AAMI[s] for s in self.beat_symbols])

    def rhythm_at(self, sample: int) -> str:
        label = "(N"
        for onset, name in self.rhythm:
            if onset > sample:
                break
            label = name
        return label


# PhysioNet's official AWS Open Data copy is usually much faster than physionet.org.
MIRRORS = (
    "https://physionet-open.s3.amazonaws.com/{db}/1.0.0/{file}",
    "https://physionet.org/files/{db}/1.0.0/{file}",
)


def _fetch(db: str, file: str, dest: Path) -> None:
    import urllib.request

    last_error: Exception | None = None
    for template in MIRRORS:
        url = template.format(db=db, file=file)
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                payload = resp.read()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(payload)
            tmp.replace(dest)
            return
        except Exception as exc:  # try the next mirror
            last_error = exc
    raise RuntimeError(f"could not download {db}/{file}: {last_error}")


def download(force: bool = False) -> None:
    """Fetch mitdb and the three nstdb noise records from PhysioNet (~100 MB)."""
    try:  # use the OS certificate store (fixes SSL errors behind corporate proxies / AV)
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass

    jobs = [("mitdb", MITDB_DIR, r, ext) for r in ALL_RECORDS for ext in ("hea", "dat", "atr")]
    jobs += [("nstdb", NSTDB_DIR, r, ext) for r in NOISE_RECORDS for ext in ("hea", "dat")]
    for i, (db, folder, rec, ext) in enumerate(jobs, 1):
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / f"{rec}.{ext}"
        if dest.exists() and not force:
            continue
        print(f"[{i:3d}/{len(jobs)}] {db}/{rec}.{ext}", flush=True)
        _fetch(db, f"{rec}.{ext}", dest)
    print("PhysioNet data ready in", DATA_DIR)


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m ecglab download` first to fetch the PhysioNet data."
        )


def load_record(name: str, lead: str = "MLII") -> Record:
    """Load one MIT-BIH record, preferring the requested lead (MLII by default)."""
    import wfdb

    _require(MITDB_DIR / f"{name}.hea")
    rec = wfdb.rdrecord(str(MITDB_DIR / name))
    ann = wfdb.rdann(str(MITDB_DIR / name), "atr")

    ch = rec.sig_name.index(lead) if lead in rec.sig_name else 0
    signal = rec.p_signal[:, ch].astype(np.float64)

    symbols = np.asarray(ann.symbol)
    samples = np.asarray(ann.sample, dtype=np.int64)
    is_beat = np.isin(symbols, list(BEAT_SYMBOLS))

    rhythm = [
        (int(s), note.strip("\x00"))
        for s, sym, note in zip(samples, symbols, ann.aux_note)
        if sym == "+" and note.startswith("(")
    ]
    return Record(
        name=name,
        signal=signal,
        fs=float(rec.fs),
        lead=rec.sig_name[ch],
        beat_samples=samples[is_beat],
        beat_symbols=symbols[is_beat],
        rhythm=rhythm,
    )


def load_noise(name: str) -> np.ndarray:
    """Return channel 0 of an nstdb noise record (bw, ma or em) in millivolts at 360 Hz."""
    import wfdb

    if name not in NOISE_RECORDS:
        raise ValueError(f"unknown noise record {name!r}; choose from {sorted(NOISE_RECORDS)}")
    _require(NSTDB_DIR / f"{name}.hea")
    rec = wfdb.rdrecord(str(NSTDB_DIR / name))
    return rec.p_signal[:, 0].astype(np.float64)
