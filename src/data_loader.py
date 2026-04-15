import numpy as np
import wfdb
from pathlib import Path

WINDOW = 180

AAMI_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q',
}

CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

DS1 = ['101','106','108','109','112','114','115','116','118','119',
       '122','124','201','203','205','207','208','209','215','220','223','230']

DS2 = ['100','103','105','111','113','117','121','123','200','202',
       '210','212','213','214','219','221','222','228','231','232','233','234']

EXCLUDED = ['102', '104', '107', '217']

def segment_record(rec_path: str) -> tuple[np.ndarray, np.ndarray]:
    ''' Given a record path (without extension), load the signal and annotation,
        loop over the R peaks and their corresponding symbols, slice out a window
        around each R peak, and return two numpy arrays: one for the segmented beats
        and one for the corresponding labels.'''
    # 1. load signal and annotation
    record = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, 'atr')
    signal = record.p_signal[:, 0]  # use only the first channel (MLII)
    r_peaks = ann.sample
    symbols = ann.symbol

    # 2. loop over r-peaks and symbols
    beats = []
    labels = []
    for r, s in zip(r_peaks, symbols):
    # 3. skip if symbol not in AAMI_MAP
        if s not in AAMI_MAP:
            continue
        label = CLASS_TO_IDX[AAMI_MAP[s]]
        start = r - WINDOW
        end = r + WINDOW
    # 4. skip if window goes out of bounds
        if start < 0 or end > len(signal):
            continue
    # 5. slice the window, append to list
        beat = signal[start:end]
        beats.append(beat)
        labels.append(label)
    # return np.array of beats, np.array of labels
    return np.array(beats, dtype=np.float32), np.array(labels, dtype=np.int64)


def load_dataset(data_dir: str):
    ''' Load and segment records from DS1 and DS2, return train/test splits.'''
    train_beats, train_labels = [], []
    test_beats,  test_labels  = [], []

    for rec_id in DS1:
        beats, labels = segment_record(str(Path(data_dir) / rec_id))
        # append to train lists
        train_beats.append(beats)
        train_labels.append(labels)

    for rec_id in DS2:
        beats, labels = segment_record(str(Path(data_dir) / rec_id))
        # append to test lists
        test_beats.append(beats)
        test_labels.append(labels)

    return (
        np.concatenate(train_beats), np.concatenate(train_labels),
        np.concatenate(test_beats),  np.concatenate(test_labels)
    )