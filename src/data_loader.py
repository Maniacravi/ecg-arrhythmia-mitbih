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

def segment_record(rec_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''Given a record path (without extension), load the signal and annotation,
    loop over the R peaks and their corresponding symbols, slice out a window
    around each R peak, and return arrays for beats, labels, peaks, and RR features.'''
    
    # 1. load signal and annotation
    record = wfdb.rdrecord(rec_path)
    ann    = wfdb.rdann(rec_path, 'atr')
    signal = record.p_signal[:, 0]
    fs     = record.fs  # 360 Hz

    # 2. filter to valid beats only
    valid_peaks = [(r, s) for r, s in zip(ann.sample, ann.symbol)
                   if s in AAMI_MAP]

    # 3. compute mean RR interval for this record (patient-specific)
    all_rr  = [(valid_peaks[i+1][0] - valid_peaks[i][0]) / fs
               for i in range(len(valid_peaks) - 1)]
    mean_rr = np.mean(all_rr)

    # 4. loop over valid peaks, skip first and last (no prev/next)
    beats, labels, peaks, rr_features = [], [], [], []

    for i, (r, s) in enumerate(valid_peaks):
        if i == 0 or i == len(valid_peaks) - 1:
            continue

        start, end = r - WINDOW, r + WINDOW
        if start < 0 or end > len(signal):
            continue

        prev_r  = valid_peaks[i-1][0]
        next_r  = valid_peaks[i+1][0]
        pre_rr  = (r - prev_r) / fs
        post_rr = (next_r - r) / fs
        ratio   = pre_rr / mean_rr

        beats.append(signal[start:end])
        labels.append(CLASS_TO_IDX[AAMI_MAP[s]])
        peaks.append(r)
        rr_features.append([pre_rr, post_rr, ratio])

    return (np.array(beats,       dtype=np.float32),
            np.array(labels,      dtype=np.int64),
            np.array(peaks,       dtype=np.int64),
            np.array(rr_features, dtype=np.float32))


def load_dataset(data_dir: str):
    ''' Load and segment records from DS1 and DS2, return train/test splits.'''
    train_beats, train_labels, train_peaks, train_rr = [], [], [], []
    test_beats,  test_labels,  test_peaks,  test_rr  = [], [], [], []

    for rec_id in DS1:
        beats, labels, peaks, rr = segment_record(str(Path(data_dir) / rec_id))
        train_beats.append(beats)
        train_labels.append(labels)
        train_peaks.append(peaks)
        train_rr.append(rr)

    for rec_id in DS2:
        beats, labels, peaks, rr = segment_record(str(Path(data_dir) / rec_id))
        test_beats.append(beats)
        test_labels.append(labels)
        test_peaks.append(peaks)
        test_rr.append(rr)

    return (
        np.concatenate(train_beats),  np.concatenate(train_labels),
        np.concatenate(train_peaks),  np.concatenate(train_rr),
        np.concatenate(test_beats),   np.concatenate(test_labels),
        np.concatenate(test_peaks),   np.concatenate(test_rr)
    )