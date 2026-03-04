""" this is a script to prepare the dataframes for hmm analysis
it extracts sampling bout (+ a 2-3 second buffer before and after the relevant sampling bout)
it then splits the data into blocks of sampling bouts on which to run an hmm classifier. 
It saves the dataframes in the relevant derivative folder for each session.
it also creates df for each sampling bout per session."""

import numpy as np
import pandas as pd
from pathlib import Path


def load_dataframe(fpath):
    df = pd.read_csv(fpath)
    return df

def split_df(df, buffer_frames=150, merge_gap=75):
    if 'sampling' not in df.columns:
        raise AttributeError("DataFrame must contain 'sampling' column. your dataframe does not contain a sampling column")
    mask = df['sampling'].eq(1).to_numpy()
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError("No sampling bouts found in the DataFrame. are you sure you have annotated this session?")

    runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)

    merged = []
    start, end = runs[0][0], runs[0][-1]
    for r in runs[1:]:
        gap = r[0] - end - 1
        if gap < merge_gap:
            end = r[-1]
        else:
            merged.append((start, end))
            start, end = r[0], r[-1]
    merged.append((start, end))

    # extract buffered segments
    n = len(df)
    segments = []
    for s, e in merged:
        bs = max(0, s - buffer_frames)
        be = min(n - 1, e + buffer_frames)
        seg = df.iloc[bs:be + 1].copy()
        seg['orig_index'] = df.index.values[bs:be + 1]
        segments.append(seg)
    return segments
    