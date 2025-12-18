import json
import math
import os

import h5py
import hdf5plugin
import numpy as np


def read_file(path):
    return h5py.File(
        path,
        "r",
    )


def get_min_and_max_mass_to_charge(f):
    mz_min, mz_max = None, None
    for spectrum_id in f.keys():
        mz = f[str(spectrum_id)]["x"][:]
        if mz_min is None or mz_max is None:
            mz_min, mz_max = mz.min(), mz.max()
        else:
            mz_min = min(mz_min, mz.min())
            mz_max = max(mz_max, mz.max())
    return mz_min, mz_max


def bin_spectra(f, sorted_spectrum_ids, bins, output_file):
    bin_centers = (bins[:-1] + bins[1:]) / 2

    with h5py.File(output_file, "w") as out_f:
        for spectrum_id in sorted_spectrum_ids:
            print(f"Running for spectrum {spectrum_id}...")
            mz = f[str(spectrum_id)]["x"][:]
            intensities = f[str(spectrum_id)]["y"][:]

            # Normalize intensities using TIC
            tic = intensities.sum()
            if tic > 0:
                intensities = intensities / tic

            binned, _ = np.histogram(mz, bins=bins, weights=intensities)

            grp = out_f.create_group(str(spectrum_id))
            grp.create_dataset("x", data=bin_centers, dtype=np.float32)
            grp.create_dataset("y", data=binned, dtype=np.float32)


def save_hyperparameters(hparams, assets_path, run_id):
    hyperparameters_path = os.path.join(assets_path, "hyperparameters")
    os.makedirs(hyperparameters_path, exist_ok=True)

    with open(
        os.path.join(hyperparameters_path, f"{run_id}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(hparams, f, indent=4, default=str)


def load_hyperparameters(assets_path, run_id):
    hyperparameters_path = os.path.join(assets_path, "hyperparameters")

    with open(
        os.path.join(hyperparameters_path, f"{run_id}.json"), "r", encoding="utf-8"
    ) as f:
        hyperparameters = json.load(f)

    return hyperparameters
