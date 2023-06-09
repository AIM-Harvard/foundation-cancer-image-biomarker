#!/usr/bin/env python
# Modified from original source on Fri Feb 11 2022
# Copyright (c) 2022 Suraj Pai
# AIM Harvard

"""
Ke Yan
Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
National Institutes of Health Clinical Center
May 2018
THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

"""
A simple demo to load 2D 16-bit slices from DeepLesion and save to 3D nifti volumes.
The nifti volumes can be viewed in software such as 3D slicer and ITK-SNAP.
"""
import numpy as np
import nibabel as nib
from functools import partial
import os
import cv2
import csv
from multiprocessing import Pool
import pandas as pd

FMT = "%s_%03d-%03d.nii.gz"  # format of the nifti file name to output


def slices2nifti(ims, fn_out, spacing):
    """save 2D slices to 3D nifti file considering the spacing"""
    if len(ims) < 300:  # cv2.merge does not support too many channels
        V = cv2.merge(ims)
    else:
        V = np.empty((ims[0].shape[0], ims[0].shape[1], len(ims)))
        for i in range(len(ims)):
            V[:, :, i] = ims[i]

    # the transformation matrix suitable for 3D slicer and ITK-SNAP
    T = np.array(
        [
            [0, -spacing[1], 0, 0],
            [-spacing[0], 0, 0, 0],
            [0, 0, -spacing[2], 0],
            [0, 0, 0, 1],
        ]
    )
    img = nib.Nifti1Image(V, T)
    nib.save(img, fn_out)
    print(f"Saved: {fn_out}")


def load_slices(load_dir, slice_idxs):
    """load slices from 16-bit png files"""
    slice_idxs = np.array(slice_idxs)
    assert np.all(slice_idxs[1:] - slice_idxs[:-1] == 1)
    ims = []
    for slice_idx in slice_idxs:
        fn = "%03d.png" % slice_idx
        path = os.path.join(load_dir, fn)
        im = cv2.imread(path, -1)  # -1 is needed for 16-bit image
        assert im is not None, "error reading %s" % path
        # the 16-bit png file has a intensity bias of 32768
        ims.append((im.astype(np.int32) - 32768).astype(np.int16))
    return ims


def read_DL_info(info_fn):
    """read spacings and image indices in DeepLesion"""
    df = pd.read_csv(info_fn)
    patient_info = [
        (i, j, k)
        for i, j, k in zip(
            df["Patient_index"].values, df["Study_index"].values, df["Series_ID"].values
        )
    ]
    spacings = (
        df["Spacing_mm_px_"].apply(lambda x: [float(d) for d in x.split(",")]).values
    )

    return patient_info, spacings


def main(args):
    patient_idxs, spacings = read_DL_info(args.info_fn)

    dir_out = args.output_dir
    dir_in = args.png_dir

    if not os.path.exists(dir_out):
        os.mkdir(dir_out)

    img_dirs = os.listdir(dir_in)
    img_dirs.sort()

    with Pool(args.n_cores) as p:
        p.map(
            partial(
                save_dir_to_nifti,
                idxs=patient_idxs,
                spacings=spacings,
                dir_in=dir_in,
                dir_out=dir_out,
            ),
            img_dirs,
        )


def save_dir_to_nifti(dir1, idxs, spacings, dir_in, dir_out):
    # find the image info according to the folder's name
    idxs1 = np.array([int(d) for d in dir1.split("_")])
    i1 = np.where(np.all(idxs == idxs1, axis=1))[0]
    spacings1 = spacings[i1[0]]

    fns = os.listdir(os.path.join(dir_in, dir1))
    slices = [int(d[:-4]) for d in fns if d.endswith(".png")]
    slices.sort()

    # Each folder contains png slices from one series (volume)
    # There may be several sub-volumes in each volume depending on the key slices
    # We group the slices into sub-volumes according to continuity of the slice indices
    groups = []
    for slice_idx in slices:
        if len(groups) != 0 and slice_idx == groups[-1][-1] + 1:
            groups[-1].append(slice_idx)
        else:
            groups.append([slice_idx])

    for group in groups:
        # group contains slices indices of a sub-volume
        load_dir = os.path.join(dir_in, dir1)
        ims = load_slices(load_dir, group)
        fn_out = FMT % (dir1, group[0], group[-1])

        fn_out = os.path.join(dir_out, fn_out)
        slices2nifti(ims, fn_out, spacings1)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "png_dir", help="Path to directory with png images from DeepLesion"
    )
    parser.add_argument(
        "output_dir", help="Path to directory where processed images are stored"
    )
    parser.add_argument("info_fn", help="Path to info file")
    parser.add_argument(
        "--n_cores", help="Multiprocessing cores for parallel save", type=int, default=4
    )

    args = parser.parse_args()

    main(args)
