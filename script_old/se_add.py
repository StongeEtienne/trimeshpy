#!/usr/bin/env python

# Etienne.St-Onge@usherbrooke.ca

import argparse
import nibabel as nib

# use : 
# fsl5.0-fast -g  T1.nii
# python se_add.py T1_seg_1.nii  T1_seg_2.nii
#                  (gray matter) (white matter)

parser = argparse.ArgumentParser(description='add or mult for mask')
parser.add_argument('nii1', type=str, default=None, help='nifti 1')
parser.add_argument('nii2', type=str, default=None, help='nifti 2')
parser.add_argument('out', type=str, default=None, help='out nifti')

parser.add_argument('--mult', action='store_true', default=False, help='multiplication instead of addition')

args = parser.parse_args()

nii1 = nib.load(args.nii1)
nii2 = nib.load(args.nii2)

if args.mult :
    data_out = nii1.get_data() * nii2.get_data()
else :
    data_out = nii1.get_data() + nii2.get_data()

nii_out = nib.Nifti1Image(data_out, nii1.get_affine(), nii1.get_header())

nib.save(nii_out, args.out)
