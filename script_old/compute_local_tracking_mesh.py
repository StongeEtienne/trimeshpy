#!/usr/bin/env python

from __future__ import division

import argparse
import logging
import math
import os
import time
from sys import stdout

import dipy.core.geometry as gm
import nibabel as nib
import numpy as np

from trimeshpy.trimesh_vtk import save_polydata
from trimeshpy.trimeshflow_vtk import lines_to_vtk_polydata

from scilpy.tracking.dataset import Dataset
from scilpy.tracking.localTracking import _get_line
from scilpy.tracking.mask import BinaryMask
from scilpy.tracking.tracker import probabilisticTracker, deterministicMaximaTracker
from scilpy.tracking.trackingField import SphericalHarmonicField, TrackingDirection
from scilpy.tracking.tools import get_max_angle_from_curvature


def buildArgsParser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Local Streamline HARDI Tractography. The tracking is done'
        + ' inside a binary mask. Streamlines greater than minL and shorter '
        + 'than maxL are outputted. The tracking direction is chosen in the '
        + 'aperture cone defined by the previous tracking direction and the '
        + 'angular constraint. The relation between theta and the curvature '
        + "is theta=2*arcsin(step_size/(2*R)). \n\nAlgo 'det': the maxima "
        + "of the spherical "
        + "function (SF) closest aligned to the previous direction."
        + "\nAlgo 'prob': a "
        + 'direction drawn from the empirical distribution function defined '
        + 'from the SF. \nDefault parameters as in [1].',
        epilog='References: [1] Girard, G., Whittingstall K., Deriche, R., and '
        + 'Descoteaux, M. (2014). Towards quantitative connectivity analysis: '
        + 'reducing tractography biases. Neuroimage, 98, 266-278.')
    p._optionals.title = "Options and Parameters"
    
    p.add_argument(
        'sh_file', action='store', metavar='sh_file',   type=str,
        help="Spherical Harmonic file. Data must be aligned with seed_file and "
        + "must have an isotropic resolution. (nifti, see --basis).")
    p.add_argument(
        'seed_points', action='store', metavar='seed_points',   type=str,
        help="Seed_points (nx3 numpy array).")
    p.add_argument(
        'seed_dir', action='store', metavar='seed_dir',   type=str,
        help="Seed_points direction/orientation (nx3 numpy array)")
    p.add_argument(
        'mask_file', action='store', metavar='mask_file', type=str,
        help="Tracking mask (isotropic resolution, nifti).")
    p.add_argument(
        'output_file', action='store', metavar='output_file', type=str,
        help="Streamline output file (.fib).")
    
    p.add_argument(
        '--basis', action='store', metavar='BASIS', dest='basis', default='dipy', type=str,
        help="Basis used for the spherical harmonic coefficients. "
        + "Must be 'mrtrix' or 'dipy'. [%(default)s]")
    p.add_argument(
        '--algo', dest='algo', action='store', metavar='ALGO', type=str, default='det',
        help="Tracking Algorithm: 'det' or 'prob'. [%(default)s]")
    
    p.add_argument(
        '--step', dest='step_size', action='store', metavar='STEP', type=float, default=0.2,
        help='Step size in mm. [%(default)s]')
    p.add_argument(
        '-inv_seed_dir', action='store_true', default=False,
        help="inverse seed direction (if you input surface normal")
    
    deviation_angle_group = p.add_mutually_exclusive_group()
    deviation_angle_group.add_argument(
        '--theta', dest='theta', action='store', metavar='ANGLE', type=float,
        help="Maximum angle between 2 steps. ['det'=45, 'prob'=20]")
    deviation_angle_group.add_argument(
        '--curvature', dest='curvature', action='store', metavar='RADIUS', type=float,
        help='Minimum radius of curvature R in mm. Replaces --theta.')
    p.add_argument(
        '--maxL_no_dir', dest='maxL_no_dir', action='store', metavar='MAX', type=float, default=1,
        help='Maximum length without valid direction, in mm. [%(default)s]')
    
    p.add_argument(
        '--sfthres', dest='sf_threshold', action='store', metavar='THRES', type=float, default=0.1,
        help='Spherical function relative threshold. [%(default)s]')
    p.add_argument(
        '--sfthres_init', dest='sf_threshold_init', action='store', metavar='THRES', type=float, default=0.5,
        help='Spherical function relative threshold value for the initial direction. [%(default)s]')
    p.add_argument(
        '--minL', dest='min_length', action='store', metavar='MIN', type=float, default=10,
        help='Minimum length of a streamline in mm. [%(default)s]')
    p.add_argument(
        '--maxL', dest='max_length', action='store', metavar='MAX', type=int, default=300,
        help='Maximum length of a streamline in mm. [%(default)s]')
    
    p.add_argument(
        '--sh_interp', dest='field_interp', action='store', metavar='INTERP', type=str, default='tl',
        help="Spherical harmonic interpolation: 'nn' (nearest-neighbor) or 'tl' (trilinear). [%(default)s]")
    p.add_argument(
        '--mask_interp', dest='mask_interp', action='store', metavar='INTERP', type=str, default='nn',
        help="Mask interpolation: 'nn' (nearest-neighbor) or 'tl' (trilinear). [%(default)s]")
    
    p.add_argument('-test', type=int, default=None,
        help="Test compute local tracking with only 'N' first seed")
    
    #p.add_argument(
    #    '--processes', dest='nbr_processes', action='store', metavar='NBR', type=int, default=0,
    #    help='Number of sub processes to start. [cpu count]')
    
    p.add_argument('--tq', action='store_true', dest='outputTQ',
        help="Additionally output in the track querier trk format (TQ_'output_file'). [%(default)s]")
    
    p.add_argument('-f', action='store_true', dest='isForce', default=False,
        help='Force (overwrite output file). [%(default)s]')
    
    p.add_argument('-v', action='store_true', dest='isVerbose', default=False,
        help='Produce verbose output. [%(default)s]')
    return p


def generate_streamline(tracker, mask, position, direction, pft_tracker, param):

    _pos = position
    _dir = direction
    _dir /= np.linalg.norm(_dir)
    
    new_dir = get_best_peak_dir(tracker, _dir)
    tracker.initialize(_pos, new_dir)

    s = _get_line(tracker, mask, pft_tracker, param, is_forward=True)
    return s

def get_best_peak_dir(tracker, direction):
    # find closest peak direction
    min_cos = -100
    new_dir_ind = None
    
    for i in range(len(tracker.trackingField.dirs)):
        d = tracker.trackingField.dirs[i]
        new_cos = np.dot(direction, d)
        if new_cos > min_cos:
            min_cos = new_cos
            new_dir_ind = i     
    return TrackingDirection(tracker.trackingField.dirs[new_dir_ind], new_dir_ind)

def main():
    np.random.seed(int(time.time()))
    parser = buildArgsParser()
    args = parser.parse_args()

    param = {}
    
    if args.algo not in ["det", "prob"]:
        parser.error("--algo has wrong value. See the help (-h).")
    
    if args.basis not in ["mrtrix", "dipy", "fibernav"]:
        parser.error("--basis has wrong value. See the help (-h).")
    
    #if np.all([args.nt is None, args.npv is None, args.ns is None]):
    #    args.npv = 1
    
    if args.theta is not None:
        theta = gm.math.radians(args.theta)
    elif args.curvature > 0:
        theta = get_max_angle_from_curvature(args.curvature, args.step_size)
    elif args.algo == 'prob':
        theta = gm.math.radians(20)
    else:
        theta = gm.math.radians(45)
    
    if args.mask_interp == 'nn':
        mask_interpolation = 'nearest'
    elif args.mask_interp == 'tl':
        mask_interpolation = 'trilinear'
    else:
        parser.error("--mask_interp has wrong value. See the help (-h).")
        return
    
    if args.field_interp == 'nn':
        field_interpolation = 'nearest'
    elif args.field_interp == 'tl':
        field_interpolation = 'trilinear'
    else:
        parser.error("--sh_interp has wrong value. See the help (-h).")
        return
    
    param['algo'] = args.algo
    param['mask_interp'] = mask_interpolation
    param['field_interp'] = field_interpolation
    param['theta'] = theta
    param['sf_threshold'] = args.sf_threshold
    param['sf_threshold_init'] = args.sf_threshold_init
    param['step_size'] = args.step_size
    param['max_length'] = args.max_length
    param['min_length'] = args.min_length
    param['is_single_direction'] = False
    param['nbr_seeds'] = 0
    param['nbr_seeds_voxel'] = 0
    param['nbr_streamlines'] = 0
    param['max_no_dir'] = int(math.ceil(args.maxL_no_dir / param['step_size']))
    param['is_all'] = False
    param['isVerbose'] = args.isVerbose
    
    if param['isVerbose']:
        logging.basicConfig(level=logging.DEBUG)
    
    if param['isVerbose']:
        logging.info('Tractography parameters:\n{0}'.format(param))
    
    if os.path.isfile(args.output_file):
        if args.isForce:
            logging.info('Overwriting "{0}".'.format(args.output_file))
        else:
            parser.error(
                '"{0}" already exists! Use -f to overwrite it.'
                .format(args.output_file))
    
    nib_mask = nib.load(args.mask_file)
    mask = BinaryMask(
        Dataset(nib_mask, param['mask_interp']))
    
    dataset = Dataset(nib.load(args.sh_file), param['field_interp'])
    field = SphericalHarmonicField(
        dataset, args.basis, param['sf_threshold'], param['sf_threshold_init'], param['theta'])
    
    if args.algo == 'det':
        tracker = deterministicMaximaTracker(field, param['step_size'])
    elif args.algo == 'prob':
        tracker = probabilisticTracker(field, param['step_size'])
    else:
        parser.error("--algo has wrong value. See the help (-h).")
        return
    
    start = time.time()
    
    # Etienne St-Onge
    #load and transfo *** todo test with rotation and scaling
    seed_points = np.load(args.seed_points)
    seed_dirs = np.load(args.seed_dir)
    rotation = nib_mask.get_affine()[:3,:3]
    inv_rotation = np.linalg.inv(rotation)
    translation = nib_mask.get_affine()[:3,3]
    scale = np.array(nib_mask.get_header().get_zooms())
    voxel_space = nib.aff2axcodes(nib_mask.get_affine())
    
    print voxel_space
    # seed points transfo
    # LPS -> voxel_space
    if voxel_space[0] != 'L':
        print "flip X"
        seed_points[:,0] = -seed_points[:,0]
    if voxel_space[1] != 'P':
        print "flip Y"
        seed_points[:,1] = -seed_points[:,1]
    if voxel_space[2] != 'S':
        print "flip Z"
        seed_points[:,2] = -seed_points[:,2]
    
    # other transfo
    seed_points = seed_points - translation
    seed_points = seed_points.dot(inv_rotation)
    seed_points = seed_points * scale
    
    # seed dir transfo
    seed_dirs[:,0:2] = -seed_dirs[:,0:2]
    seed_dirs = seed_dirs.dot(inv_rotation)
    seed_dirs = seed_dirs * scale
    
    if args.inv_seed_dir:
        seed_dirs = seed_dirs * -1.0
    
    # Compute tractography
    nb_seeds = len(seed_dirs)
    if args.test is not None and args.test < nb_seeds:
        nb_seeds = args.test
    
    print args.algo," nb seeds: ", nb_seeds
    
    streamlines = []
    for i in range(nb_seeds):
        s = generate_streamline(tracker, mask, seed_points[i], seed_dirs[i], pft_tracker=None, param=param)
        streamlines.append(s)
        
        stdout.write("\r %d%%" % (i*101//nb_seeds))
        stdout.flush()
    stdout.write("\n done")
    stdout.flush()
    
    # transform back
    for i in range(len(streamlines)):
        streamlines[i] = streamlines[i] / scale
        streamlines[i] = streamlines[i].dot(rotation)
        streamlines[i] = streamlines[i] + translation
        # voxel_space -> LPS
        if voxel_space[0] != 'L':
            streamlines[i][:,0] = -streamlines[i][:,0]
        if voxel_space[1] != 'P':
            streamlines[i][:,1] = -streamlines[i][:,1]
        if voxel_space[2] != 'S':
            streamlines[i][:,2] = -streamlines[i][:,2]
    
    lines_polydata = lines_to_vtk_polydata(streamlines, None, np.float32)
    save_polydata(lines_polydata, args.output_file , True)
    
    lengths = [len(s) for s in streamlines]
    if nb_seeds > 0:
        ave_length = (sum(lengths) / nb_seeds) * param['step_size']
    else:
        ave_length = 0
    
    str_ave_length = "%.2f" % ave_length
    str_time = "%.2f" % (time.time() - start)
    print(str(nb_seeds) + " streamlines, with an average length of " +
          str_ave_length + " mm, done in " + str_time + " seconds.")

if __name__ == "__main__":
    main()
