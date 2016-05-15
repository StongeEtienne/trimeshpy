#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import argparse
import logging
import math
import os
import time

import dipy.core.geometry as gm
import nibabel as nib
import numpy as np
from sys import stdout

from trimeshpy.trimesh_vtk import save_polydata
from trimeshpy.trimeshflow_vtk import lines_to_vtk_polydata

from scilpy.tracking.localTracking import _get_line
from scilpy.tracking.dataset import Dataset
# from scilpy.tracking.localTracking import track
from scilpy.tracking.mask import CMC, ACT
from scilpy.tracking.tools import get_max_angle_from_curvature
from scilpy.tracking.tracker import (probabilisticTracker,
                                     deterministicMaximaTracker)
from scilpy.tracking.trackingField import (SphericalHarmonicField, 
                                           TrackingDirection)


def buildArgsParser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Local Streamline HARDI Tractography. The tracking is done'
        + ' inside partial volume estimation maps and can use the particle '
        + 'filtering tractography (PFT) algorithm. See compute_pft_maps.py to '
        + 'generate PFT required maps. Streamlines greater than minL '
        + 'and shorter than maxL are outputted. The tracking direction is '
        + 'chosen in the aperture cone defined by the previous tracking '
        + 'direction and the angular constraint. The relation between theta '
        + "and the curvature is theta=2*arcsin(step_size/(2*R))."
        + " \n\nAlgo 'det': "
        + "the maxima of the spherical "
        + "function (SF) the most closely aligned to the previous direction. "
        + "\nAlgo 'prob': a "
        + 'direction drawn from the empirical distribution function defined '
        + 'from the SF. \nDefault parameters as in [1].',
        epilog='References: [1] Girard, G., Whittingstall K., Deriche, R., and '
        + 'Descoteaux, M. (2014). Towards quantitative connectivity analysis: '
        + 'reducing tractography biases. Neuroimage, 98, 266-278.')
    p._optionals.title = "Options and Parameters"

    p.add_argument(
        'sh_file', action='store', metavar='sh_file', type=str,
        help="Spherical harmonic file. Data must be aligned with \n"
        + "seed_file (isotropic resolution,nifti, see --basis).")
    p.add_argument(
        'seed_points', action='store', metavar='seed_points', type=str,
        help="Seed_points (nx3 numpy array).")
    p.add_argument(
        'seed_dir', action='store', metavar='seed_dir',   type=str,
        help="Seed_points direction/orientation (nx3 numpy array)")
    p.add_argument(
        'map_include_file', action='store', metavar='map_include_file', type=str,
        help="The probability map of ending the streamline and \nincluding it "
        + "in the output (CMC, PFT [1]). \n(isotropic resolution, nifti).")
    p.add_argument(
        'map_exclude_file', action='store', metavar='map_exclude_file', type=str,
        help="The probability map of ending the streamline and \nexcluding it "
        + "in the output (CMC, PFT [1]). \n(isotropic resolution, nifti).")
    p.add_argument(
        'output_file', action='store', metavar='output_file', type=str,
        help="Streamline output file (trk).")

    p.add_argument(
        '--basis', action='store', dest='basis', metavar='BASIS',
        default='dipy', type=str, choices=["mrtrix", "dipy"],
        help="Basis used for the spherical harmonic coefficients. "
        + "\n(must be 'mrtrix' or 'dipy'). [%(default)s]")
    p.add_argument(
        '--algo', dest='algo', action='store', metavar='ALGO', type=str,
        default='det', choices=['det', 'prob'],
        help="Algorithm to use (must be 'det' or 'prob')."
        + " [%(default)s]")
    p.add_argument(
        '-inv_seed_dir', action='store_true', default=False,
        help="inverse seed direction (if you input surface normal")

    seeding_group = p.add_mutually_exclusive_group()
    seeding_group.add_argument(
        '--npv', dest='npv', action='store',
        metavar='NBR', type=int,
        help='Number of seeds per voxel. [1]')
    seeding_group.add_argument(
        '--nt', dest='nt', action='store',
        metavar='NBR', type=int,
        help='Total number of seeds. Replaces --npv and --ns.')
    seeding_group.add_argument(
        '--ns', dest='ns', action='store',
        metavar='NBR', type=int,
        help='Number of streamlines to estimate. Replaces --npv and \n--nt. '
        + 'No multiprocessing used.')

    p.add_argument(
        '--skip', dest='skip', action='store',
        metavar='NBR', type=int,
        default=0, help='Skip the first NBR generated seeds / NBR seeds per '
        + '\nvoxel (--nt / --npv). Not working with --ns. [%(default)s]')
    p.add_argument(
        '--random', dest='random', action='store',
        metavar='RANDOM', type=int,
        default=0, help='Initial value for the random number generator.'
        + ' [%(default)s]')

    p.add_argument(
        '--step', dest='step_size', action='store',
        metavar='STEP', type=float, default=0.2,
        help='Step size in mm. [%(default)s]')

    deviation_angle_group = p.add_mutually_exclusive_group()
    deviation_angle_group.add_argument(
        '--theta', dest='theta', action='store',
        metavar='ANGLE', type=float,
        help="Maximum angle between 2 steps. ['det'=45, 'prob'=20]")
    deviation_angle_group.add_argument(
        '--curvature', dest='curvature', action='store',
        metavar='RAD', type=float,
        help='Minimum radius of curvature R in mm. Replaces --theta.')
    deviation_angle_group.add_argument(
        '--maxL_no_dir', dest='maxL_no_dir', action='store',
        metavar='MAX', type=float, default=1,
        help='Maximum length without valid direction, in mm. [%(default)s]')

    p.add_argument(
        '--sfthres', dest='sf_threshold', action='store',
        metavar='THRES', type=float, default=0.1,
        help='Spherical function relative threshold. [%(default)s]')
    p.add_argument(
        '--sfthres_init', dest='sf_threshold_init', action='store',
        metavar='THRES', type=float, default=0.5,
        help='Spherical function relative threshold value for the \ninitial '
        + 'direction. [%(default)s]')
    p.add_argument(
        '--minL', dest='min_length', action='store', metavar='MIN', type=float,
        default=10,
        help='Minimum length of a streamline in mm. [%(default)s]')
    p.add_argument(
        '--maxL', dest='max_length', action='store', metavar='MAX', type=int,
        default=300,
        help='Maximum length of a streamline in mm. [%(default)s]')

    p.add_argument(
        '--sh_interp', dest='field_interp', action='store',
        metavar='INTERP', type=str, default='tl', choices=['nn', 'tl'],
        help="Spherical harmonic interpolation: \n'nn' (nearest-neighbor) "
        + "or 'tl' (trilinear). [%(default)s]")
    p.add_argument(
        '--mask_interp', dest='mask_interp', action='store',
        metavar='INTERP', type=str, default='nn', choices=['nn', 'tl'],
        help="Mask interpolation: \n'nn' (nearest-neighbor) "
        + "or 'tl' (trilinear). [%(default)s]")

    p.add_argument(
        '--no_pft', dest='not_is_pft', action='store_true',
        help='If set, does not use the Particle Filtering \nTractography.')
    p.add_argument(
        '--particles', dest='nbr_particles', action='store',
        metavar='NBR', type=int, default=15,
        help='(PFT) Number of particles to use. [%(default)s]')
    p.add_argument(
        '--back', dest='back_tracking', action='store',
        metavar='BACK', type=float, default=2,
        help='(PFT) Length of back tracking in mm. [%(default)s]')
    p.add_argument(
        '--front', dest='front_tracking', action='store',
        metavar='FRONT', type=float, default=1,
        help='(PFT) Length of front tracking in mm. [%(default)s]')
    deviation_angle_pft_group = p.add_mutually_exclusive_group()
    deviation_angle_pft_group.add_argument(
        '--pft_theta', dest='pft_theta', action='store',
        metavar='ANGLE', type=float,
        help='(PFT) Maximum angle between 2 steps. [20]')
    deviation_angle_pft_group.add_argument(
        '--pft_curvature', dest='pft_curvature', action='store',
        metavar='RAD', type=float,
        help='(PFT) Minimum radius of curvature in mm. \nReplaces --pft_theta.')
    p.add_argument(
        '--pft_sfthres', dest='pft_sf_threshold', action='store',
        metavar='THRES', type=float, default=None,
        help='(PFT) Spherical function relative threshold. \nIf not set, '
        + '--sfthres value is used.')

    p.add_argument(
        '--all', dest='is_all', action='store_true',
        help='If set, keeps all generated streamlines.')
    p.add_argument(
        '--act', dest='is_act', action='store_true',
        help="If set, uses anatomically-constrained tractography (ACT)\n"
        + "instead of continuous map criterion (CMC).")

    p.add_argument(
        '--single_direction', dest='is_single_direction', action='store_true',
        help="If set, tracks in only one direction (forward or\nbackward) given"
        + " the initial seed. The direction is \nrandomly drawn from the ODF. "
        + "The seeding position is \nassumed to be a valid ending position "
        + "(included).")
    p.add_argument(
        '--processes', dest='nbr_processes', action='store', metavar='NBR',
        type=int, default=0,
        help='Number of sub processes to start. [cpu count]')
    p.add_argument(
        '--load_data', action='store_true', dest='isLoadData',
        help='If set, loads data in memory for all processes. \nIncreases the '
        + 'speed, and the memory requirements.')
    p.add_argument(
        '--tq', action='store_true', dest='outputTQ',
        help="If set, additionally outputs in the track querier \ntrk format "
        + "(TQ_'output_file').")
    p.add_argument(
        '-f', action='store_true', dest='isForce',
        help='If set, overwrites output file.')
    p.add_argument(
        '-v', action='store_true', dest='isVerbose',
        help='If set, produces verbose output.')
    
    p.add_argument('-test', type=int, default=None,
        help="Test compute local tracking with only 'N' first seed")
    
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
    parser = buildArgsParser()
    args = parser.parse_args()
    param = {}

    if args.pft_theta is None and args.pft_curvature is None:
        args.pft_theta = 20

    if not np.any([args.nt, args.npv, args.ns]):
        args.npv = 1

    if args.theta is not None:
        theta = gm.math.radians(args.theta)
    elif args.curvature > 0:
        theta = get_max_angle_from_curvature(args.curvature, args.step_size)
    elif args.algo == 'prob':
        theta = gm.math.radians(20)
    else:
        theta = gm.math.radians(45)

    if args.pft_curvature is not None:
        pft_theta = get_max_angle_from_curvature(args.pft_curvature, args.step_size)
    else:
        pft_theta = gm.math.radians(args.pft_theta)

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

    param['random'] = args.random
    param['skip'] = args.skip
    param['algo'] = args.algo
    param['mask_interp'] = mask_interpolation
    param['field_interp'] = field_interpolation
    param['theta'] = theta
    param['sf_threshold'] = args.sf_threshold
    param['pft_sf_threshold'] = args.pft_sf_threshold if args.pft_sf_threshold is not None else args.sf_threshold
    param['sf_threshold_init'] = args.sf_threshold_init
    param['step_size'] = args.step_size
    param['max_length'] = args.max_length
    param['min_length'] = args.min_length
    param['is_single_direction'] = args.is_single_direction
    param['nbr_seeds'] = args.nt if args.nt is not None else 0
    param['nbr_seeds_voxel'] = args.npv if args.npv is not None else 0
    param['nbr_streamlines'] = args.ns if args.ns is not None else 0
    param['max_no_dir'] = int(math.ceil(args.maxL_no_dir / param['step_size']))
    param['is_all'] = args.is_all
    param['is_act'] = args.is_act
    param['theta_pft'] = pft_theta
    if args.not_is_pft:
        param['nbr_particles'] = 0
        param['back_tracking'] = 0
        param['front_tracking'] = 0
    else:
        param['nbr_particles'] = args.nbr_particles
        param['back_tracking'] = int(
            math.ceil(args.back_tracking / args.step_size))
        param['front_tracking'] = int(
            math.ceil(args.front_tracking / args.step_size))
    param['nbr_iter'] = param['back_tracking'] + param['front_tracking']
    param['mmap_mode'] = None if args.isLoadData else 'r'

    if args.isVerbose:
        logging.basicConfig(level=logging.DEBUG)

    logging.debug('Tractography parameters:\n{0}'.format(param))

    if os.path.isfile(args.output_file):
        if args.isForce:
            logging.info('Overwriting "{0}".'.format(args.output_file))
        else:
            parser.error(
                '"{0}" already exists! Use -f to overwrite it.'
                .format(args.output_file))

    include_dataset = Dataset(
        nib.load(args.map_include_file), param['mask_interp'])
    exclude_dataset = Dataset(
        nib.load(args.map_exclude_file), param['mask_interp'])
    if param['is_act']:
        mask = ACT(include_dataset, exclude_dataset,
                   param['step_size'] / include_dataset.size[0])
    else:
        mask = CMC(include_dataset, exclude_dataset,
                   param['step_size'] / include_dataset.size[0])

    dataset = Dataset(nib.load(args.sh_file), param['field_interp'])
    field = SphericalHarmonicField(
        dataset, args.basis, param['sf_threshold'],
        param['sf_threshold_init'], param['theta'])

    if args.algo == 'det':
        tracker = deterministicMaximaTracker(field, param['step_size'])
    elif args.algo == 'prob':
        tracker = probabilisticTracker(field, param['step_size'])
    else:
        parser.error("--algo has wrong value. See the help (-h).")
        return

    pft_field = SphericalHarmonicField(
        dataset, args.basis, param['pft_sf_threshold'],
        param['sf_threshold_init'], param['theta_pft'])

    pft_tracker = probabilisticTracker(pft_field, param['step_size'])
    
    # ADD Seed input
    # modify ESO
    nib_mask = nib.load(args.map_include_file)
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
    print scale
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
    # end modify ESO
    
    
    # tracker to modify
    # modify ESO
    start = time.time()
    streamlines = []
    for i in range(nb_seeds):
        s = generate_streamline(tracker, mask, seed_points[i], seed_dirs[i], pft_tracker=pft_tracker, param=param)
        streamlines.append(s)
        stdout.write("\r %d%%" % (i*101//nb_seeds))
        stdout.flush()
    
    stdout.write("\n done")
    stdout.flush()
    stop = time.time()
    # end modify ESO

    
    # ADD save fiber output
    # modify ESO
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
    # end modify ESO

    lengths = [len(s) for s in streamlines]
    if nb_seeds > 0:
        ave_length = (sum(lengths) / nb_seeds) * param['step_size']
    else:
        ave_length = 0
    
    str_ave_length = "%.2f" % ave_length
    str_time = "%.2f" % (stop - start)
    print(str(nb_seeds) + " streamlines, with an average length of " +
          str_ave_length + " mm, done in " + str_time + " seconds.")

if __name__ == "__main__":
    main()
