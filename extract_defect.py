# Importing packages
import sys
import os
import argparse
from time import time

import numpy as np
import cv2 as cv

from dipy.io.image import load_nifti, save_nifti
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric

from dipy.align.imaffine import (MutualInformationMetric,
                                 AffineRegistration)

from skimage.filters import rank, threshold_multiotsu
from skimage.morphology import disk

import matplotlib.pyplot as plt

from dipy.align import (affine_registration, translation,
                        rigid, affine, register_series, rigid_scaling)

from scipy.ndimage import affine_transform


# function for affine registration
def affine_registration_3d(moving, static):
    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)

    # small number of iterations for this example
    level_iters = [10, 10]
    sigmas = [1.0, 0.0]
    factors = [2, 1]

    pipeline = [rigid_scaling]

    xformed_img, reg_affine = affine_registration(
        moving,
        static,
        moving_affine=np.eye(4),
        static_affine=np.eye(4),
        nbins=32,
        metric='MI',
        pipeline=pipeline,
        level_iters=level_iters,
        sigmas=sigmas,
        factors=factors)
    
    return xformed_img


# loading model file
# The model file was created through this list of scans
# More detail can be found in how the model_volume.nii.gz can be created
# through the manuscript.
model, _ = load_nifti('model_volume.nii.gz')
shape = model.shape
model = model / np.max(model)
model = np.nan_to_num(model)

# Loading the passed arguments
# The correct way to run the script would be something in this line
# python extract_defect.py "Z:\Source Files\Automontage Output Files\Python Output Wide Field\OutputFolder" csv_file_path -o output_dir_path -s 24 -e 84 -b 0.4 -d 
# or
# python extract_defect.py -s 24 -e 84 -b 0.5 -d input_dir output_dir
# -o selects the output directory, -s the starting slab, -e the last slab and -b a threshold used in the method
# -d will add an option to use thickness image when binarizing
# lower threshold will increase predicted regions
# if csv is not provided input_dir_path should have sub directories inside
# named 'subject number'_'scan number'_'which eye' (e.g. 1022_1_OS)
# The sub directory should contain ILM images with names accordingly
# 'sub directory name'_ILMimage_'depth'+'.jpg or .png'
# and depth should be a range from 0 to 160 at least
# e.g. 1022_1_OS_ILMimage_0.jpg ~ 1022_1_OS_ILMimage_160.jpg

parser = argparse.ArgumentParser(
                    prog='Extract damage by registration',
                    description='The damaged region of the wide field image' +
                                'is selected as a difference between' +
                                'registered image and the model',
                    epilog='contact pjsjongsung@gmail.com for help')

parser.add_argument('input_dir_path')
parser.add_argument('csv_file_path')

parser.add_argument('-o', '--output_dir_path', required=False,
                    default=None)

parser.add_argument('-s', '--slab_start',
                    default=24)
parser.add_argument('-e', '--slab_end',
                    default=80)
parser.add_argument('-b', '--bin_th',
                    default=0.5)
parser.add_argument('-d', '--depth_th',
                    action='store_true')


args = parser.parse_args()

input_dir = args.input_dir_path
csv_path = args.csv_file_path
slab_start = int(args.slab_start)//4
slab_end = int(args.slab_end)//4+1
bin_th = float(args.bin_th)
depth_th = args.depth_th

if args.output_dir_path is None:
    output_dir = csv_path
    csv_path = None
else:
    output_dir = args.output_dir_path

if os.path.exists(input_dir) == False:
    raise Exception("The input directory does not exist!")

if os.path.exists(output_dir) == False:
    os.makedirs(output_dir)

if csv_path is None:
    subj_list = os.listdir(input_dir)
    subj_paths = [os.path.join(input_dir, s) for s in subj_list]

else:
    subj_list = []
    subj_paths = []
    import csv
    with open(csv_path, mode='r') as csvfile:
        csvreader = csv.reader(csvfile)
        for idx, row in enumerate(csvreader):
            if idx == 0:
                continue
            subj_id = row[0]
            image_n = row[1]
            image_id = row[2]
            eye = row[3]
            subj_list.append(subj_id+'_'+image_n+'_'+eye)
            subj_paths.append(os.path.join(input_dir, 'ID_'+subj_id+'_'+image_n, eye, 'AC_'+image_id))


# metric to be used in registration
metric = CCMetric(3, radius=3)

# Iterate over all sub directories    
for sub_path, sub_dir in zip(subj_paths, subj_list):
    s_t = time()

    # Note that we are using a fixed shape here
    # The numbers might have to change if applied on a different site
    # As long as there is sufficient depth to normalize the raw data
    # it should be fine.
    volume = []

    # Load and normalize/blur volume
    for i in range(0, 164, 4):
        try:
            image = cv.imread(os.path.join(sub_path, sub_dir+'_NormalizedILM_'+str(i)+'.jpg'), 0)
        except:
            image = cv.imread(os.path.join(sub_path, sub_dir+'_NormalizedILM_'+str(i)+'.png'), 0)
        volume.append(image)
    
    volume = np.stack(volume, axis=-1)
    for i in range(0, 164, 4):
        image = volume[..., i//4]
        image = rank.mean_percentile(image, footprint=disk(7), p0=.1, p1=.9)
        volume[..., i//4] = image
    if sub_dir.endswith('OD'):
        volume = np.flip(volume, axis=1)
    image = volume / np.max(volume)
    image = np.nan_to_num(image)
    shape = image.shape


    # Slicing and binarizing the image
    # As a target for registration
    image_slice = image[..., slab_start:slab_end]
    model_slice = model[..., slab_start:slab_end]

    ths = threshold_multiotsu(model_slice)
    th_model = np.where(np.all([model_slice < ths[1], model_slice>ths[0]], axis=0), 1, 0)
    if depth_th == False:
        ths = threshold_multiotsu(image_slice)
        th_image = np.where(np.all([image_slice < ths[1], image_slice>ths[0]], axis=0), 1, 0)    
    else:
        thickness_map = cv.imread(os.path.join(sub_path, sub_dir+'_Thickness.jpg'), 0)[:image.shape[0], :image.shape[1]] / 8
        th_image = np.zeros(th_model.shape)
        for i in range(th_image.shape[-1]):
            th_image[..., i] = np.where(thickness_map <= (i+slab_start), 1, 0)
        if sub_dir.endswith('OD'):
            th_image = np.flip(th_image, axis=1)


    # Registration step
    # [20, 20] stands for number of iterations per scale
    # Since this is a iterative registration method
    sym = SymmetricDiffeomorphicRegistration(metric, [20, 20])

    # This calculates the affine transform
    th_image = affine_registration_3d(th_image, th_model)

    dmap = sym.optimize(th_model, th_image)
    deformed_image = dmap.transform(th_image)

    # We average out the image
    c_image = np.zeros(shape[:2])
    c_image = np.mean(np.abs(deformed_image - th_model), axis=-1)

    # thresholding the image to make it serve as a
    # prior for cutting out unwanted predictions
    image_slice = np.min(image[..., 6:21], axis=-1)
    hist, bins = np.histogram(image_slice[image_slice!=0], bins=255)
    ths2 = threshold_multiotsu(classes=4, hist=(hist, bins[1:]))

    # Final output created using binarized registration results
    # and masking them by the thresholded image
    final_output = np.where(c_image>bin_th, 1, 0) * np.where(image_slice <= ths2[1], 1, 0)

    # flip back if it was OD
    if sub_dir.endswith('OD'):
        final_output = np.flip(final_output, axis=1)

    # Saved to a numpy file to make it easier to load
    # To load the numpy array, one can just
    # np.load('1022_1_OS.npz')['pred']
    np.savez(os.path.join(output_dir, sub_dir+'.npz'), pred=final_output.astype(np.int16))

    # Saving the image as well for preview
    cv.imwrite(os.path.join(output_dir, sub_dir+'.png'), np.clip(np.round(final_output*255), 0, 255).astype(np.uint8))

    print(time()-s_t)