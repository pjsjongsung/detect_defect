import sys
import os
from time import time

import numpy as np
import cv2 as cv
from dipy.io.image import save_nifti
from skimage.filters import rank
from skimage.morphology import disk
# Loading the passed arguments
# The correct way to run the script would be something in like
# python create_model_template.py "Z:\Source Files\Automontage Output Files\Python Output Wide Field\OutputFolder" csv_file_path
# or
# python create_model_template.py input_dir
csv_path = None
input_dir = sys.argv[1]
if len(sys.argv) > 2:
    csv_path = sys.argv[2]

if os.path.exists(input_dir) == False:
    raise Exception("The input directory does not exist!")

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

model_volume = None
s_idx = 0
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

    if model_volume is None:
        shape = image.shape
        model_volume = np.zeros(shape+(len(subj_paths),))

    model_volume[..., s_idx] = image
    s_idx += 1

model = np.zeros(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        for k in range(shape[2]):
            line = model_volume[i, j, k]
            try:
                model[i, j, k] = np.median(line[line!=0])
            except:
                model[i, j, k] = 0

model = np.nan_to_num(model)

save_nifti('model_volume.nii.gz', model, np.eye(4))