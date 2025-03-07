# detect_defect
This repository provides code for detecting defects in the nerve fiber layer in wide field images using registration to template.

The code requires follow packages to run

```
numpy
scipy
opencv-python
scikit-image
matplotlib
dipy
```

Details on the code are included in the comments of the code, but basically you would want to do something similar to below
1. `python create_model_template.py input_dir` to create model_volume.nii.gz
2. `python extract_defect.py -s 24 -e 84 -b 0.5 -d input_dir output_dir` to run the analysis

model_volume.nii.gz has been included for reference.

The manuscript regarding the code will be available soon. Please email pjsjongsung@gmail.com or create an issue for any questions regarding the code.