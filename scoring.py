import nibabel as nib
from nibabel.processing import resample_from_to 
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import json


# Import the SynthSeg module and save the directories into the path for further reference
synthseg_home = os.path.dirname('Models/SynthSeg/')
sys.path.append(synthseg_home)
model_dir = 'Models/SynthSeg/models'
labels_dir = 'Models/SynthSeg/data/labels_classes_priors'
from Models.SynthSeg.SynthSeg.predict_synthseg import predict as atlas_segmentation


# Function to call SynthSeg
def call_synthseg(input_file, output_path):
    # Define the arguments for SynthSeg

    args = {}
    args['robust'] = False
    args['parc'] = True
    args['fast'] = False
    args['post'] = False
    args['crop'] = 198
    args['ct'] = False
    args['resample'] = None
    args['vol'] = None
    args['qc'] = None
    args['post'] = None
    if args['robust']:
        args['path_model_segmentation'] = os.path.join(model_dir, 'synthseg_robust_2.0.h5')
        args['fast'] = True
    else:
        args['path_model_segmentation'] = (model_dir + '/synthseg_2.0.h5') #os.path.join(model_dir, 'synthseg_1.0.h5')
        
    args['path_model_parcellation'] = os.path.join(model_dir, 'synthseg_parc_2.0.h5')
    args['path_model_qc'] = os.path.join(model_dir, 'synthseg_qc_2.0.h5')
    args['labels_segmentation'] = os.path.join(labels_dir, 'synthseg_segmentation_labels_2.0.npy')
    args['labels_denoiser'] = os.path.join(labels_dir, 'synthseg_denoiser_labels_2.0.npy')
    args['labels_parcellation'] = os.path.join(labels_dir, 'synthseg_parcellation_labels.npy')
    args['labels_qc'] = os.path.join(labels_dir, 'synthseg_qc_labels_2.0.npy')
    args['names_segmentation_labels'] = os.path.join(labels_dir, 'synthseg_segmentation_names_2.0.npy')
    args['names_parcellation_labels'] = os.path.join(labels_dir, 'synthseg_parcellation_names.npy')
    args['names_qc_labels'] = os.path.join(labels_dir, 'synthseg_qc_names_2.0.npy')
    args['topology_classes'] = os.path.join(labels_dir, 'synthseg_topological_classes_2.0.npy')
    args['n_neutral_labels'] = 19
    args['v1'] = False


    # Run the segmentation model
    atlas_segmentation(
            path_images=input_file,
            path_segmentations=output_path,
            path_model_segmentation=args['path_model_segmentation'],
            labels_segmentation=args['labels_segmentation'],
            robust=args['robust'],
            fast=args['fast'],
            v1=args['v1'],
            do_parcellation=args['parc'],
            n_neutral_labels=args['n_neutral_labels'],
            names_segmentation=args['names_segmentation_labels'],
            labels_denoiser=args['labels_denoiser'],
            path_posteriors=args['post'],
            path_resampled=args['resample'],
            path_volumes=args['vol'],
            path_model_parcellation=args['path_model_parcellation'],
            labels_parcellation=args['labels_parcellation'],
            names_parcellation=args['names_parcellation_labels'],
            path_model_qc=args['path_model_qc'],
            labels_qc=args['labels_qc'],
            path_qc_scores=args['qc'],
            names_qc=args['names_qc_labels'],
            cropping=args['crop'],
            topology_classes=args['topology_classes'],
            ct=args['ct'])


# Function to compute weights based on the type of tissue as labeled
def compute_weights(data):
    # Pre-defined dictionaries
    rj = {
        "ventricles": 0.1,
        "white matter fibers": 0.2,
        "functional cortical areas": 0.3,
        "blood vessels": 0.4,
        "critical region": 1,
    }
    
    ajk = {
        "ventricles": 0.25,
        "functional cortical areas": {
            "speech areas": 0.3,
            "vision areas": 0.1,
            "sensory areas": 0.2,
            "motor areas": 0.4
        },
        "white matter fibers": {
            "speech fibers": 0.1,
            "vision fibers": 0.2,
            "sensory fibers": 0.3,
            "motor fibers": 0.4
        },
        "blood vessels": 0.25,
        "critical region": 0.25,
    }

    def get_rj_value(label): 
        if label in rj:
            return rj[label]
        if "areas" in label:
            return rj["functional cortical areas"]
        if "fibers" in label:
            return rj["white matter fibers"]
        return 0.1 # Default value if no specific rj value found


    # Function to get ajk value based on label
    def get_ajk_value(label):
        if label in ajk and label not in ["functional cortical areas", "white matter fibers"]:
            return ajk[label]
        for area_type, value in ajk["functional cortical areas"].items():
            if area_type in label:
                return value
        for area_type, value in ajk["white matter fibers"].items():
            if area_type in label:
                return value
        return 0.25  # Default value if no specific ajk value found
    
    # Calculate weights and update the data
    for entry in data:
        label = entry["Label"]
        rj_value = get_rj_value(label) # Default to 1 if label not found in rj
        ajk_value = get_ajk_value(label)
        weight = rj_value * ajk_value
        entry["Weight"] = weight
    
    return data




# ------------------ SynthSeg Structure Segmentation ------------------
tumor_seg_file_path = "MR images/Images/BraTS20_Training_001_seg.nii"
tumor_seg_load = nib.load(tumor_seg_file_path)
tumor_seg_voxels = tumor_seg_load.get_fdata()

# Import the original T1 image capture to subtract the mask and input it into SynthSeg
t1_file_path = "MR images/Images/BraTS20_Training_001_t1.nii"
t1_load = nib.load(t1_file_path)
t1_voxels = t1_load.get_fdata()

# Mask out the tumor in the image
tumor_mask = np.where((tumor_seg_voxels == 1) | (tumor_seg_voxels == 4), 1, 0)
t1_voxels *= (1 - tumor_mask)

# Save the masked file to process with SynthSeg
nib_path = 'MR images/SynthSeg/TumorSubtraction.nii.gz'
img = nib.Nifti1Image(t1_voxels.astype(np.int32), tumor_seg_load.affine)
nib.save(img, nib_path)


atlas_seg_file_path = "MR images/SynthSeg/TumorSubtraction_synthseg.nii.gz"
call_synthseg(nib_path, atlas_seg_file_path)


# ------------------ Combining SynthSeg and mmFormer Output to a Single File ------------------

atlas_seg_file_path = "MR images/SynthSeg/TumorSubtraction_synthseg.nii.gz"

atlas_seg_load = nib.load(atlas_seg_file_path)
atlas_seg_voxels = atlas_seg_load.get_fdata()

tumor_seg_combine = np.where((tumor_seg_voxels == 1) | (tumor_seg_voxels == 4), 600, 0)
tumor_seg_combine += np.where((tumor_seg_voxels == 2), 498, 0)

full_seg_voxels = tumor_seg_combine + atlas_seg_voxels

nib_path = 'MR images/Labels/FullSegmentation_008_Test.nii.gz'
img = nib.Nifti1Image(full_seg_voxels.astype(np.int32), tumor_seg_load.affine)
nib.save(img, nib_path)

nib_path = 'MR images/Labels/FullSegmentation_008_Test.nii.gz'

full_seg_load = nib.load(nib_path)
full_seg_voxels = full_seg_load.get_fdata()

# Read JSON file
with open('tissue_labels.json', 'r') as file:
    data = json.load(file)

# tumor_seg_combine = np.zeros(full_seg_voxels.shape)
tissue_weights = compute_weights(data)
tumor_seg_combine = np.where(full_seg_voxels != 0, 0.025, 0)

# Loop through the dictionary to replace values within the array
for value in tissue_weights:
    weight = value["Weight"]
    if value["Region"] == "Tumor":
        weight = -0.1
    if value["Region"] == "wmsa":
        weight = 0.025
    tissue_indx = int(value["Index"])
    tumor_seg_combine[full_seg_voxels == tissue_indx] = weight


print(np.unique(tumor_seg_combine))
nib_path = 'MR images/Labels/WeightedSegmentation_001_Test3.nii.gz'
img = nib.Nifti1Image(tumor_seg_combine.astype(np.float32), full_seg_load.affine)
nib.save(img, nib_path)
