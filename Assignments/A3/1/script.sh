#!/bin/bash

# Define main data directory
DATA_DIR="../../data"
OUT_DIR="output"


# Preprocess and run first-level analysis for each subject
for SUBJECT in $(ls ${DATA_DIR} | grep S); do
    echo "Processing Subject: ${SUBJECT}"
    
    SUBJECT_DIR="${DATA_DIR}/${SUBJECT}"
    OUT_SUBJECT_DIR="${OUT_DIR}/${SUBJECT}"
    mkdir -p ${OUT_SUBJECT_DIR}
    FUNC_IMG="${SUBJECT_DIR}/raw_fMRI_raw_bold.nii.gz"
    STRUCT_IMG="${SUBJECT_DIR}/raw_T1_raw_anat_defaced.nii.gz"
    MASK_IMG="${SUBJECT_DIR}/raw_fMRI_raw_bold_brain_mask.nii.gz"
    
    # Step 1: Brain Extraction
    bet ${STRUCT_IMG} ${OUT_SUBJECT_DIR}/T1_brain.nii.gz -R -f 0.5 -g 0
    bet ${FUNC_IMG} ${OUT_SUBJECT_DIR}/func_brain.nii.gz -F -f 0.3
    
    # Step 2: Motion Correction
    mcflirt -in ${FUNC_IMG} -out ${OUT_SUBJECT_DIR}/func_mc.nii.gz -refvol 0 -plots

    # Step 3: Spatial Smoothing
    fslmaths ${OUT_SUBJECT_DIR}/func_mc.nii.gz -s 4 ${OUT_SUBJECT_DIR}/func_smooth.nii.gz

    # Step 4: Temporal Filtering
    fslmaths ${OUT_SUBJECT_DIR}/func_smooth.nii.gz -bptf 50 -1 ${OUT_SUBJECT_DIR}/filtered_func_data.nii.gz

    # Step 5: Registration to MNI Template
    flirt -in ${OUT_SUBJECT_DIR}/T1_brain.nii.gz -ref ${FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz -omat ${OUT_SUBJECT_DIR}/anat2mni.mat
    flirt -in ${OUT_SUBJECT_DIR}/filtered_func_data.nii.gz -ref ${FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz -applyxfm -init ${OUT_SUBJECT_DIR}/anat2mni.mat -out ${OUT_SUBJECT_DIR}/func_mni.nii.gz
done

