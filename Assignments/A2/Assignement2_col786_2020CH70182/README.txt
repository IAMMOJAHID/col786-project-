# README: Functional MRI Preprocessing and Statistical Analysis Results  
## Author: MOJAHID HUSSAIN 
This directory contains preprocessed functional MRI data and statistical analysis results from FEAT. Below is a description of the key files.  

---
For each sub-part (i.e 1):
## **1. Preprocessed Data Files**  
These files contain fMRI data after preprocessing steps such as motion correction, spatial smoothing, and temporal filtering.

- **filtered_func_data.nii.gz**  
  - Final preprocessed functional MRI data after motion correction, spatial smoothing, and high-pass filtering.  
  - Located in: `1/1.feat/`  

- **example_func.nii.gz**  
  - A representative single-volume fMRI scan used for alignment.  
  - Located in: `1/1.feat/reg/`  

- **motion_parameters.txt**  
  - File containing motion correction parameters for quality assessment.  
  - Located in: `1/1.feat/mc/`  

---

## **2. Registered (Non-Thresholded) Z-stat Maps**  
These files represent statistical results of the GLM contrasts before applying significance thresholds.

- **stats/zstat1.nii.gz**  
  - Non-thresholded Z-statistic map for Contrast 1: [Describe contrast, e.g., "Language vs. Control"].  
  - Located in: `1/1.feat/stats/`  

- **stats/zstat2.nii.gz**  
  - Non-thresholded Z-statistic map for Contrast 2: [Describe contrast, e.g., "Audio Sentence vs. Video Sentence"].  
  - Located in: `1/1.feat/stats/`  

- **stats/zstat3.nii.gz**  
  - Non-thresholded Z-statistic map for Contrast 3: [Describe contrast, e.g., "Video Language vs. Visual Control"].  
  - Located in: `1/1.feat/stats/`  

---

## **3. Registration Files**  
These files help align fMRI data to anatomical and standard brain space.

- **reg/example_func2highres.nii.gz**  
  - Functional scan registered to the subject’s anatomical (T1) scan.  
  - Located in: `1/1.feat/reg/`  

- **reg/example_func2standard.nii.gz**  
  - Functional scan registered to the standard MNI152 template.  
  - Located in: `1/1.feat/reg/`  

- **reg/highres2standard.nii.gz**  
  - Anatomical scan registered to the standard MNI152 template.  
  - Located in: `1/1.feat/reg/`  

---

## **4. Additional Notes**  
- Thresholded z-stat maps are available in `1/1.feat/stats/thresh_zstatX.nii.gz`.  
- For details on contrast design, refer to `1/1.feat/design.fsf`.  
- Quality control images are available in `1/1.feat/tsplot/`.  

If you have any questions, please contact **ch7200182@iitd.ac.in**.  

NOTE: Similar directories and files locations I have followed for part 2 and 3.
