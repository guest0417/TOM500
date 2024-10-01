import os
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pydicom import dcmread

# Constants
IGNORE_BACKGROUND = True
DICOM_PATH = Path(r"F:\data\MRI\dcm\T2")
NII_PATH = Path(r"F:\data\seg\reviewerA")

# List of prediction files
predictions = [f for f in NII_PATH.iterdir() if f.suffix == ".nii.gz"]

def read_img(path):
    """Read an image using SimpleITK and return the array, spacing, and size."""
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img), img.GetSpacing(), img.GetSize()

def process(filename):
    """Process a single prediction file to extract features."""
    base_name = filename.stem.split('_')
    dicom_subfolder = base_name[1]
    nii_path = NII_PATH / filename
    dicom_path = DICOM_PATH / base_name[0] / dicom_subfolder
    files = [dicom_path / file for file in os.listdir(dicom_path)]
    
    dcms = []
    xaxis = []
    for file in files:
        ds = dcmread(str(file))
        if ds.SeriesDescription == "T2W_DRIVE":
            dcms.append(ds)
            xaxis.append(ds.ImagePositionPatient[1])
    
    # Sort DICOMs based on xaxis
    sorted_indices = np.argsort(xaxis)
    dcms = [dcms[i] for i in sorted_indices]
    
    nii_array, voxel_spacing, size = read_img(nii_path)
    unique_values = np.unique(nii_array.flatten())
    intensities = np.zeros(len(unique_values))
    volumes = np.zeros(len(unique_values))
    thicknesses = np.zeros(4)
    
    for value in unique_values:
        sum_intensity = 0
        count_nonzero = 0
        for i, ds in enumerate(dcms):
            rawimg = ds.pixel_array
            cur_page = nii_array[i, :, :]
            label = np.where(cur_page == value, 1, 0)
            seg_img = rawimg * label
            
            sum_intensity += np.sum(seg_img)
            count_nonzero += np.count_nonzero(seg_img)
        
        avg_intensity = sum_intensity / count_nonzero if count_nonzero else 0
        intensities[value] = avg_intensity
        volumes[value] = count_nonzero * np.prod(voxel_spacing)
    
    results = np.concatenate((intensities, volumes), axis=None)
    return results

if __name__ == "__main__":
    # Create a ProcessPoolExecutor with the number of CPU cores
    executor = ProcessPoolExecutor(max_workers=cpu_count())
    
    # Process predictions in parallel
    all_task = {executor.submit(process, filename): filename for filename in predictions}
    
    # Collect results
    results_dict = {}
    for task in tqdm(all_task):
        results_dict[all_task[task].name] = task.result()
    
    # Save results to CSV
    columns = [f"SI{i}" for i in range(10)] + [f"V{i}" for i in range(10)]
    results_df = pd.DataFrame.from_dict(results_dict, orient="index", columns=columns)
    results_df.to_csv("si_and_vol_all.csv", index=True)