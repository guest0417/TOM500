import os
from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count, freeze_support

def process_directory(root):
    """
    Process a directory containing DICOM files and convert them to a single NIfTI file.
    
    Args:
        root (str): The root directory containing DICOM files.
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(root)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    # Turn the list of files into a single nii.gz file
    root_path = Path(root)
    output_filename = f"{root_path.parts[-1]}.nii.gz"
    output_path = Path(r"F:\data\MRI\nii\T2") / output_filename
    sitk.WriteImage(image, str(output_path))

if __name__ == '__main__':
    # Ensure that the freeze_support() is called for Windows compatibility
    freeze_support()
    
    # Create a ProcessPoolExecutor with the number of available CPU cores
    num_threads = cpu_count()
    
    # Use the executor to process directories in parallel
    futures = []
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        for root, dirs, files in os.walk(r"F:\data\MRI\dcm\T2"):
            if "-" in root:
                future = executor.submit(process_directory, root)
                futures.append(future)
        
        # Use tqdm to display a progress bar for the futures
        for future in tqdm(futures):
            future.result()