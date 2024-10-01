import os
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

# Constants
IGNORE_BACKGROUND = True
PREDICTIONS_DIR = Path(r"C:\Users\guest0417\Desktop\reviewerA")
GROUND_TRUTH_DIR = Path(r"C:\Users\guest0417\Desktop\TOM500_STAPLE")

# List of prediction files
predictions = [f for f in PREDICTIONS_DIR.iterdir() if f.suffix == ".nii.gz"]

# Dictionary to store results
results = {}

def process_prediction(prediction):
    # Read prediction and ground truth images
    pred_img = sitk.ReadImage(str(PREDICTIONS_DIR / prediction))
    pred_arr = sitk.GetArrayFromImage(pred_img)
    gt_img = sitk.ReadImage(str(GROUND_TRUTH_DIR / prediction))
    gt_arr = sitk.GetArrayFromImage(gt_img)
    
    # Number of classes in ground truth
    class_num = len(np.unique(gt_arr))
    res = np.zeros(7)
    
    # Calculate metrics for each class
    for index in range(class_num):
        if IGNORE_BACKGROUND and index == 0:
            continue
        
        pred_arr_tmp = (pred_arr == index).astype(int)
        gt_arr_tmp = (gt_arr == index).astype(int)
        
        intersection = np.sum(pred_arr_tmp * gt_arr_tmp)
        sum_pred = np.sum(pred_arr_tmp)
        sum_gt = np.sum(gt_arr_tmp)
        
        res_tmp = np.zeros(7)
        res_tmp[0] = 2 * intersection / (sum_pred + sum_gt)  # Dice
        res_tmp[1] = intersection / (sum_pred + sum_gt - intersection)  # Jaccard
        res_tmp[2] = intersection / sum_gt  # Sensitivity
        res_tmp[3] = np.sum((pred_arr_tmp + gt_arr_tmp) == 0) / np.sum(gt_arr_tmp == 0)  # Specificity
        res_tmp[4] = intersection / sum_pred  # Precision
        res_tmp[5] = intersection / sum_gt  # Recall
        res_tmp[6] = (intersection + np.sum((pred_arr_tmp + gt_arr_tmp) == 0)) / pred_arr_tmp.size  # Accuracy
        
        res += res_tmp
    
    # Average the results over the number of classes
    return res / (class_num - 1) if IGNORE_BACKGROUND else res / class_num

if __name__ == "__main__":
    # Create a ThreadPoolExecutor with the number of CPU cores
    num_threads = cpu_count()
    executor = ThreadPoolExecutor(max_workers=num_threads)
    
    # Process predictions in parallel
    futures = {executor.submit(process_prediction, prediction.name): prediction.name for prediction in predictions}
    
    # Collect results
    for future in tqdm(futures):
        results[futures[future]] = future.result()
    
    # Save results to CSV
    results_df = pd.DataFrame.from_dict(results, orient="index", columns=["dice", "jaccard", "sensitivity", "specificity", "precision", "recall", "accuracy"])
    results_df.to_csv("results.csv", index=True)