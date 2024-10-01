import os
import nibabel as nib
from tqdm import tqdm

# Define paths
images_path = r"C:\Users\zaq12\Desktop\imagesTr"
labels_path = r"C:\Users\zaq12\Desktop\labelsTr"

# List all label files
label_files = os.listdir(labels_path)

# Iterate over each label file
for label_file in tqdm(label_files):
    try:
        # Construct file paths
        image_file_path = os.path.join(images_path, label_file)
        label_file_path = os.path.join(labels_path, label_file)

        # Load NIfTI files
        nii_image = nib.load(image_file_path)
        nii_label = nib.load(label_file_path)

        # Extract header information from the image file
        image_header = nii_image.header

        # Apply header information to the label file
        affine = image_header.get_best_affine()
        nii_label.set_sform(affine)
        nii_label.set_qform(affine)

        # Save the aligned NIfTI label file
        nib.save(nii_label, label_file_path)
    except Exception as e:
        # Print the file name and error message if an exception occurs
        print(f"Error processing {label_file}: {e}")