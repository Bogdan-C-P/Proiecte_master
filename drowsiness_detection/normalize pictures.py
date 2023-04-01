import os
import cv2

input_dir = "poze_raw"
output_dir = "poze_grayscale"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for dirpath, dirnames, filenames in os.walk(input_dir):
    for dirname in dirnames:
        input_subdir = os.path.join(input_dir, dirname)
        output_subdir = os.path.join(output_dir, dirname)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        
        for filename in os.listdir(input_subdir):
            input_path = os.path.join(input_subdir, filename)
            output_path = os.path.join(output_subdir, filename)
            
            # Load the input image in color
            input_image = cv2.imread(input_path)
            
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
            
            # Normalize the grayscale image
            normalized_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
            
            # Save the normalized grayscale image
            cv2.imwrite(output_path, normalized_image)
