import cv2
import os

# set input and output file paths
input_file = '30.mp4'
output_folder = 'poze ochi/'

# create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# open input video file
cap = cv2.VideoCapture(input_file)

# get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

# calculate interval for extracting frames
interval = int(fps / 10) # extract 10 frames per second
frame_number = 0

# loop through video frames
while cap.isOpened():
    # read frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # check if frame number is within interval
    if frame_number % interval == 0:
        # center crop frame
        height, width, channels = frame.shape
        crop_size = min(height, width)
        top = (height - crop_size) // 2
        bottom = (height - crop_size) // 2
        left = (width - crop_size) // 2
        right = (width - crop_size) // 2
        cropped_frame = frame[top:height-bottom, left:width-right]
        cropped_frame = cv2.resize(cropped_frame, (224, 224))
        
        # save cropped frame as image file
        frame_path = os.path.join(output_folder, f'30_frame_{frame_number:04d}.jpg')
        cv2.imwrite(frame_path, cropped_frame)
    
    frame_number += 1

# release resources
cap.release()