In this study, we collected video data of people's eyes in a natural environment. We extracted a few frames from each second of the videos to obtain a dataset of eye images. We then classified these images into two classes: "open eyes" and "closed eyes".

We have two options for classifying the images. The first option is to manually label each image by determining whether the eyes are open or closed. The second option is to use a pre-trained classifier to help us classify the images automatically. In the latter case, we set a threshold value for the classifier's output to ensure a certain level of security.

Once we have classified the images, we restructure the dataset to create a balanced training and testing set. We then use this dataset to train and test our own classification model.
