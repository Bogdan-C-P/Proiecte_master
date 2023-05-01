import torch
from PIL import Image
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt



# Define transformation to apply to input image
data_transforms = {
    'train': transforms.Compose([
        transforms.Grayscale(),  # convert image to grayscale
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ]),
    'val': transforms.Compose([
        transforms.Grayscale(),  # convert image to grayscale
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ]),
}
data_dir = 'dataset_2'
image_datasets_train = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                                data_transforms['train'])

class_names = image_datasets_train.classes
# Load the model
model_conv = torch.jit.load('full_jit_scripted.pt')

# Set the model to evaluate mode
model_conv.eval()


# Define function to classify an image
def classify_image(image_path):
    # Load the image and apply the transformation
    image = Image.open(image_path)
    image = data_transforms['val'](image).unsqueeze(0)

    # Run the image through the model
    with torch.no_grad():
        output = model_conv(image)

    # Get the predicted class (0 = closed eyes, 1 = open eyes)
    _, predicted = torch.max(output.data, 1)

    return predicted.item()


# Example usage
image_path = 'D:\\Downloads\\facultate\\Master An1\\SEM2\ACABI\\achizitii\\poze_grayscale\\ochi_deschisi\\1_frame_0138.jpg'
predicted_class = classify_image(image_path)
print('Predicted class:', class_names[predicted_class])


image = Image.open(image_path)
plt.imshow(image, cmap='gray')
plt.title(class_names[predicted_class])
plt.show()