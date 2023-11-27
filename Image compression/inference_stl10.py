import os
import torch
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from compress_rgb import compress_RGB, scale_values
import numpy as np
from PIL import Image
from compress_rgb import compress_RGB

img_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.24703223,  0.24348513, 0.26158784))
])


testset = datasets.STL10(root='.', split='train', download=False, transform=img_transform)

test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

SIZE = 96

class Interpolate(nn.Module):
    def __init__(self, size = 8, mode='bilinear'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x



class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), stride=(2, 2), padding=1),  # b, 16, 16, 16
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3), stride=(1, 1), padding=1),  # b, 32, 16, 16
            nn.ReLU(),
            nn.Conv2d(32, 3, (3, 3), stride=(1, 1), padding=1),  # b, 3, 16, 16
        )

        self.interpolate = Interpolate(size=int(SIZE/2))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 32, (3, 3), stride=(1, 1), padding=(1, 1)),  # b, 32, 16, 16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (3, 3), stride=(1, 1), padding=(1, 1)),  # b, 16, 16, 16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),  # b, 3, 32, 32
            nn.Tanh()
        )

    def encoder_f(self, x):
        return self.encoder(x)

    def interp(self, x):
        return self.interpolate(x)

    def decoder_f(self, x):
        return self.decoder(x)

    def forward(self, x):
        com_img = self.encoder_f(x)
        #print('comm img size', com_img.size())
        interp_img = self.interp(com_img)
        #print('interp_img size', interp_img.size())
        decomp_img = self.decoder_f(interp_img)
        #print('decomp_img size', decomp_img.size())
        return com_img, decomp_img


def to_img(x, size=SIZE):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, size, size)

    return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Autoencoder()

model.load_state_dict(torch.load('conv_autoencoder_16.pth'))

model = model.to(device)
model.eval()



img, label = next(iter(test_loader))
img = img.to(device)
img = img.reshape(3, SIZE, SIZE)
with torch.no_grad():
    image = img.unsqueeze(0)
    output = model(image)[1].cpu().data
    com_img = model.encoder_f(image)
    int_img = model.interp(com_img)
    print(int_img)
    #print(com_img.size())
    test_img = com_img.cpu().squeeze(0).permute((1, 2, 0)).numpy()
    #print(test_img)
    #print(test_img.shape)
    min_value = np.min(test_img)
    max_value = np.max(test_img)
    converted_to_255 = ((test_img - min_value) / (max_value - min_value)) * 255
    converted_to_255 = np.round(converted_to_255)
    #print(np.round(converted_to_255))


    pil_image = Image.fromarray(np.uint8(converted_to_255))
    # pil_image.save('uncompressed.jpg', quality=100)
    pil_image.save('compressed_image.jpg', quality=100)
    compressed_img = Image.open('compressed_image.jpg')

    converted_back_to_original = ((np.array(compressed_img) / 255) * (max_value - min_value)) + min_value
    compressed_img_torch = torch.from_numpy(converted_back_to_original).cuda().float().permute(2, 0, 1).unsqueeze(0)
    #print(compressed_img_torch)
    decomp_img = model.decoder_f(compressed_img_torch)


plt.subplot(1, 3, 1)
plt.imshow(img.cpu().permute(1, 2, 0))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(to_img(output).squeeze(0).permute((1, 2, 0)))  # Assuming output is a tensor
plt.title('Result from Model')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(to_img(decomp_img).cpu().squeeze(0).permute((1, 2, 0)))  # Assuming output is a tensor
plt.title('With jpeg')
plt.axis('off')
plt.show()