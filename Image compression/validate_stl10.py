import os
import torch
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt


img_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.24703223,  0.24348513, 0.26158784))
])

SIZE = 96


trainset = datasets.STL10(root='.', split='train', download=False, transform=img_transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                           shuffle=True)

testset = datasets.STL10(root='.', split='test', download=False, transform=img_transform)

test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)



class Interpolate(nn.Module):
    def __init__(self, size=8, mode='bilinear'):
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


def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, SIZE, SIZE)

    return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Autoencoder()

model.load_state_dict(torch.load('conv_autoencoder_16.pth'))

model = model.to(device)
model.eval()


img, label = next(iter(train_loader))
#print(label)
img = img.to(device)
img = img.reshape(3, SIZE, SIZE)
with torch.no_grad():
    image = img.unsqueeze(0)
    output = model(image)[1].cpu().data
    #print(img.size(), image.size(), output.size())
    diff = to_img(image.cpu() - output)


plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img.cpu().permute(1, 2, 0))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(to_img(output).squeeze(0).permute((1, 2, 0)))  # Assuming output is a tensor
plt.title('Result from Model')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(diff.squeeze(0).permute((1, 2, 0)))  # Assuming output is a tensor
plt.title('Difference')
plt.axis('off')
plt.show()
