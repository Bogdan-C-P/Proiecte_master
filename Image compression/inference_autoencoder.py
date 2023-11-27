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

testset = datasets.CIFAR10(root='.', train=True, download=False, transform=img_transform)

test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, n_channels, n_hid, encoding_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, n_hid, kernel_size=(3, 3), stride=(2, 2), padding=1),  # n_hid 16 16
            nn.GELU(),
            nn.Conv2d(n_hid, n_hid, kernel_size=(3, 3), stride=(1, 1), padding=1),  # n_hid 16 16
            nn.GELU(),
            nn.Conv2d(n_hid, 2 * n_hid, kernel_size=(3, 3), stride=(2, 2), padding=1),  # n_hid 8 8
            nn.GELU(),
            nn.Conv2d(2 * n_hid, 2 * n_hid, kernel_size=(3, 3), stride=(1, 1), padding=1),  # n_hid 8 8
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(2 * n_hid * 8 * 8, encoding_dim),
        )

        self.linear = nn.Sequential(
            nn.Linear(encoding_dim, 2 * n_hid * 8 * 8),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2 * n_hid, 2 * n_hid, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1)),  # n_hid 8 8
            nn.GELU(),
            nn.Conv2d(2 * n_hid, 2 * n_hid, kernel_size=(3, 3), stride=(1, 1), padding=1),  # n_hid 8 8
            nn.GELU(),
            nn.ConvTranspose2d(2 * n_hid, n_hid, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), output_padding=(1, 1)),  # n_hid 16 16
            nn.GELU(),
            nn.Conv2d(n_hid, n_hid, kernel_size=(3, 3), stride=(1, 1), padding=1),  # n_hid 16 16
            nn.GELU(),
            nn.ConvTranspose2d(n_hid, n_channels, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), output_padding=(1, 1)),  # n_hid 32 32
            nn.Tanh()
        )


    def forward(self, x):
        encoding = self.encoder(x)
        #print(encoding.size())
        linear_decoding = self.linear(encoding)
        linear_decoding = linear_decoding.reshape(linear_decoding.shape[0], -1, 8, 8)
        #print(linear_decoding.size())
        decom_img = self.decoder(linear_decoding)
        #print(decom_img.size())
        return encoding, decom_img



def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 32, 32)

    return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoEncoder(n_channels=3,
                    n_hid=32,
                    encoding_dim=256).cuda()

model.load_state_dict(torch.load('conv_autoencoder_downsample_512.pth'))

model = model.to(device)
model.eval()


img, label = next(iter(test_loader))
print(label)
img = img.to(device)
img = img.reshape(3, 32, 32)
with torch.no_grad():
    # Add a batch dimension and pass the image through the model
    image = img.unsqueeze(0)
    com_img = model.encoder(image)
    print(com_img)
    #plt.imshow(to_img(com_img.cpu()).squeeze(0).permute((1, 2, 0)))
    #plt.show()

    output = model(image)[1].cpu().data
    print(img.size(), image.size(), output.size())
    diff = to_img(image.cpu() - output)

''
plt.figure(figsize=(12,4))

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

