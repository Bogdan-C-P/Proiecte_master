import os
import torch
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
from torchvision.utils import save_image

img_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5,), (0.5,))
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

trainset = datasets.CIFAR10(root='.', train=True, download=False, transform=img_transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                           shuffle=True)

testset = datasets.CIFAR10(root='.', train=False, download=False, transform=img_transform)

test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)


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
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 32, 32)
    return x


num_epochs = 20
learning_rate = 1e-3

model = AutoEncoder(n_channels=3,
                    n_hid=32,
                    encoding_dim=256).cuda()

#model.load_state_dict(torch.load('conv_autoencoder_downsample_512.pth'))
model.train()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)


for epoch in range(num_epochs):
    total_loss = 0
    for data in train_loader:
        img, _ = data
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output[1], img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step(loss.item())

    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, total_loss))

    # ===================save========================
    torch.save(model.state_dict(), 'conv_autoencoder_downsample_512.pth')

torch.save(model.state_dict(), 'conv_autoencoder_downsample_512.pth')

