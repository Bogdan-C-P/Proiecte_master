import os
import torch
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms
from torch.autograd import Variable


class Interpolate(nn.Module):
    def __init__(self, size=8, mode='bilinear'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


class Autoencoder_interpolate(nn.Module):
    def __init__(self):
        super(Autoencoder_interpolate, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, (5, 5), stride=(2, 2), padding=2),  # b, 16, 48, 48
            nn.ReLU(),
            nn.Conv2d(16, 32, (3,3), stride=(1, 1), padding=1),  # b, 32, 48, 48
            nn.ReLU(),
            nn.Conv2d(32, 3, (3, 3), stride=(1, 1), padding=1),  # b, 3, 48, 48
        )

        self.interpolate = Interpolate(size=48)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 32, (3, 3), stride=(1, 1), padding=(1, 1)),  # b, 32, 48, 48
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (3,3), stride=(1, 1), padding=(1,1)),  # b, 16, 48, 48
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, (5, 5), stride=(2, 2), padding=(2,2), output_padding=(1, 1)),  # b, 3, 96, 96
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
        interp_img = self.interp(com_img)
        decomp_img = self.decoder_f(interp_img)
        return com_img, decomp_img


if __name__ == '__main__':

    img_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    trainset = datasets.STL10(root='.', split='train', download=True, transform=img_transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                            shuffle=True)

    testset = datasets.STL10(root='.', split='test', download=True, transform=img_transform)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

    num_epochs = 20
    learning_rate = 1e-3

    model = Autoencoder_interpolate().cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in range(num_epochs):
        total_loss = 0
        for data in train_loader:
            img, _ = data
            img = Variable(img).cuda()
            print(img.size())
            # ===================forward=====================
            output = model(img)
            loss = criterion(output[1], img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # ===================log========================
        print(f'epoch [{epoch+1}/{num_epochs}], loss:{total_loss:.4f}')

        torch.save(model.state_dict(), 'conv_autoencoder_interp.pth')

    torch.save(model.state_dict(), 'conv_autoencoder_interp.pth')

