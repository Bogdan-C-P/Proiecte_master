import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.autograd import Variable

SIZE = 96

class Autoencoder_basic(nn.Module):
    def __init__(self, n_channels=3, n_hid=32, encoding_dim=512):
        super(Autoencoder_basic, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, n_hid, kernel_size=(3, 3), stride=(2, 2), padding=1),  # n_hid 48 48
            nn.GELU(),
            nn.Conv2d(n_hid, n_hid, kernel_size=(3, 3), stride=(1, 1), padding=1),  # n_hid 48 48
            nn.GELU(),
            nn.Conv2d(n_hid, 2 * n_hid, kernel_size=(3, 3), stride=(2, 2), padding=1),  # n_hid 24 24
            nn.GELU(),
            nn.Conv2d(2 * n_hid, 2 * n_hid, kernel_size=(3, 3), stride=(1, 1), padding=1),  # n_hid 24 24
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(2 * n_hid * int(SIZE/4) * int(SIZE/4), encoding_dim)
        )

        self.linear = nn.Sequential(
            nn.Linear(encoding_dim, 2 * n_hid * int(SIZE/4) * int(SIZE/4)),
            nn.GELU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2 * n_hid, 2 * n_hid, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1)),  # n_hid 24 24
            nn.GELU(),
            nn.Conv2d(2 * n_hid, 2 * n_hid, kernel_size=(3, 3), stride=(1, 1), padding=1),  # n_hid 24 24
            nn.GELU(),
            nn.ConvTranspose2d(2 * n_hid, n_hid, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), output_padding=(1, 1)),  # n_hid 48 48
            nn.GELU(),
            nn.Conv2d(n_hid, n_hid, kernel_size=(3, 3), stride=(1, 1), padding=1),  # n_hid 48 48
            nn.GELU(),
            nn.ConvTranspose2d(n_hid, n_channels, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), output_padding=(1, 1)),  # n_hid 96 96
            nn.Tanh()
        )


    def forward(self, x):
        encoding = self.encoder(x)
        linear_decoding = self.linear(encoding)
        linear_decoding = linear_decoding.reshape(linear_decoding.shape[0], -1, int(SIZE/4), int(SIZE/4))
        decom_img = self.decoder(linear_decoding)
        return encoding, decom_img


if __name__ == '__main__':
    img_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    trainset = datasets.STL10(root='../', split='train', download=False, transform=img_transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                            shuffle=True)

    testset = datasets.STL10(root='../', split='test', download=False, transform=img_transform)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

    num_epochs = 20
    learning_rate = 1e-3
    model = Autoencoder_basic(n_channels=3,
                              n_hid=32,
                              encoding_dim=512).cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)

    model.train()
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
        print(f'epoch [{epoch+1}/{num_epochs}], loss:{total_loss:.4f}')

        # ===================save========================
        torch.save(model.state_dict(), 'conv_autoencoder_basic.pth')

    torch.save(model.state_dict(), 'conv_autoencoder_basic.pth')

