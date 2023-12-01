import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.functional as F


SIZE = 96


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation='relu'):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 16, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 16, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        #print( m_batchsize, C, width, height )
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out



class Autoencoder_Attention(nn.Module):
    def __init__(self, n_channels=3, n_hid=32, encoding_dim=1024):
        super(Autoencoder_Attention, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, n_hid, kernel_size=(3, 3), stride=(2, 2), padding=1),  # n_hid 48 48
            nn.GELU(),
            nn.Conv2d(n_hid, n_hid, kernel_size=(3, 3), stride=(1, 1), padding=1),  # n_hid 48 48
            nn.GELU(),
            nn.Conv2d(n_hid, 2 * n_hid, kernel_size=(3, 3), stride=(2, 2), padding=1),  # n_hid 24 24
            nn.GELU(),
            nn.Conv2d(2 * n_hid, 2 * n_hid, kernel_size=(3, 3), stride=(1, 1), padding=1),  # n_hid 24 24
            nn.GELU(),

        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * n_hid * int(SIZE / 4) * int(SIZE / 4), encoding_dim),
            nn.Linear(encoding_dim, 2 * n_hid * int(SIZE/4) * int(SIZE/4)),
            nn.GELU()
        )

        self.attention = Self_Attn(2 * n_hid)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2 * n_hid, 2 * n_hid, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1)),  # n_hid 24 24
            nn.GELU(),
            nn.ConvTranspose2d(2 * n_hid, 2 * n_hid, kernel_size=(3, 3), stride=(1, 1), padding=1),  # n_hid 24 24
            nn.GELU(),
            nn.ConvTranspose2d(2 * n_hid, n_hid, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), output_padding=(1, 1)),  # n_hid 48 48
            nn.GELU(),
            nn.ConvTranspose2d(n_hid, n_hid, kernel_size=(3, 3), stride=(1, 1), padding=1),  # n_hid 48 48
            nn.GELU(),
            nn.ConvTranspose2d(n_hid, n_channels, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), output_padding=(1, 1)),  # n_hid 96 96
            nn.Tanh()
        )

    def forward(self, x):
        encoding = self.encoder(x)

        attention_map = self.attention(encoding)
        linear_decoding = self.linear(encoding)
        linear_decoding = linear_decoding.reshape(linear_decoding.shape[0], -1, int(SIZE/4), int(SIZE/4))

        decom_img = self.decoder(linear_decoding+ attention_map)
        return encoding, decom_img


if __name__ == '__main__':
    img_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    trainset = datasets.STL10(root='.', split='train', download=False, transform=img_transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                            shuffle=True)

    testset = datasets.STL10(root='.', split='test', download=False, transform=img_transform)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

    num_epochs = 100
    learning_rate = 1e-4
    model = Autoencoder_Attention(n_channels=3,
                                  n_hid=32,
                                  encoding_dim=1024).cuda()
    model.load_state_dict(torch.load('attention_AE_model.pth'))
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

