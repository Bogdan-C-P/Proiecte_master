import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import cv2


SIZE = 96


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation='relu'):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim //2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim //2, kernel_size=1)
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


def apply_jpeg_on_encoding(encoding):
    test_img = encoding.cpu().permute(1, 2, 0).numpy()
    min_value = np.min(test_img)
    max_value = np.max(test_img)
    converted_to_255 = ((test_img - min_value) / (max_value - min_value)) * 255
    converted_to_255 = np.uint8(np.round(converted_to_255))

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, encimg = cv2.imencode('.jpg', converted_to_255, encode_param)
    converted_to_255 = cv2.imdecode(encimg, 1)
    converted_back_to_original = ((np.array(converted_to_255) / 255) * (max_value - min_value)) + min_value
    compressed_img_torch = torch.from_numpy(
        converted_back_to_original).cuda().float().permute(2, 0, 1)
    return compressed_img_torch


class Interpolate(nn.Module):
    def __init__(self, size=8, mode='bilinear'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


class Autoencoder_with_attention(nn.Module):
    def __init__(self):
        super(Autoencoder_with_attention, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), stride=(2, 2), padding=1),  # b, 16, 48, 48
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3), stride=(1, 1), padding=1),  # b, 32, 48, 48
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), stride=(2, 2), padding=1),  # b, 32, 24, 24
            nn.ReLU(),
            nn.Conv2d(32, 3, (3, 3), stride=(1, 1), padding=1),  # b, 3, 24, 24
        )
        self.attention = Self_Attn(3)
        self.interpolate = Interpolate(24)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 32, (3, 3), stride=(1, 1), padding=(1, 1)),  # b, 32, 24, 24
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),  # b, 32, 48, 48
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (3, 3), stride=(1, 1), padding=(1, 1)),  # b, 16, 48, 48
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),  # b, 3, 96, 96
            nn.Tanh()
        )

    def encoder_f(self, x):
        return self.encoder(x)

    def apply_attention(self, x):
        out = self.attention(x)
        return out

    def decoder_f(self, x):
        return self.decoder(x)

    def forward(self, x):
        com_img = self.encoder_f(x)
        #print(com_img.size())


        attention_map = self.apply_attention(com_img)
        #print(attention_map.size())
        interp = self.interpolate(attention_map)

        #print(interp.size())
        decomp_img = self.decoder_f(interp)
        #print()
        return com_img, decomp_img


if __name__ == '__main__':
    img_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    trainset = datasets.STL10(root='.', split='train', download=False, transform=img_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)

    testset = datasets.STL10(root='.', split='test', download=False, transform=img_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

    num_epochs = 60
    learning_rate = 1e-3
    model = Autoencoder_with_attention().cuda()

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
        torch.save(model.state_dict(), 'conv_autoencoder_attention.pth')

    torch.save(model.state_dict(), 'conv_autoencoder_attention.pth')

