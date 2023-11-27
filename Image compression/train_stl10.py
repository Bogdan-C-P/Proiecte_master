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
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

trainset = datasets.STL10(root='.', split='train', download=True, transform=img_transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                           shuffle=True)

testset = datasets.STL10(root='.', split='test', download=True, transform=img_transform)

test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)



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

        self.interpolate = Interpolate(size=48)

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
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 32, 32)
    return x


num_epochs = 40

learning_rate = 1e-3

model = Autoencoder().cuda()




criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)


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
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, total_loss))

    torch.save(model.state_dict(), 'conv_autoencoder_16.pth')

torch.save(model.state_dict(), 'conv_autoencoder_16.pth')

