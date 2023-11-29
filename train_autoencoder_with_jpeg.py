import torch
from torch import nn
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import cv2


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



class Autoencoder_jpeg(nn.Module):
    def __init__(self):
        super(Autoencoder_jpeg, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), stride=(2, 2), padding=1),  # b, 16, 48, 48
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3), stride=(1, 1), padding=1),  # b, 32, 48, 48
            nn.ReLU(),
            nn.Conv2d(32, 3, (3, 3), stride=(1, 1), padding=1),  # b, 3, 48, 48
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 32, (3, 3), stride=(1, 1), padding=(1, 1)),  # b, 32, 48, 48
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (3, 3), stride=(1, 1), padding=(1, 1)),  # b, 16, 48, 48
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),  # b, 3, 96, 96
            nn.Tanh()
        )

    def encoder_f(self, x):
        return self.encoder(x)

    def decoder_f(self, x):
        return self.decoder(x)

    def forward(self, x):
        com_img = self.encoder_f(x)
        batch_size = com_img.size(0)

        decompressed_images = []
        with torch.no_grad():
            for i in range(batch_size):
                compressed_img_torch = apply_jpeg_on_encoding(com_img[i])
                decompressed_images.append(compressed_img_torch)
            reconstructed_images = torch.stack(decompressed_images, dim=0)

        with torch.enable_grad():
            decomp_img = self.decoder_f(reconstructed_images)

        return com_img, decomp_img


if __name__ == '__main__':
    img_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    trainset = datasets.STL10(root='../', split='train', download=False, transform=img_transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)

    testset = datasets.STL10(root='../', split='test', download=False, transform=img_transform)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

    num_epochs = 20
    learning_rate = 1e-3
    model = Autoencoder_jpeg().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

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
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, total_loss))
    #    torch.save(model.state_dict(), 'conv_autoencoder_jpeg.pth')
    #torch.save(model.state_dict(), 'conv_autoencoder_jpeg.pth')

