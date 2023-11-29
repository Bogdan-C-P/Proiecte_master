import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from train_autoencoder_interp import Autoencoder_interpolate
from train_autoencoder_basic import Autoencoder_basic
from train_autoencoder_with_jpeg import Autoencoder_jpeg, apply_jpeg_on_encoding
from train_attention import Autoencoder_with_attention
img_transform = transforms.Compose([
    transforms.ToTensor(),
])

testset = datasets.STL10(root='../', split='train', download=True, transform=img_transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

SIZE = 96


def to_img(x, size=SIZE):
    """
    Convert tensor x in float value between [0,1]
     Args:
          x: torch tensor
          size: int
    Return:
        torch tensor
    """
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, size, size)
    return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'

img, label = next(iter(test_loader))
img = img.to(device)

model_interpolate = Autoencoder_interpolate().to(device).eval()
model_interpolate.load_state_dict(torch.load('conv_autoencoder_interp.pth'))
model_interpolate.eval()

model_basic = Autoencoder_basic().to(device).eval()
model_basic.load_state_dict(torch.load('conv_autoencoder_basic.pth'))
model_basic.eval()

model_with_jpeg = Autoencoder_jpeg().to(device)
model_with_jpeg.load_state_dict(torch.load('conv_autoencoder_jpeg.pth'))
model_with_jpeg.eval()

model_attention = Autoencoder_with_attention().to(device).eval()
model_attention.load_state_dict(torch.load('conv_autoencoder_attention.pth'))
model_attention.eval()


with torch.no_grad():
    linear_out = model_basic(img)[1].cpu().data
    jpeg_out = model_with_jpeg(img)[1].cpu().data

    # For interpolation at inference we replace interpolation with jpeg compression
    encoding = model_interpolate.encoder_f(img).squeeze(0)
    jpeg_interpolation = apply_jpeg_on_encoding(encoding).unsqueeze(0)
    interpolate_out = model_interpolate.decoder_f(jpeg_interpolation)

    encoding = model_attention.apply_attention(model_attention.encoder_f(img)).squeeze(0)
    jpeg_interpolation = apply_jpeg_on_encoding(encoding).unsqueeze(0)
    attention_out = model_attention.decoder_f(jpeg_interpolation)

#   DE ADAUGAT CALCUL METRICI SSIM

from skimage.metrics import structural_similarity as ssim
import numpy as np

def make_img(img):
    orig_img = to_img(img).cpu().squeeze(0).permute((1, 2, 0)).numpy()
    min_value = np.min(orig_img)
    max_value = np.max(orig_img)
    converted_to_255 = ((orig_img - min_value) / (max_value - min_value)) * 255
    orig_img = np.uint8(np.round(converted_to_255))
    return orig_img

plt.figure(figsize=(16, 5))

plt.subplot(1, 5, 1)
orig_img = make_img(img)
plt.imshow(orig_img)
plt.title('Original image')
plt.axis('off')

plt.subplot(1, 5, 2)
AE_image = make_img(linear_out)
#ssim_index, _ = ssim(orig_img, AE_image, win_size=min(AE_image.shape[:2]) // 7 * 2 + 1, full=True)
plt.imshow(AE_image)
plt.title('Basic AE \n SSIM: ') #+ str(ssim_index))
plt.axis('off')

plt.subplot(1, 5, 3)
plt.imshow(to_img(interpolate_out).cpu().squeeze(0).permute((1, 2, 0)))
plt.title('AE + Interpolation')
plt.axis('off')

plt.subplot(1, 5, 4)
plt.imshow(to_img(jpeg_out).cpu().squeeze(0).permute((1, 2, 0)))
plt.title('AE + JPEG')
plt.axis('off')


plt.subplot(1, 5, 5)
plt.imshow(to_img(attention_out).cpu().squeeze(0).permute((1, 2, 0)))
plt.title('AE + JPEG + Attention')
plt.axis('off')
plt.show()
