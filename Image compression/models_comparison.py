import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from train_end_to_end_interp import Autoencoder_interpolate
from train_autoencoder_basic import Autoencoder_basic
from attention_AE import Autoencoder_Attention
from train_autoencoder_with_jpeg import End_to_end_jpeg, apply_jpeg_on_encoding
from train_end_to_end_with_attention import End_to_end_with_attention
from jpeg_with_attention import JPG_with_attention
from skimage.metrics import structural_similarity as ssim
from fully_conv import FCNs, VGGNet
import numpy as np
import sys
import cv2


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

model_with_jpeg = End_to_end_jpeg().to(device)
model_with_jpeg.load_state_dict(torch.load('conv_end_to_end_jpeg.pth'))
model_with_jpeg.eval()

model_attention = End_to_end_with_attention().to(device).eval()
model_attention.load_state_dict(torch.load('End_to_end_with_attention.pth'))
model_attention.eval()


AE_with_attention = Autoencoder_Attention().to(device).eval()
AE_with_attention.load_state_dict(torch.load('attention_AE_model.pth'))
AE_with_attention.eval()

jpg_with_attention = JPG_with_attention().to(device).eval()
jpg_with_attention.load_state_dict(torch.load('jpeg_with_attention.pth'))
jpg_with_attention.eval()


vgg_model = VGGNet(requires_grad=True, remove_fc=True)
FCN = FCNs(pretrained_net=vgg_model).to(device)
FCN.load_state_dict(torch.load('FCN.pth'))


with torch.no_grad():
    linear_out = model_basic(img)[1].cpu().data
    jpeg_out = model_with_jpeg(img)[1].cpu().data
    ae_attention_output = AE_with_attention(img)[1].cpu().data
    jpg_with_attention_output = jpg_with_attention(img)[1].cpu().data
    FCN_out = FCN(img).cpu().data

    # For interpolation at inference we replace interpolation with jpeg compression
    encoding = model_interpolate.encoder_f(img).squeeze(0)
    jpeg_interpolation_1 = apply_jpeg_on_encoding(encoding).unsqueeze(0)
    interpolate_out = model_interpolate.decoder_f(jpeg_interpolation_1)

    encoding = model_attention.apply_attention(model_attention.encoder_f(img)).squeeze(0)
    jpeg_interpolation = apply_jpeg_on_encoding(encoding).unsqueeze(0)
    attention_out = model_attention.decoder_f(jpeg_interpolation)


def make_img(img):
    orig_img = to_img(img).cpu().squeeze(0).permute((1, 2, 0)).numpy()
    min_value = np.min(orig_img)
    max_value = np.max(orig_img)
    converted_to_255 = ((orig_img - min_value) / (max_value - min_value)) * 255
    orig_img = np.uint8(np.round(converted_to_255))
    return orig_img


plt.figure(figsize=(16, 8))

plt.subplot(1, 8, 1)
orig_img = make_img(img)
plt.imshow(orig_img)
print("Size of orig image: ", sys.getsizeof(orig_img))
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
result, encimg = cv2.imencode('.jpg', orig_img, encode_param)
plt.title('Original image \n Size of comp:' + str(sys.getsizeof(encimg)))
plt.axis('off')

plt.subplot(1, 8, 2)
AE_image = make_img(linear_out)
ssim_index, _ = ssim(orig_img, AE_image, win_size=3, multichannel=True, full=True)
plt.imshow(AE_image)
plt.title('Basic AE \nSSIM: ' + str(round(ssim_index, 3))
          + '\nSize of comp:' + str(sys.getsizeof(model_basic.encoder(img))))
plt.axis('off')


plt.subplot(1, 8, 3)
ae_attention_output = make_img(ae_attention_output)
ssim_index, _ = ssim(orig_img, ae_attention_output, win_size=3, multichannel=True, full=True)
plt.imshow(ae_attention_output)
memory = sys.getsizeof(AE_with_attention.linear(AE_with_attention.encoder(img))) \
         + sys.getsizeof(AE_with_attention.attention(AE_with_attention.encoder(img)))
plt.title('Basic AE + Attention \n SSIM: ' + str(round(ssim_index, 3))
          + '\nSize of comp:' + str(memory))
plt.axis('off')

plt.subplot(1, 8, 4)
interpolate_out = make_img(interpolate_out)
ssim_index, _ = ssim(orig_img, interpolate_out, win_size=3, multichannel=True, full=True)
plt.imshow(interpolate_out)
plt.title('End to end + Interpolation \n SSIM: ' + str(round(ssim_index, 3))
          + '\nSize of comp:' + str(sys.getsizeof(jpeg_interpolation_1)))
plt.axis('off')

plt.subplot(1, 8, 5)
jpeg_out = make_img(jpeg_out)
ssim_index, _ = ssim(orig_img, jpeg_out, win_size=3, multichannel=True, full=True)
plt.imshow(jpeg_out)
plt.title('End to end + JPEG \n SSIM: ' + str(round(ssim_index, 3))
           + '\nSize of comp:' + str(sys.getsizeof(apply_jpeg_on_encoding(
    model_with_jpeg.encoder_f(img).detach().squeeze(0)))))
plt.axis('off')


plt.subplot(1, 8, 6)
attention_out = make_img(attention_out)
ssim_index, _ = ssim(orig_img, attention_out, win_size=3, multichannel=True, full=True)
plt.imshow(attention_out)
plt.title('End to end + JPEG + Attention \n SSIM: ' + str(round(ssim_index, 3))
            + '\nSize of comp:' + str(sys.getsizeof(jpeg_interpolation)))
plt.axis('off')

plt.subplot(1, 8, 7)
jpg_with_attention_output = make_img(jpg_with_attention_output)
ssim_index, _ = ssim(orig_img, jpg_with_attention_output, win_size=3, multichannel=True, full=True)
plt.imshow(jpg_with_attention_output)
plt.title('JPG + Attention \n SSIM: ' + str(round(ssim_index, 3))
          + '\nSize of comp:' + str(sys.getsizeof(apply_jpeg_on_encoding(
            jpg_with_attention.apply_attention(jpg_with_attention.encoder_f(img)).detach().squeeze(0)))))
plt.axis('off')


plt.subplot(1, 8, 8)
FCN_out = make_img(FCN_out)
ssim_index, _ = ssim(orig_img, jpg_with_attention_output, win_size=3, multichannel=True, full=True)
plt.imshow(FCN_out)
plt.title('FCN \n SSIM: ' + str(round(ssim_index, 3))
          + '\nSize of comp:' + str(sys.getsizeof(FCN.get_pretrained_output(img))))
plt.axis('off')

plt.show()
