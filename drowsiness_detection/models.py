import torch
import torch.nn as nn
import torchvision.models as models

model_conv = models.mobilenet_v3_large()
model_conv.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
# Replace the last layer to output the correct number of classes
model_conv.classifier = nn.Sequential(
    nn.Linear(in_features=960, out_features=1280, bias=True),
    nn.Hardswish(),
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=1000, bias=True),
    nn.Linear(in_features=1000, out_features=2, bias=True),
)
# Load the model
model_conv.load_state_dict(torch.load('proiect_acabi_achizitii_mixed_mobile_net.pth'))

model_conv.eval()

scripted_module = torch.jit.script(model_conv)

# Export full jit version model (not compatible lite interpreter), leave it here for comparison
scripted_module.save("full_jit_scripted.pt")
# Export lite interpreter version model (compatible with lite interpreter)
scripted_module._save_for_lite_interpreter("lite_interp_ver.ptl")