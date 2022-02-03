import torch
import torchvision
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import time
import os
from pprint import pprint

from src.models import PretrainedUNet

from optparse import OptionParser
args=None
parser = OptionParser()

parser.add_option("--input-dir", dest="input_dir", help="input dir path")
parser.add_option("--tmp-dir", dest="tmp_dir", help="tmp dir path")

class_names = ['COVID19', 'NORMAL', 'PNEUMONIA']

options, args = parser.parse_args(args)

input_dir_path = options.input_dir
tmp_data_dir = options.tmp_dir

data_dir = input_dir_path.rstrip('/')
tmp_data_dir = os.path.join(tmp_data_dir.rstrip('/'), str(int(time.time()*10000)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if not os.path.exists(tmp_data_dir):
    os.makedirs(tmp_data_dir)

# image segmentation
unet = PretrainedUNet(
            in_channels=1,
            out_channels=2, 
            batch_norm=True, 
            upscale_mode="bilinear"
        )

seg_model_name = "unet-6v.pt"
unet.load_state_dict(torch.load(os.path.join('models', seg_model_name), map_location=torch.device("cpu")))
unet.to(device)
unet.eval()

def image_seg(image_path):
    origin = Image.open(image_path).convert("P")
    origin = torchvision.transforms.functional.resize(origin, (512, 512))
    origin = torchvision.transforms.functional.to_tensor(origin) - 0.5
    with torch.no_grad():
        origin = torch.stack([origin])
        origin = origin.to(device)
        out = unet(origin)
        softmax = torch.nn.functional.log_softmax(out, dim=1)
        out = torch.argmax(softmax, dim=1)
        
        origin = origin[0].to("cpu")
        out = out[0].to("cpu")

    trimed = origin*out
    img = torchvision.transforms.functional.to_pil_image(trimed + 0.5).convert("RGB")
    return img

for root, dirs, files in os.walk(data_dir):
    new_root = root.replace(data_dir, tmp_data_dir, 1)
    for dirname in dirs:
        dir_path = os.path.join(new_root, dirname)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    for filename in files:
        origin_path = os.path.join(root, filename)
        file_path = os.path.join(new_root, filename)
        img = image_seg(origin_path)
        img.save(file_path)

model_eval = torch.load('models/model_resnet18_pretrained_covid_pneumonia_normal.pth', map_location='cpu')
#model_eval.to(device)
model_eval.eval()
model_eval_ff = torch.load('models/model_resnet18_pretrained_fixed_ft_covid_pneumonia_normal.pth', map_location='cpu')
#model_eval_ff.to(device)
model_eval_ff.eval()

loader = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name).convert("RGB")
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image

result_model1 = {'predict': {}, 'count': {'COVID19': 0, 'NORMAL': 0, 'PNEUMONIA': 0}}
result_model2 = {'predict': {}, 'count': {'COVID19': 0, 'NORMAL': 0, 'PNEUMONIA': 0}}

for item in os.listdir(tmp_data_dir):
    image_path = os.path.join(tmp_data_dir, item)
    if os.path.isfile(image_path):
        image = image_loader(image_path)

        model_output = model_eval(image)
        pred = np.argmax(model_output.detach().numpy())
        class_name = class_names[pred]
        result_model1['predict'][item] = class_name
        curent_num = result_model1['count'][class_name]
        new_num = curent_num + 1
        result_model1['count'][class_name] = new_num

        model_output_ff = model_eval_ff(image)
        pred_ff = np.argmax(model_output_ff.detach().numpy())
        class_name_ff = class_names[pred_ff]
        result_model2['predict'][item] = class_name_ff
        curent_num = result_model2['count'][class_name_ff]
        new_num = curent_num + 1
        result_model2['count'][class_name_ff] = new_num


print('model1 result:')
pprint(result_model1)
print('\nmodel2 result:')
pprint(result_model2)
