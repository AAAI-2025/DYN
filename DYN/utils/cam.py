import os
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from torchvision.models.segmentation import deeplabv3_resnet50
import torch
import torch.functional as F
import numpy as np
import requests
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def apply_grad_cam(model, target_layers, targets=None):
    model = model
    target_layers = [target_layers[-1]]
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_path="/data/datasets/ILSVRC/train/n02086646/n02086646_38.JPEG"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img_tensor = data_transform(img)
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    # Grad CAM
    cam = GradCAM(model=model, target_layers=target_layers)#, use_cuda=True
    # targets = [ClassifierOutputTarget(281)]     # cat
    # targets = [ClassifierOutputTarget(259)]  # dog
    # targets = None
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32)/255.,
                                      grayscale_cam, use_rgb=True)

    the_image = Image.fromarray(visualization)
    the_image.save(f'./test.jpg')  

