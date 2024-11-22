# pip install torch torchvision diffusers pillow
import torch # PyTorch，幫助處理神經網路運算
from torchvision import transforms # 用來對圖片進行縮放、標準化等處理
from diffusers import DDPMPipeline # diffusers 庫提供的工具，用來載入和使用擴散模型

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transfer learning # 使用"google/ddpm-cifar10-32"
from torchvision.transforms import ToPILImage
from torchvision.transforms import  ToTensor
from PIL import Image
import os
# 使用預訓練的CIFAR-10擴散模型
model = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to("cuda")


