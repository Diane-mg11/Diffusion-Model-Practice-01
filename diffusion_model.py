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
# 先訓練我的擴散模型 (遷移學習)
# 使用預訓練模型
model = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to("cpu")


# 載入已有的 PCB 瑕疵照片，進行預處理
def preprocess_pcb_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = ToTensor()  # 將圖片轉為張量
    image = transform(image).unsqueeze(0).to("cpu") * 2 - 1  # 正規化到 [-1, 1]
    return image

# 加入原始PCB照片輔助生成
def generate_augmented_images(pcb_image_path, num_variations):
    original_image = preprocess_pcb_image(pcb_image_path)
    generated_images = []
    for _ in range(num_variations):
        noise = torch.randn_like(original_image)  # 根據原始大小生成噪聲
        noisy_image = original_image + noise * 0.5  # 加入噪聲
        generated_image = model(noisy_image)["sample"]
        generated_images.append(generated_image)
    return generated_images

# 測試生成
pcb_image_path = r"C:\Users\dream\Desktop\projects\Diffussion\train"
generated_pcb_images = generate_augmented_images(pcb_image_path, num_variations=5)
save_path = r"C:\Users\dream\Desktop\projects\Diffusion\200"
# note. Python 可能會將 \t 解釋為一個 Tab 符號，導致路徑無法正確識別
# 加上 r 後，路徑會被正確解析為字串中的每個字元，而不會解讀為特殊符號


# 保存生成的圖像
os.makedirs(save_path, exist_ok=True)
for idx, img in enumerate(generated_pcb_images):
    img = (img.squeeze().cpu() * 0.5 + 0.5).clamp(0, 1)  # 還原數據範圍
    ToPILImage()(img).save(os.path.join(save_path, f"pcb_generated_{idx}.png"))




