import matplotlib.pylab as plt
from PIL import Image, ImageFont
from PIL import ImageDraw

from torchvision import transforms

def draw_text(t, text_list):
    img = transforms.ToPILImage()(t).convert("RGB")
    font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-L.ttf", size=18)
    offset = 10
    step = 30
    for i in range(len(text_list)):
        ImageDraw.Draw(
            img  # Image
        ).text(
            (offset , offset+ (i*step)),  # Coordinates
            text_list[i],  # Text
            (255, 0, 0),  # Color
            font=font
        )
    return transforms.ToTensor()(img)
