import matplotlib.pylab as plt
from PIL import Image, ImageFont
from PIL import ImageDraw

from torchvision import transforms

def draw_text(t, text):
    img = transforms.ToPILImage()(t).convert("RGB")
    font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-L.ttf", size=36)
    ImageDraw.Draw(
        img  # Image
    ).text(
        (100, 100),  # Coordinates
        text,  # Text
        (255, 0, 0),  # Color
        font=font
    )
    #plt.imshow(img)
    #plt.show()
    return img
