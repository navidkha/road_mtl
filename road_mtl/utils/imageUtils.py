import matplotlib.pylab as plt
from PIL import Image, ImageFont
from PIL import ImageDraw
import numpy as np
from torchvision import transforms


def _list_to_tuple(lst):
    # lst.len is 4
    return ((lst[0], lst[1]), ) + ((lst[2], lst[3]), )

def draw_text_box(img_tensor, text_list_pred, text_list_lbl, box_list_pred, box_list_lbl):

    img = img_tensor.permute(1, 2, 0).numpy()
    img_width = img.shape[1]
    img = Image.fromarray(
        (((img - img.min()) / (-img.min() + img.max())) * 255).astype(np.uint8)
    )
    draw = ImageDraw.Draw(img)
    # load font
    font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-L.ttf", size=11)
    color_pred = (255, 0, 0)

    cords_pred = []
    for i in range(len(box_list_pred)):
        cords_pred.append(_list_to_tuple(box_list_pred[i]))

    for i in range(len(text_list_pred)):
        draw.text(
            (cords_pred[i][0][0], cords_pred[i][1][1]),  # Coordinates
            text_list_pred[i],  # Text
            color_pred,  # Color
            font=font
        )
        draw.rectangle(_list_to_tuple(box_list_pred[i]), outline=color_pred)

    cords_lbl = []
    for i in range(len(box_list_lbl)):
        cords_lbl.append(_list_to_tuple(box_list_lbl[i]))
    color_lbl = (0, 255, 0)
    for i in range(len(text_list_lbl)):
        draw.text(
            (cords_lbl[i][0][0], cords_lbl[i][1][1]),  # Coordinates
            text_list_lbl[i],  # Text
            color_lbl,  # Color
            font=font
        )
        draw.rectangle(cords_lbl[i], outline=color_lbl)
    return img


# def draw_on_image(img, measurements: dict, action):
#     """Draw text on the image
#
#     Args:
#         img: (torch.Tensor) frame
#         measurements: (dict) ground truth values
#         action: (torch.Tensor) predicted actions
#     """
#     control = measurements["control"]
#     speed = measurements["speed"]
#     command = measurements["command"]
#     steer = action[0].item()
#     pedal = action[1].item()
#     if pedal > 0:
#         throttle = pedal
#         brake = 0
#     else:
#         throttle = 0
#         brake = -pedal
#
#     steer_gt = control[0]
#     pedal_gt = control[1]
#     if pedal_gt > 0:
#         throttle_gt = pedal_gt
#         brake_gt = 0
#     else:
#         throttle_gt = 0
#         brake_gt = -pedal_gt
#
#     img = img.permute(1, 2, 0).numpy()
#     img_width = img.shape[1] // 2
#     img = Image.fromarray(
#         (((img - img.min()) / (-img.min() + img.max())) * 255).astype(np.uint8)
#     )
#     draw = ImageDraw.Draw(img)
#     # load font
#     fnt = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-L.ttf", size=14)
#     draw.text((5, 10), "Speed: %.3f" % speed, fill=(0, 255, 0, 255), font=fnt)
#     draw.text((5, 30), "Steer: %.3f" % steer, fill=(255, 0, 0, 255), font=fnt)
#     draw.text((5, 50), "Throttle: %.3f" % throttle, fill=(255, 0, 0, 255), font=fnt)
#     draw.text((5, 70), "Brake: %.3f" % brake, fill=(255, 0, 0, 255), font=fnt)
#
#     draw.text(
#         (img_width, 10),
#         "Command: %i" % command.argmax(),
#         fill=(0, 255, 0, 255),
#         font=fnt,
#     )
#     draw.text(
#         (img_width, 30), "Steer (GT): %.3f" % steer_gt, fill=(0, 255, 0, 255), font=fnt
#     )
#     draw.text(
#         (img_width, 50),
#         "Throttle (GT): %.3f" % throttle_gt,
#         fill=(0, 255, 0, 255),
#         font=fnt,
#     )
#     draw.text(
#         (img_width, 70), "Brake (GT): %.3f" % brake_gt, fill=(0, 255, 0, 255), font=fnt
#     )
#
#     return img
