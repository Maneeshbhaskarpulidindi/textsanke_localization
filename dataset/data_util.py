from PIL import Image
import numpy as np
import cv2

def pil_load_img11(path):
    image = Image.open(path)
    image = np.array(image)
    return image

def pil_load_img(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
    
# from PIL import Image, ImageOps
# import numpy as np

# def pil_load_img(path):
#     # Respect camera EXIF orientation and return RGB numpy
#     img = Image.open(path)
#     img = ImageOps.exif_transpose(img)
#     img = img.convert("RGB")
#     return np.array(img)
