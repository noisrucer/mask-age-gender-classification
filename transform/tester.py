from transforms import Transform
import numpy as np
from PIL import Image

trsfm = Transform(2)()
img_np = np.random.randint(0,255, (300,300,3), dtype=np.uint8)
img_pil = Image.fromarray(img_np)
out = trsfm(image=img_pil)['image']
print(out)
print(out.shape)
