import os
from PIL import Image

sizes = set()
for r, _, fs in os.walk('archive/malimg_dataset'):
    for f in fs:
        if f.endswith('.png'):
            sizes.add(Image.open(os.path.join(r, f)).size)
            if len(sizes) > 10:
                break
print(sizes)
