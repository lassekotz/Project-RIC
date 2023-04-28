import numpy as np
from pathlib import Path
import os
from PIL import Image

i = 0
for file in os.listdir("Data/BigDataset/images"):
    image = Image.open("Data/BigDataset/images/" + file)
    array = np.array(image)
    array = np.expand_dims(array, 0)

    if i == 0:
        ds = array
    else:
        ds = np.append(ds, array, 0)

    if i > 500:
        break
    i += 1

np.save("custom_sample_ds.npy", ds)