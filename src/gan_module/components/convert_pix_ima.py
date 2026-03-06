# src/

import pandas as pd
import numpy as np
from PIL import Image
import os

df = pd.read_csv("notebook/data/images/fashion_image.csv")

os.makedirs("notebook/data/images_dataset", exist_ok=True)

for i, row in df.iterrows():

    pixels = row[1:].values.reshape(28,28)

    image = Image.fromarray(pixels.astype('uint8'))

    image.save(f"notebook/data/images_dataset/img_{i}.png")