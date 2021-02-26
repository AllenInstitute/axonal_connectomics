from acpreprocessing import downsampling
import json
import random
from multiprocessing import Pool


with open("configs/example_ds_gif.json", 'r') as j:
     cfg = json.loads(j.read())


for c in cfg:
	downsampling.downsample_gif(c)
