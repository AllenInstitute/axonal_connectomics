from acpreprocessing import downsample_tiffs_and_extract_metadata
import json
import random
from multiprocessing import Pool



#configs/config_downsample_tifs_487748_36_NeuN_NFH488_25X_1XPBS.json

with open("configs/config_downsample_tifs_487748_36_NeuN_NFH488_25X_1XPBS.json", 'r') as j:
     cfg = json.loads(j.read())

random.shuffle(cfg)

with Pool(5) as p:
	p.map(downsample_tiffs_and_extract_metadata.downsample_tiffs_and_extract_metadata, cfg)


#for c in cfg:
#	downsample_tiffs_and_extract_metadata.downsample_tiffs_and_extract_metadata(c)
