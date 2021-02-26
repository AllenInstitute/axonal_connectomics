from acpreprocessing import deskewing
import json

with open("configs/example_deskew.json", 'r') as j:
     cfg = json.loads(j.read())


for c in cfg:
	deskewing.deskew_and_save_tiff(c)
