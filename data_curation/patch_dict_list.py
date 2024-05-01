import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
import time
from tqdm import tqdm

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--patch_path', type=str, default = 'test_dataset/patches')

args = parser.parse_args()

patch_path = args.patch_path

all_dict = dict()
all_list = []

index = 0

for wsi in tqdm(os.listdir(patch_path)):

	if os.path.isdir(patch_path + '/' + wsi):
		all_dict[wsi] = []
		all_dict[wsi].append(index)

		for wsi_crop in os.listdir(patch_path + '/' + wsi):
			if wsi_crop.split('.')[-1] == 'jpg':
				index += 1
				all_list.append(patch_path + '/' + wsi + '/' + wsi_crop)

		all_dict[wsi].append(index)
	

with open(patch_path + '/all_list.pickle', 'wb') as f:
	pickle.dump(all_list, f)

with open(patch_path + '/all_dict.pickle', 'wb') as f:
	pickle.dump(all_dict, f)
