import cv2
import numpy as np
import json
from openslide    import open_slide, ImageSlide
import matplotlib.pyplot as plt
import json
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import math
import pickle

from tqdm import tqdm
from multiprocessing import Pool
import os 

from skimage.morphology import square
from skimage.color import rgb2gray
from skimage.morphology import binary_closing, binary_opening


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default = 10)
parser.add_argument('--background_threshold', type=int, default = 220)
parser.add_argument('--data_path', type=str, default = 'test_dataset/slides')
parser.add_argument('--list_dict_path', type=str, default = 'test_dataset/patches')
parser.add_argument('--save_path', type=str, default = 'test_dataset/Handcrafted_features/tissue_statistics')
parser.add_argument('--hovernet_json_path', type=str, default = 'test_dataset/Hovernet_output/json')
args = parser.parse_args()


hovernet_mag = 40
patch_extract_mag = 5
patch_mag_ratio = hovernet_mag/patch_extract_mag
patch_size = 224
height = int(patch_size * patch_mag_ratio)
width = int(patch_size * patch_mag_ratio)
use_index = True  # since our patches are saved as row col

background_threshold = args.background_threshold

data_path = args.data_path
save_path = args.save_path
NUM_WORKERS = args.workers
list_dict_path = args.list_dict_path
json_path = args.hovernet_json_path


if not os.path.isdir(save_path.split('tissue_statistics')[0]):
	os.mkdir(save_path.split('tissue_statistics')[0])

if not os.path.isdir(save_path):
	os.mkdir(save_path)

	
with open(os.path.join(list_dict_path, 'all_dict.pickle'), 'rb') as f:
	all_dict = pickle.load(f)

with open(os.path.join(list_dict_path, 'all_list.pickle'), 'rb') as f:
	all_list = pickle.load(f)


final_list = []

for file in os.listdir(json_path):
	if 'json' in file:
		final_list.append(file.split('/')[-1][:-4])


def single_crop_features(slide, cell_contour_list, start_coordinate_list, end_coordinate_list, patch_name):
	
	_, column, row = patch_name.split('/')[-1].split('.')[0].split('_')

	column = int(column) * patch_mag_ratio
	row = int(row) * patch_mag_ratio

	if use_index:
		column = column * patch_size
		row = row * patch_size
	
	start_x_point = int(column)
	stop_x_point= int(column + width)
	start_y_point = int(row)
	stop_y_point =  int(row + height)
	
	wsi_crop_patch = slide.read_region((start_x_point,start_y_point), 0, (stop_x_point-start_x_point, stop_y_point-start_y_point))
	wsi_crop_patch = np.array(wsi_crop_patch)[:,:,:3]

	background_mask = wsi_crop_patch.copy()
	
	background_mask = background_mask.sum(2)/3
	background_mask = np.where(background_mask<background_threshold, background_mask,0)
	background_mask = np.where(background_mask!=0, 1, 0)
	background_mask = binary_closing(background_mask, square(10))
	background_mask = binary_opening(background_mask, square(5))

	x_lower = np.where(start_coordinate_list[:,0]>start_x_point)[0].copy()
	x_upper = np.where(end_coordinate_list[:,0]<stop_x_point)[0].copy()
	y_lower = np.where(start_coordinate_list[:,1]>start_y_point)[0].copy()
	y_upper = np.where(end_coordinate_list[:,1]<stop_y_point)[0].copy()

	x_intersection = np.intersect1d(x_lower, x_upper)
	y_intersection = np.intersect1d(y_lower, y_upper)
	centroids_in_region = np.intersect1d(x_intersection, y_intersection).copy()
	
	contours = cell_contour_list[centroids_in_region].copy()

	wsi_crop_patch_segmentation = Image.new("RGB", (stop_x_point-start_x_point, stop_y_point-start_y_point))

	draw = ImageDraw.Draw(wsi_crop_patch_segmentation)

	for i,_ in enumerate(contours):
		temp_contour = contours[i].copy()
		temp_contour[:,0] = temp_contour[:,0] - start_x_point
		temp_contour[:,1] = temp_contour[:,1] - start_y_point
		points = tuple(map(tuple, temp_contour))

		draw.polygon((points), fill=(255, 0, 0))

	wsi_crop_patch_segmentation = np.array(wsi_crop_patch_segmentation)[:,:,0]
	wsi_crop_patch_segmentation = np.where(wsi_crop_patch_segmentation>0,1,0).astype(np.uint8)

	tissue_mask = (1-wsi_crop_patch_segmentation)*background_mask
	wsi_crop_patch_gray = (rgb2gray(wsi_crop_patch)*255).astype(np.uint8)  # gray
	wsi_crop_patch_gray = cv2.GaussianBlur(wsi_crop_patch_gray, (5, 5), 0)

	tissue_statistics = []
	
	######## percentage features - 3 feat  ########
	cell_percent = (wsi_crop_patch_segmentation).sum()/wsi_crop_patch_segmentation.size
	background_percent = (1-background_mask).sum()/background_mask.size
	
	if cell_percent+background_percent == 1:
		return [None]*num_feat
	
	tissue_statistics.append(cell_percent) # percent of cell region 
	tissue_statistics.append(1-cell_percent-background_percent) # percent of tissue 
	tissue_statistics.append(background_percent) # percent of background 
		
	return tissue_statistics


def run_extraction(file_name):

	svs_file_path = data_path + '/' + file_name + 'svs'
			
	json_file_path = json_path + '/' + file_name + 'json'
	
	save_file_path = save_path + '/' + file_name + 'pickle'
	
	try:
		slide = open_slide(svs_file_path)
	except:
		print('Error opening:', svs_file_path)
		return None

	try:
		slide_mag = int(slide.properties['aperio.AppMag'][:2])
	except:
		return None

	if slide_mag != 40:
		print('Not extracting:', svs_file_path, ' - ', slide_mag)
		return None

	else:
		mag_ratio = hovernet_mag/slide_mag  # will be 1 

		if not os.path.isfile(save_file_path):

			image_patches_list = np.array(all_list)[all_dict[file_name[:-1]][0]:all_dict[file_name[:-1]][1]]
			
			with open(json_file_path) as f:
				pred_data = json.load(f)				

			start_coordinate_list = []
			end_coordinate_list = []
			cell_contour_list = []

			for i in pred_data['nuc'].keys():
				temp_contour = np.array(pred_data['nuc'][i]['contour'])
				cell_contour_list.append(temp_contour)

				x_min = np.array(temp_contour)[:,0].min()
				x_max = np.array(temp_contour)[:,0].max()

				y_min = np.array(temp_contour)[:,1].min()
				y_max = np.array(temp_contour)[:,1].max()

				start_coordinate_list.append([x_min, y_min])
				end_coordinate_list.append([x_max, y_max])

			cell_contour_list = np.array(cell_contour_list, dtype=object)//mag_ratio
			start_coordinate_list = np.array(start_coordinate_list, dtype=object)//mag_ratio
			end_coordinate_list = np.array(end_coordinate_list, dtype=object)//mag_ratio

			final_feature_dict = {}

			for patch_name in tqdm(image_patches_list):
				final_feature_dict[patch_name] = single_crop_features(slide, cell_contour_list, start_coordinate_list, end_coordinate_list, patch_name)


			with open(save_file_path, 'wb') as f:
				pickle.dump(final_feature_dict, f)

	return None


	
def prepare_and_save(file):
	run_extraction(file)

p = Pool(NUM_WORKERS)
print(p.map(prepare_and_save, final_list))

