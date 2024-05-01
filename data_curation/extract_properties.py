import cv2
import numpy as np
import json
from openslide    import open_slide, ImageSlide
import matplotlib.pyplot as plt
import json
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from skimage.measure import regionprops, label, regionprops_table
import math
import pickle
import os
from multiprocessing import Pool
import shutil
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from tqdm import tqdm
from skimage.color import rgb2gray
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default = 10)
parser.add_argument('--data_path', type=str, default = 'test_dataset/slides')
parser.add_argument('--json_path', type=str, default = 'test_dataset/Hovernet_output/json')
parser.add_argument('--save_path', type=str, default = 'test_dataset/cell_property')

args = parser.parse_args()

NUM_WORKERS = args.workers
hovernet_mag = 40

data_path = args.data_path
json_path = args.json_path
save_path = args.save_path


if not os.path.isdir(save_path):
	os.mkdir(save_path)


final_list = []

for file in os.listdir(json_path):
	if 'json' in file:
		final_list.append(file.split('/')[-1][:-4])

print(len(final_list))

def run_extraction(file_name):
	svs_file_path = None
	
	if os.path.isfile(data_path + '/' + file_name + 'svs'):
		svs_file_path = data_path  + '/' + file_name + 'svs'

	json_file_path = json_path + '/' + file_name + 'json'
	save_pickle_file_path = save_path + '/' + file_name + 'pickle'

	if svs_file_path != None:

		if not os.path.isfile(save_pickle_file_path):
		# if True:
			print(file_name)

			try:
				slide = open_slide(svs_file_path)
				slide_mag = int(slide.properties['aperio.AppMag'][:2])
				mag_ratio = hovernet_mag/slide_mag

				with open(json_file_path) as f:
					pred_data = json.load(f)

				if len(list(pred_data['nuc'].keys()))==0:
					print('No nuclei in the image, skip')

				elif len(list(pred_data['nuc'].keys()))>=1:
					prop_dict = {}
					for keys in tqdm(list(pred_data['nuc'].keys())):
					# for keys in list(pred_data['nuc'].keys())[:10]:

						temp_contour = np.array(pred_data['nuc'][keys]['contour']).copy()//mag_ratio

						x_min = int(temp_contour[:,0].min()) -1
						x_max = int(temp_contour[:,0].max()) +1
						y_min = int(temp_contour[:,1].min()) -1
						y_max = int(temp_contour[:,1].max()) +1

						wsi_cell_crop = slide.read_region((x_min,y_min), 0, (x_max-x_min, y_max-y_min))
						wsi_cell_crop = np.array(wsi_cell_crop)[:,:,:3]

						cell_image = Image.new("RGB", (x_max-x_min, y_max-y_min))

						draw = ImageDraw.Draw(cell_image)

						temp_contour[:,0] = temp_contour[:,0] - x_min
						temp_contour[:,1] = temp_contour[:,1] - y_min

						draw.polygon((tuple(map(tuple, temp_contour))), fill=(1))

						cell_image = np.array(cell_image)[:,:,0]

						# shape features
						properties_list = ["area", "perimeter", "orientation", "eccentricity", "solidity", "axis_major_length", "axis_minor_length"]

						cell_property = regionprops_table(cell_image, intensity_image=wsi_cell_crop, properties=properties_list)

						temp_prop_list = np.array(list(cell_property.values())).reshape(-1).tolist()

						wsi_cell_crop = (rgb2gray(wsi_cell_crop)*255).astype(np.uint8)

						# intensity features
						wsi_cell_crop_masked = wsi_cell_crop[np.where(cell_image==1)[0], np.where(cell_image==1)[1]].copy()
						temp_prop_list.append(wsi_cell_crop_masked.mean())
						temp_prop_list.append(wsi_cell_crop_masked.std())
						temp_prop_list.append(skew(wsi_cell_crop_masked))
						temp_prop_list.append(kurtosis(wsi_cell_crop_masked))

						# texture features					
						glcm = graycomatrix(wsi_cell_crop*cell_image, 
							distances=[1], angles=[0], levels=256)

						temp_prop_list.extend([graycoprops(glcm, 'contrast')[0][0], graycoprops(glcm, 'dissimilarity')[0][0], graycoprops(glcm, 'homogeneity')[0][0], graycoprops(glcm, 'energy')[0][0]])

						prop_dict[keys] = {}
						prop_dict[keys]['centroid'] = np.array(pred_data['nuc'][keys]['centroid'])//mag_ratio
						prop_dict[keys]['type'] = pred_data['nuc'][keys]['type']
						prop_dict[keys]['properties'] = temp_prop_list


					with open(save_pickle_file_path, 'wb') as f:
						pickle.dump(prop_dict, f)

					del prop_dict
					del pred_data

			except:
				print('error loading: ' + svs_file_path)
			
	else:
		print('error loading: ' + file_name)
		
def prepare_and_save(file):
	run_extraction(file)


p = Pool(NUM_WORKERS)
print(p.map(prepare_and_save, final_list))


