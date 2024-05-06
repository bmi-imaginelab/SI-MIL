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

from utils import *
from tqdm import tqdm

from multiprocessing import Pool
import os 
from scipy.stats import skew, kurtosis

import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default = 10)
parser.add_argument('--data_path', type=str, default = 'test_dataset/slides')
parser.add_argument('--cell_properties_path', type=str, default = 'test_dataset/cell_property')
parser.add_argument('--list_dict_path', type=str, default = 'test_dataset/patches')
parser.add_argument('--save_path', type=str, default = 'test_dataset/Handcrafted_features/sna_statistics')

args = parser.parse_args()

NUM_WORKERS = args.workers
hovernet_mag = 40 # default for all experiments in SI-MIL
patch_extract_mag = 5 # defaulf for SI-MIL. Can be changes as required
patch_mag_ratio = hovernet_mag/patch_extract_mag
patch_size = 224  # defaulf for SI-MIL. Can be changes as required
height = int(patch_size * patch_mag_ratio)
width = int(patch_size * patch_mag_ratio)
no_of_cell_types = 5 # hovernet pannuke
outlier_removal = 0.05  # removing outliers which could be caused by incorrect segmentations from hovernet.
use_index = True  # since our patches are saved as row col

data_path = args.data_path
cell_properties_path = args.cell_properties_path
save_path = args.save_path
list_dict_path = args.list_dict_path


if not os.path.isdir(save_path.split('sna_statistics')[0]):
	os.mkdir(save_path.split('sna_statistics')[0])

	
if not os.path.isdir(save_path):
	os.mkdir(save_path)


with open(os.path.join(list_dict_path, 'all_dict.pickle'), 'rb') as f:
	all_dict = pickle.load(f)

with open(os.path.join(list_dict_path, 'all_list.pickle'), 'rb') as f:
	all_list = pickle.load(f)
	


feat_number = (6*4)   # 6 statistics on 4 set of SNA features


# Assuming cell_centroids is your data
# cell_centroids = np.array([[x1, y1], [x2, y2], ..., [xn, yn]])
# You can adjust k according to your needs.

k = 6  # knn graph. Can be changed as required.

def create_knn_graph(cell_centroids, k):

	nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(cell_centroids)
	distances, indices = nbrs.kneighbors(cell_centroids)
	G = nx.Graph()
	for i in range(cell_centroids.shape[0]):
		for j in range(1, k+1):
			G.add_edge(i, indices[i][j])
	return G


def calculate_features(values):
	
	values = np.sort(values)
	n = int(outlier_removal * len(values))
	values = values[n:len(values)-n]
	
	# Calculate statistics
	max_value = np.max(values)
	min_value = np.min(values)
	mean_value = np.mean(values)
	std_value = np.std(values)
	skew_value = skew(values)
	kurtosis_value = kurtosis(values)
	
	return [max_value, min_value, mean_value, std_value, skew_value, kurtosis_value]

def single_crop_features(key_list, cell_centroid_list, type_list, property_list, patch_name):
	
	_, column, row = patch_name.split('/')[-1].split('.')[0].split('_')

	column = int(column) * patch_mag_ratio
	row = int(row) * patch_mag_ratio

	if use_index:
		column = column * patch_size
		row = row * patch_size
	
	start_x_point = column
	stop_x_point= column + width
	start_y_point = row
	stop_y_point =  row + height

	x_lower = np.where(cell_centroid_list[:,0]>start_x_point)[0].copy()
	x_upper = np.where(cell_centroid_list[:,0]<stop_x_point)[0].copy()
	y_lower = np.where(cell_centroid_list[:,1]>start_y_point)[0].copy()
	y_upper = np.where(cell_centroid_list[:,1]<stop_y_point)[0].copy()

	x_intersection = np.intersect1d(x_lower, x_upper)
	y_intersection = np.intersect1d(y_lower, y_upper)
	centroids_in_region = np.intersect1d(x_intersection, y_intersection).copy()


	points = cell_centroid_list[centroids_in_region].copy()
	points_keys = key_list[centroids_in_region].copy()
	points_type = type_list[centroids_in_region].copy()
	points_properties = property_list[centroids_in_region].copy()

	points[:,0] -= start_x_point
	points[:,1] -= start_y_point
	
	if points.shape[0] <= k+1:
		return [None]*feat_number 
	
	G = create_knn_graph(points, k)

	# Katz Centrality
	# katz = nx.katz_centrality(G, max_iter=1000)
	# Node Degree
	degree = dict(G.degree())
	# Clustering Coefficient
	clustering_coefficient = nx.clustering(G)
	# Closeness Centrality
	closeness_centrality = nx.closeness_centrality(G)
	# Degree Centrality
	degree_centrality = nx.degree_centrality(G)
	# Eigen Vector Centrality
	# eigen_vector_centrality = nx.eigenvector_centrality(G, max_iter=1000)

	# Assume the SNA measures are stored in a dictionary called measures
	measures = {
		# 'katz': list(katz.values()),
		'degree': list(degree.values()),
		'clustering_coefficient': list(clustering_coefficient.values()),
		'closeness_centrality': list(closeness_centrality.values()),
		'degree_centrality': list(degree_centrality.values()),
		# 'eigen_vector_centrality': list(eigen_vector_centrality.values()),
	}

	sna_feat = []
	for measure_name, measure_values in measures.items():
		sna_feat.extend(calculate_features(measure_values))

	return sna_feat

def run_extraction(cell_pickle_path):
	
	slide = open_slide(data_path + '/' + cell_pickle_path[:-6] + 'svs')
	
	try:
		slide_mag = int(slide.properties['aperio.AppMag'][:2])
	except:
		return None
		
	if slide_mag != 40:
		print('Not extracting:', cell_pickle_path, ' - ', slide_mag)
		return None

	else:
		if not os.path.isfile(save_path + '/' + cell_pickle_path):
		# if not None:
			print(cell_pickle_path)

			image_patches_list = np.array(all_list)[all_dict[cell_pickle_path[:-7]][0]:all_dict[cell_pickle_path[:-7]][1]]

			with open(cell_properties_path + '/' + cell_pickle_path, 'rb') as f:
				cell_prop = pickle.load(f)

			cell_centroid_list = []
			type_list = []
			property_list = []
			key_list = []
			for i in cell_prop.keys():
				type_list.append(np.array(cell_prop[i]['type']))
				cell_centroid_list.append(np.array(cell_prop[i]['centroid']))
				property_list.append(np.array(cell_prop[i]['properties']))

			key_list = np.array(list(cell_prop.keys()))
			cell_centroid_list = np.round(np.array(cell_centroid_list))
			type_list = np.array(type_list)
			property_list = np.array(property_list)

			final_feature_dict = {}

			for patch_name in tqdm(image_patches_list):
				final_feature_dict[patch_name] = single_crop_features(key_list, cell_centroid_list, type_list, property_list, patch_name)

			with open(save_path + '/' + cell_pickle_path, 'wb') as f:
				pickle.dump(final_feature_dict, f)

	return None

def prepare_and_save(file):
	run_extraction(file)

	

p = Pool(NUM_WORKERS)
print(p.map(prepare_and_save, os.listdir(cell_properties_path)[::]))

