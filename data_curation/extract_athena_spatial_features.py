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
from sklearn.metrics import pairwise_distances
from collections import Counter
from utils import *
from tqdm import tqdm

from multiprocessing import Pool
import os 
from scipy.stats import skew, kurtosis

import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx

from astropy.stats import RipleysKEstimator
import community as community_louvain
from collections import defaultdict
import math
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default = 10)
parser.add_argument('--data_path', type=str, default = 'test_dataset/slides')
parser.add_argument('--cell_properties_path', type=str, default = 'test_dataset/cell_property')
parser.add_argument('--list_dict_path', type=str, default = 'test_dataset/patches')
parser.add_argument('--save_path', type=str, default = 'test_dataset/Handcrafted_features/athena_statistics')

args = parser.parse_args()

NUM_WORKERS = args.workers
hovernet_mag = 40 # default for all experiments in SI-MIL
patch_extract_mag = 5 # defaulf for SI-MIL. Can be changes as required

patch_size = 224  # defaulf for SI-MIL. Can be changes as required

no_of_cell_types = 5 # hovernet pannuke
outlier_removal = 0.05  # removing outliers which could be caused by incorrect segmentations from hovernet.
use_index = True  # since our patches are saved as row col
tumor_id = 1 # tumor id in hovernet prediction


data_path = args.data_path
cell_properties_path = args.cell_properties_path
save_path = args.save_path
list_dict_path = args.list_dict_path


if not os.path.isdir(save_path.split('athena_statistics')[0]):
	os.mkdir(save_path.split('athena_statistics')[0])

	
if not os.path.isdir(save_path):
	os.mkdir(save_path)


with open(os.path.join(list_dict_path, 'all_dict.pickle'), 'rb') as f:
	all_dict = pickle.load(f)

with open(os.path.join(list_dict_path, 'all_list.pickle'), 'rb') as f:
	all_list = pickle.load(f)
	


feat_number=58



# Assuming cell_centroids is your data
# cell_centroids = np.array([[x1, y1], [x2, y2], ..., [xn, yn]])
# You can adjust k and threshold according to your needs.

k = 6
threshold = 200


def create_knn_graph_with_class(cell_centroids, cell_labels, k, threshold):

	nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(cell_centroids)
	distances, indices = nbrs.kneighbors(cell_centroids)

	G = nx.Graph()
	# Add nodes with label information
	for i, centroid in enumerate(cell_centroids):
		G.add_node(i, pos=centroid, label=cell_labels[i])
	# Add edges based on distance threshold
	for i in range(cell_centroids.shape[0]):
		for j in range(1, k+1):
			# if distances[i][j] <= threshold:
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

def single_crop_features(key_list, cell_centroid_list, type_list, property_list, patch_name, patch_mag_ratio, height, width):
	
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
	
	points = points[points_type>0]   # removing none cell class type
	points_keys = points_keys[points_type>0] 
	points_properties = points_properties[points_type>0] 
	points_type = points_type[points_type>0] 

	if points.shape[0] <= k+1:   # change to None
		return [0]*feat_number  
	

	G = create_knn_graph_with_class(points, points_type, k, threshold)

	athena_feat = [] 
	
	# https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/bioinformatics/38/11/10.1093_bioinformatics_btac303/2/btac303_supplementary_data.pdf?Expires=112871218517&Signature=CeJtoFacx7bJ7wPfBhcz5b5qkIAMGypxY37~sUIMpBJ3JFqLcsDyB-QeJ2vt0sL-x5TooZMR7ZiYTT53WVenuR8jAn1227k7KOudHB1NNrBz~pumYN-aLqGDIXJgX0a12LotJ7E123c2oZ3LVmmU9nht94P7WnOPKhLJqQHKl-N3rsU4S02wI98uY8u8WDGc8nBES12HX3AN5v-DHJ4Ss7SC3cATQohENdkcXJ1gCaiHAGVVlahqnoKmbnPGtWNOBSXj21NoIsvPaMbzGNzJjR9ZM~IC12Bn0JLsRmNKXoz8eMYBBy~0MB4124wIu1FVlsAiy09hdacbTF4mpdiY127bxaLAQ__&Key-Pair-Id=APKAIE5G5CRDK12RD3PGA
	
	# ************ 25 Spatial statistics scores - k function features (class wise) ************
	
	rkE = RipleysKEstimator(area=float(height*width),
							 x_max=float(width), x_min=0,
							 y_max=float(height), y_min=0)

	radii = np.linspace(0, min(height, width) / 2, 5)
	k_function_feat = []
	for point_class in range(1,no_of_cell_types+1):
		# if we have no observations of the given id, K is zero
		if len(points[points_type==point_class]) > 100:   # if less points then not good enough k-function
			K = rkE(data=points[points_type==point_class], radii=radii, mode='ripley')
			L = np.sqrt(K / np.pi)  # transform, to stabilise variance
			res = L - radii
			k_function_feat.extend(res.tolist())
			
		else:
			k_function_feat.extend([None]*len(radii))
		
	athena_feat.extend(k_function_feat)   # 5 bins * 5 cell types = 25 features
	
	
	# ************ 1 Graph-theoretic scores - modularity ************
	
	
	# Form communities based on labels
	communities = {}
	for i in range(1,no_of_cell_types+1):
		communities[i] = []
	for node, data in G.nodes(data=True):
		communities[data['label']].append(node)
	communities = list(communities.values())

	athena_feat.append(nx.community.modularity(G, communities))  # 1 features
	

	# ************ 4 Global Information-theoretic scores - Richeness, Shannon index, Simpson index, Renyi entropy ************   
	
	#, Rao’s quadratic entropy
	# hill number not more informative. Already covers richness, shannon index, simpson index
	# for renqi, not taking 0,1,2 as it already covers in richness, shannon index, simpson index
	
	richness = len(np.unique(points_type))
	probabilities = np.unique(points_type, return_counts=True)[1]/ len(points_type)
	shannon_global = -(probabilities * np.log2(probabilities)).sum()
	simpson_global = (probabilities**2).sum()
	renyi_global_infi = -math.log(probabilities.max(), 2)  # (Min-entropy)
	
	
# 	distances = pairwise_distances(points/[width,height])
# 	label_counter = Counter(points_type)
# 	relative_abundance = np.array([label_counter[j] / len(points_type) for j in range(1, 1+no_of_cell_types)])
# 	points_relative_abundance = relative_abundance[points_type-1].reshape(-1,1)
# 	raos_quadratic_entropy = ((points_relative_abundance.T@distances)@points_relative_abundance/2) 

	
	athena_feat.append(richness)  # 1 features
	athena_feat.append(shannon_global)  # 1 features
	athena_feat.append(simpson_global)  # 1 features
	athena_feat.append(renyi_global_infi)  # 1 features
	# athena_feat.append(raos_quadratic_entropy)  # 1 features
	
	
	# ************ 20 Local Information-theoretic scores - Richeness, Shannon index, Simpson index, Renyi entropy ************  

	richness_per_cell = []
	shannon_per_cell = []
	simpson_per_cell = []
	renyi_local_infi_per_cell = []

	for i in range(len(points)):
		local_points_ids = list(G.neighbors(i))  
		local_points_ids.append(i)
		local_points_types = [points_type[j] for j in local_points_ids]  # Assuming you have a cell_types list
		probabilities = np.unique(local_points_types, return_counts =True)[1] / len(local_points_ids)

		richness_local = len(np.unique(local_points_types))
		shannon_local = -(probabilities * np.log2(probabilities)).sum()
		simpson_local = (probabilities**2).sum()
		renyi_local_infi = -math.log(probabilities.max(), 2)  # (Min-entropy)

		richness_per_cell.append(richness_local)
		shannon_per_cell.append(shannon_local)
		simpson_per_cell.append(simpson_local)
		renyi_local_infi_per_cell.append(renyi_local_infi)

	richness_local_hist = np.array([Counter(richness_per_cell)[j] for j in range(1, 1+no_of_cell_types)]).tolist()
	shannon_local_hist = np.histogram(shannon_per_cell, bins=5, range=(0, -(1/no_of_cell_types * np.log2(1/no_of_cell_types)) * no_of_cell_types))[0].tolist()
	simpson_local_hist = np.histogram(simpson_per_cell, bins=5, range=((np.array([1/no_of_cell_types]*no_of_cell_types)**2).sum(), 1))[0].tolist()
	renyi_local_infi_hist = np.histogram(renyi_local_infi_per_cell, bins=5, range=(0, -math.log(1/no_of_cell_types, 2)))[0].tolist()

	athena_feat.extend(richness_local_hist)  # 5 features
	athena_feat.extend(shannon_local_hist)  # 5 features
	athena_feat.extend(simpson_local_hist)  # 5 features
	athena_feat.extend(renyi_local_infi_hist)  # 5 features
	
	# ************ 8 Global Cell interaction scores - Infiltration, Classic, HistoCAT, Proportion ************   
	# Infiltration
	A = nx.adjacency_matrix(G)
	A = A.toarray()

	# infiltration_score_list = [-1] * (no_of_cell_types*no_of_cell_types - no_of_cell_types)  # -1 represents no cell type i-i interaction
	
	infiltration_score_list = [None] * ((no_of_cell_types-1)*2)  # -1 represents no cell type i-i interaction # only tumor with all other infiltration vice-versa
	
	number_dict = {}
	for i in range(1,no_of_cell_types+1):
		for j in range(i, no_of_cell_types+1):
			number_dict[i,j] = A[np.where(points_type==i)[0]][:,np.where(points_type==j)[0]].sum()
			if i != j:
				number_dict[j,i] = number_dict[i,j]
			else:
				number_dict[i,j] = number_dict[i,j]/2
	counter = 0
	
# 	for i in range(1,no_of_cell_types+1):
# 		if number_dict[i,i] == 0:
# 			counter += no_of_cell_types-1
# 			continue
# 		for j in range(1, no_of_cell_types+1):      
# 			if i == j:
# 				continue
# 			infiltration_score_list[counter] = number_dict[i,j]/number_dict[i,i]
# 			counter += 1
	
	counter = 0
	if number_dict[tumor_id,tumor_id] == 0:
		counter += no_of_cell_types-1
	else:
		for j in range(1, no_of_cell_types+1):      
			if tumor_id == j:
				continue
			infiltration_score_list[counter] = number_dict[tumor_id,j]/number_dict[tumor_id,tumor_id]
			counter += 1

	for j in range(1, no_of_cell_types+1):      
		if tumor_id == j:
			continue
		if number_dict[j,j] == 0:
			counter += 1
			continue
			
		infiltration_score_list[counter] = number_dict[j,tumor_id]/number_dict[j,j]
		counter += 1

	
		
	athena_feat.extend(infiltration_score_list) # 8 features
	
	
	
	# ************ Local Cell interaction scores - Infiltration ************   
	# TBD

	# Neighborhood analysis score
	# TBD
	
	return athena_feat



def run_extraction(cell_pickle_path):
	

	slide = open_slide(data_path + '/' + cell_pickle_path[:-6] + 'svs')
	
	try:
		slide_mag = int(slide.properties['aperio.AppMag'][:2])
		patch_mag_ratio = slide_mag/patch_extract_mag
		height = int(patch_size * patch_mag_ratio)
		width = int(patch_size * patch_mag_ratio)

	except:
		return None
	
	# if slide_mag != 40:
	# 	print('Not extracting:', cell_pickle_path, ' - ', slide_mag)
	# 	return None

	# else:
	if not os.path.isfile(save_path + '/' + cell_pickle_path):
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
			final_feature_dict[patch_name] = single_crop_features(key_list, cell_centroid_list, type_list, property_list, patch_name, patch_mag_ratio, height, width)

		with open(save_path + '/' + cell_pickle_path, 'wb') as f:
			pickle.dump(final_feature_dict, f)

	return None

def prepare_and_save(file):
	run_extraction(file)

p = Pool(NUM_WORKERS)
print(p.map(prepare_and_save, os.listdir(cell_properties_path)[::-1]))

