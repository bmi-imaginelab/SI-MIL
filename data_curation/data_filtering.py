import pickle
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--feat_path', type=str, default = 'test_dataset/Handcrafted_features')
parser.add_argument('--save_path', type=str, default = 'test_dataset')
parser.add_argument('--list_dict_path', type=str, default = 'test_dataset/patches')
parser.add_argument('--bins', type=int, default = 10) 
parser.add_argument('--norm_feat', type=str, default = 'bin')

args = parser.parse_args()

save_path = args.save_path
norm_feat = args.norm_feat
bins = args.bins


features_csv = pd.read_csv(os.path.join(feat_path, 'cell_athena_sna.csv'))

dataset_split_path = os.path.join(list_dict_path, 'train_dict.pickle')
dataset_split_path_test = os.path.join(list_dict_path, 'test_dict.pickle')


with open(dataset_split_path, 'rb') as f:
	dataset_split_dict = pickle.load(f)

with open(dataset_split_path_test, 'rb') as f:
	dataset_split_dict_test = pickle.load(f)

	
features_csv = features_csv.set_index('Unnamed: 0')

print(features_csv.shape)

	
bags_path = list(dataset_split_dict.keys())
np.random.seed(0)
np.random.shuffle(bags_path)

train_path = bags_path


train_index_select_list = []
for i in bags_path:
	train_index_select_list.extend(list(np.arange(dataset_split_dict[i][0], dataset_split_dict[i][1])))
print(features_csv.shape, features_csv.iloc[train_index_select_list].shape)


normalized_df = features_csv.copy()
for i, col in tqdm(enumerate(features_csv.columns)):
	try:
		_, intervals = pd.qcut(features_csv.iloc[train_index_select_list, i], bins, labels=False, retbins=True, duplicates='drop')

		normalized_df[col] = pd.cut(features_csv[col], bins=intervals, labels=False, include_lowest=True)


	except:
		normalized_df[col] = -1

back_percentage = np.where(normalized_df["percent_of_background_region"]<7)[0]

cell_percentage = np.where(normalized_df["percent_of_cell_region"]>1)[0]


low_cell_patches = np.intersect1d(np.where(normalized_df["number of neoplastic cells"]==0)[0], np.where(normalized_df["number of inflamatory cells"]==0)[0])

low_cell_patches = np.intersect1d(low_cell_patches, np.where(normalized_df["number of connective cells"]==0)[0])
low_cell_patches = np.intersect1d(low_cell_patches, np.where(normalized_df["number of necrosis cells"]==0)[0])

if remove_noneoplastic == 'False':
	low_cell_patches = np.intersect1d(low_cell_patches, np.where(normalized_df["number of no-neoplastic cells"]==0)[0])

print('low_cell_patches ',low_cell_patches.shape, ' cell_percentage ', cell_percentage.shape, 'back_percentage ', back_percentage.shape)
print(np.intersect1d(cell_percentage, back_percentage).shape)

remove_low_cell_patches = np.intersect1d(np.intersect1d(cell_percentage, back_percentage), low_cell_patches)

print(remove_low_cell_patches.shape)

filter_indices = np.setdiff1d(np.intersect1d(cell_percentage, back_percentage), remove_low_cell_patches)
print(filter_indices.shape)


list_filtered = list(features_csv.index)

for i,_ in enumerate(list_filtered):
	list_filtered[i] = _.split('/')[-2] + '/' + _.split('/')[-1].split('_')[0] + '_' + str(int(int(_.split('/')[-1].split('_')[1]))) + '_' + str(int(int(_.split('/')[-1].split('_')[2].split('.')[0])))
	
features_csv.index = list_filtered


with open(dataset_split_path, 'rb') as f:
	dataset_split_dict = pickle.load(f)

with open(dataset_split_path_test, 'rb') as f:
	dataset_split_dict_test = pickle.load(f)

	
dataset_split_dict_train_filtered = dict()
dataset_split_dict_test_filtered = dict()

all_dict = dict()

wsi = list_filtered[0].split('/')[0]
all_dict[wsi] = [0]
counter = 0
for i in list_filtered:
	
	if i.split('/')[0] == wsi:
		counter += 1
		continue
	
	else:
		all_dict[wsi].append(counter)

		wsi = i.split('/')[0]
		all_dict[wsi] = [counter]
		
	counter += 1
	
all_dict[wsi].append(counter)


for i in dataset_split_dict:
	try:
		if all_dict[i][1]-all_dict[i][0]<10:
			print(i, all_dict[i][1]-all_dict[i][0])
			continue
			
		dataset_split_dict_train_filtered[i] = all_dict[i]
		dataset_split_dict_train_filtered[i].append(dataset_split_dict[i][-1])
	except:
		continue
		

for i in dataset_split_dict_test:
	try:
		if all_dict[i][1]-all_dict[i][0]<10:
			print(i, all_dict[i][1]-all_dict[i][0])
			continue
			
		dataset_split_dict_test_filtered[i] = all_dict[i]
		dataset_split_dict_test_filtered[i].append(dataset_split_dict_test[i][-1])
		
	except:
		continue
		
	
bags_path = list(dataset_split_dict_train_filtered.keys())
np.random.seed(0)
np.random.shuffle(bags_path)

train_path = bags_path


train_index_select_list = []
for i in bags_path:
	train_index_select_list.extend(list(np.arange(dataset_split_dict_train_filtered[i][0], dataset_split_dict_train_filtered[i][1])))
print(features_csv.shape, features_csv.iloc[train_index_select_list].shape)



if norm_feat == 'max':
	train_max = features_csv.iloc[train_index_select_list].max(axis=0)
	# Normalize the DataFrame across the feature dimension
	normalized_df = features_csv.copy()
	# Normalize the non-missing values using mean and std
	normalized_df[~np.isnan(features_csv)] = (features_csv[~np.isnan(features_csv)]) / (train_max+1e-6)
	normalized_df = normalized_df.fillna(-1)

elif norm_feat == 'meanstd':
	train_mean = features_csv.iloc[train_index_select_list].mean(axis=0)
	train_std = features_csv.iloc[train_index_select_list].std(axis=0)
	# Normalize the DataFrame across the feature dimension
	normalized_df = features_csv.copy()
	# Normalize the non-missing values using mean and std
	normalized_df[~np.isnan(features_csv)] = (features_csv[~np.isnan(features_csv)] - train_mean) / (train_std+1e-6)
	normalized_df = normalized_df.fillna(-1)

elif norm_feat == 'bin':
	normalized_df = features_csv.copy()
	for i, col in tqdm(enumerate(features_csv.columns)):
		try:
			
			non_nan_indices = np.where(np.array(~np.isnan(features_csv[col])) == True)[0]
			
			_, intervals = pd.qcut(features_csv.iloc[train_index_select_list, i], bins, labels=False, retbins=True, duplicates='drop')

			normalized_df[col] = pd.cut(features_csv[col], bins=intervals, labels=False, include_lowest=True)

		except:
			print(i, col)
			normalized_df[col] = -1
		
	normalized_df = normalized_df.fillna(-1)

elif norm_feat == 'none':
	normalized_df = features_csv.copy()
	normalized_df = normalized_df.fillna(-1)
	
	
features_csv = features_csv.iloc[filter_indices]

print('After removing intervals', features_csv.shape)

normalized_df = normalized_df.iloc[filter_indices]

print('After removing intervals', normalized_df.shape)

	
	
	

##################################################################################
hovernet_mag = 40
patch_extract_mag = 5
patch_mag_ratio = hovernet_mag/patch_extract_mag
height = 224*patch_mag_ratio
width = 224*patch_mag_ratio
		
if remove_noneoplastic == 'True':
	cell_parse = 5
	labeldict = {1: 'neoplastic', 2: 'inflamatory', 3: 'connective', 4: 'necrosis'}
else:
	cell_parse = 6
	labeldict = {1: 'neoplastic', 2: 'inflamatory', 3: 'connective', 4: 'necrosis', 5 : 'no-neoplastic'}

cell_stat_column_name = []

for i in range(1, cell_parse):
		 # number of each type of cells
		cell_stat_column_name.append('number of ' + labeldict[i] + ' cells')
		
for i in range(1, cell_parse):
	for cell_feat in ["area", "orientation", "eccentricity", "solidity", "intensity_mean", "intensity_std", "contrast", "dissimilarity", "homogeneity", "energy"]:    
		for stats in ['mean', 'std', 'skew', 'kurtosis']:
						cell_stat_column_name.append(labeldict[i] + " cells: " + stats + ' of their '+  cell_feat)
print(len(cell_stat_column_name))
sna_column_name = []
for cell_feat in ["degree", "clustering_coefficient", "closeness_centrality", "degree_centrality"]:  
		for stats in ['max', 'mean', 'std', 'skew', 'kurtosis']:
				sna_column_name.append(stats + " of cells' "+  cell_feat)
print(len(sna_column_name))

		
athena_column_name = []

radii = np.linspace(0, min(height, width) / 2, 5)

for rad in radii[1:]:        
	athena_column_name.append('neoplastic' + " cells: k-function at radius " + str(int(rad)) + ' pixels')

						
athena_column_name.append('Graph modularity with cell types as community')

athena_column_name.append('Gloabl Richness (number of cell-types present)')
athena_column_name.append('Gloabl Shannon index')
athena_column_name.append('Gloabl Simpson index')
athena_column_name.append('Gloabl Renyi entropy (at alpha=infinity)')


athena_column_name.append('Local richness skew')
athena_column_name.append('Local Shannon index skew')
athena_column_name.append('Local Simpson index skew')
athena_column_name.append('Local Renyi entropy (at alpha=infinity) skew')


for i in list(labeldict.values())[1:]:
		athena_column_name.append('Infiltration of '+ i+ ' cells in neoplastic region')
		
for i in list(labeldict.values())[1:]:
		athena_column_name.append('Infiltration of neoplastic cells in '+ i+ ' region')
		

print(len(athena_column_name))

	
final_select_concepts = cell_stat_column_name
final_select_concepts.extend(sna_column_name)
final_select_concepts.extend(athena_column_name)

##################################################################################	
print(normalized_df.shape)
print(list(normalized_df.columns)[:10])
normalized_df = normalized_df[final_select_concepts]
print(normalized_df.shape)
print(list(normalized_df.columns)[:10])


normalized_df.to_csv(os.path.join(save_path, 'binned_hcf.csv'))

features_csv = features_csv.fillna(-1)


print(features_csv.shape)
print(list(features_csv.columns)[:10])
features_csv = features_csv[final_select_concepts]
print(features_csv.shape)
print(list(features_csv.columns)[:10])


features_csv.to_csv(os.path.join(save_path, 'raw_hcf.csv'))


list_filtered = list(features_csv.index)


with open(dataset_split_path, 'rb') as f:
	dataset_split_dict = pickle.load(f)

with open(dataset_split_path_test, 'rb') as f:
	dataset_split_dict_test = pickle.load(f)

	
dataset_split_dict_train_filtered = dict()
dataset_split_dict_test_filtered = dict()

all_dict = dict()

wsi = list_filtered[0].split('/')[0]
all_dict[wsi] = [0]
counter = 0
for i in list_filtered:
	
	if i.split('/')[0] == wsi:
		counter += 1
		continue
	
	else:
		all_dict[wsi].append(counter)

		wsi = i.split('/')[0]
		all_dict[wsi] = [counter]
		
	counter += 1
	
all_dict[wsi].append(counter)


for i in dataset_split_dict:
	try:
		if all_dict[i][1]-all_dict[i][0]<10:
			print(i, all_dict[i][1]-all_dict[i][0])
			continue
			
		dataset_split_dict_train_filtered[i] = all_dict[i]
		dataset_split_dict_train_filtered[i].append(dataset_split_dict[i][-1])
	except:
		continue
		

for i in dataset_split_dict_test:
	try:
		if all_dict[i][1]-all_dict[i][0]<10:
			print(i, all_dict[i][1]-all_dict[i][0])
			continue
			
		dataset_split_dict_test_filtered[i] = all_dict[i]
		dataset_split_dict_test_filtered[i].append(dataset_split_dict_test[i][-1])
		
	except:
		continue


with open(os.path.join(save_path, 'train_dict.pickle'), 'wb') as f:
	pickle.dump(dataset_split_dict_train_filtered, f)
	

with open(os.path.join(save_path, 'test_dict.pickle'), 'wb') as f:
	pickle.dump(dataset_split_dict_test_filtered, f)