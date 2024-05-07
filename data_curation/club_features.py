import os
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--feat_path', type=str, default = 'test_dataset/Handcrafted_features')
parser.add_argument('--column_name_path', type=str, default = 'test_dataset')
parser.add_argument('--list_dict_path', type=str, default = 'test_dataset/patches')
parser.add_argument('--remove_cell_type', type=str, default = 'none')  # 'none', 'no-neoplastic'. Use remove no-neoplastic when using dataset that doesn't have no-neoplastic. Like lung dataset
parser.add_argument('--use_kfunction', type=str, default = 'False')

args = parser.parse_args()



use_kfunction = args.use_kfunction
remove_cell_type = args.remove_cell_type



feat_path = args.feat_path
column_name_path = args.column_name_path
list_dict_path = args.list_dict_path

with open(os.path.join(list_dict_path, 'all_dict.pickle'), 'rb') as f:
	all_dict = pickle.load(f)


with open(os.path.join(column_name_path,'cell_stat_column_name.pickle'), 'rb') as f:
	cell_stat_column_name = pickle.load(f)
	
with open(os.path.join(column_name_path,'sna_column_name.pickle'), 'rb') as f:
	sna_column_name = pickle.load(f)
	
with open(os.path.join(column_name_path,'athena_column_name.pickle'), 'rb') as f:
	athena_column_name = pickle.load(f)

	
athena_dir = feat_path + '/athena_statistics'
cell_dir = feat_path + '/cell_statistics'
sna_dir = feat_path + '/sna_statistics'
tissue_dir = feat_path + '/tissue_statistics'


all_dict_key_list = []

for i, _ in enumerate(list(all_dict.keys())):
	exist = False
	for j in os.listdir(cell_dir):
		if _ + '.' in j:
			exist = True
			break
	if exist:
		all_dict_key_list.append(j)

print(len(all_dict_key_list), len(all_dict))

select_dir = cell_dir

feat_list = []

for i in tqdm(all_dict_key_list):
	with open(select_dir + '/' + i, 'rb') as f:
		temp_feat = pickle.load(f)
		
	feat_list.append(temp_feat)

merged_dict = {}

for d in tqdm(feat_list):
	merged_dict.update(d)
	
df_cell = pd.DataFrame.from_dict(merged_dict, orient='index', columns=cell_stat_column_name)

	

select_dir = sna_dir

feat_list = []

for i in tqdm(all_dict_key_list):
	with open(select_dir + '/' + i, 'rb') as f:
		temp_feat = pickle.load(f)
		
	feat_list.append(temp_feat)
		
merged_dict = {}

for d in tqdm(feat_list):
	merged_dict.update(d)
		
df_sna = pd.DataFrame.from_dict(merged_dict, orient='index', columns=sna_column_name)

	

select_dir = athena_dir
feat_list = []

for i in tqdm(all_dict_key_list):
	with open(select_dir + '/' + i, 'rb') as f:
		temp_feat = pickle.load(f)
		
	feat_list.append(temp_feat)
		
merged_dict = {}

for d in tqdm(feat_list):
	merged_dict.update(d)
	  
df_athena = pd.DataFrame.from_dict(merged_dict, orient='index', columns=athena_column_name)



select_dir = tissue_dir
feat_list = []

for i in tqdm(all_dict_key_list):
	with open(select_dir + '/' + i, 'rb') as f:
		temp_feat = pickle.load(f)
		
	feat_list.append(temp_feat)
		
merged_dict = {}

for d in tqdm(feat_list):
	merged_dict.update(d)
	  
df_tissue = pd.DataFrame.from_dict(merged_dict, orient='index', columns=['percent_of_cell_region', 'percent_of_tissue_region', 'percent_of_background_region'])

features_csv = pd.concat((df_cell, df_sna, df_athena, df_tissue), axis=1)

####################################################################

if use_kfunction == 'False':
	include_list = []
	for i, _ in enumerate(list(features_csv.columns)):
		if 'inflamatory cells: k-function' in _:
			continue

		if 'connective cells: k-function' in _:
			continue

		if 'necrosis cells: k-function' in _:
			continue

		if 'no-neoplastic cells: k-function' in _:
			continue

		include_list.append(i)

	features_csv = features_csv.iloc[:, np.array(include_list)]

	print('After removing k-function of other cells', features_csv.shape)

print('remove_cell_type:', remove_cell_type)

include_list = []
for i, _ in enumerate(list(features_csv.columns)):
	if not remove_cell_type in _:
		include_list.append(i)

features_csv = features_csv.iloc[:, np.array(include_list)]
feats_size = len(include_list)

print('After removing cell type ' + remove_cell_type + ' columns', features_csv.shape)
print('Feat size after:', feats_size)



shannon_group = ['Local Shannon index histogram-bin: 0', 'Local Shannon index histogram-bin: 1',  'Local Shannon index histogram-bin: 2', 'Local Shannon index histogram-bin: 3', 'Local Shannon index histogram-bin: 4']

simpson_group = ['Local Simpson index histogram-bin: 0', 'Local Simpson index histogram-bin: 1',  'Local Simpson index histogram-bin: 2', 'Local Simpson index histogram-bin: 3', 'Local Simpson index histogram-bin: 4']

renqi_group = ['Local Renyi entropy (at alpha=infinity) histogram-bin: 0', 'Local Renyi entropy (at alpha=infinity) histogram-bin: 1',  'Local Renyi entropy (at alpha=infinity) histogram-bin: 2', 'Local Renyi entropy (at alpha=infinity) histogram-bin: 3', 'Local Renyi entropy (at alpha=infinity) histogram-bin: 4']


richness_group = ['Local Richness, number of times 1 cell-types present', 'Local Richness, number of times 2 cell-types present',	'Local Richness, number of times 3 cell-types present',	'Local Richness, number of times 4 cell-types present',	'Local Richness, number of times 5 cell-types present']


features_csv['Local Shannon index skew'] = features_csv[shannon_group].skew(1)
features_csv['Local Simpson index skew'] = features_csv[simpson_group].skew(1)
features_csv['Local Renyi entropy (at alpha=infinity) skew'] = features_csv[renqi_group].skew(1)
features_csv['Local richness skew'] = features_csv[richness_group].skew(1)


features_csv = features_csv.drop(shannon_group, axis=1)
features_csv = features_csv.drop(simpson_group, axis=1)
features_csv = features_csv.drop(renqi_group, axis=1)
features_csv = features_csv.drop(richness_group, axis=1)

feats_size = features_csv.shape[1]


print('After removing local histogram bins', features_csv.shape)
print('Feat size after:', feats_size)



features_csv.to_csv(feat_path + '/cell_athena_sna.csv')
