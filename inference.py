import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
import torch.nn as nn
import wandb
from tqdm import tqdm
import pickle
import json
torch.manual_seed(0)
save_type = 'test'
save = True
import torch.nn.functional as F

def test(test_list, dataset_split_dict, dataset_split_deep_dict, features_array, features_deep_array, milnet, criterion, feats_size, feats_size_deep, args):
	milnet.eval()
	total_loss = 0
	test_labels = []
	test_predictions = []
	test_bag_features = dict()
	attention_test_bag_patch = dict()
	attention_test_bag_feature = dict()

	Tensor = torch.cuda.FloatTensor
	with torch.no_grad():
		for i in range(len(test_list)):

			split_info = dataset_split_dict[test_list[i]]

			label = split_info[-1]

			feats = features_array[split_info[0] : split_info[1]]
			
			split_info_deep = dataset_split_deep_dict[test_list[i]]
			feats_deep = features_deep_array[split_info_deep[0]:split_info_deep[1]]
			feats_deep = np.array(feats_deep)
			bag_feats_deep = Variable(Tensor([feats_deep]))
			bag_feats_deep = bag_feats_deep.view(-1, feats_size_deep)
		
			assert feats_deep.shape[0] == feats.shape[0]
			
			bag_label = Variable(Tensor([label]))
			bag_feats = Variable(Tensor([feats]))
			bag_feats = bag_feats.view(-1, feats_size)
			bag_prediction, bag_prediction_deep, A_patch, bag_features, bag_features_deep, A_feat = milnet(bag_feats, bag_feats_deep)
			
			loss = criterion(bag_prediction, bag_label)
			total_loss = total_loss + loss.item()
			sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_list), loss.item()))
			test_labels.append(label)
			test_predictions.append(bag_prediction.squeeze().cpu().numpy())
			
			
			test_bag_features[i] = bag_features.squeeze(0).squeeze(0)

			attention_test_bag_patch[i] = A.clone()
			attention_test_bag_feature[i] = A_feat.clone()

	test_labels = np.array(test_labels)
	test_predictions = np.array(test_predictions)
	
	if save is True:
		with open(args.model_weights_path[:-14] + save_type + '_predictions.pickle', 'wb') as f:
			pickle.dump(test_predictions, f)

		with open(args.model_weights_path[:-14] + save_type + '_labels.pickle', 'wb') as f:
			pickle.dump(test_labels, f)
	
		with open(args.model_weights_path[:-14] + save_type + '_bag_features.pickle', 'wb') as f:
			# print('saving')
			pickle.dump(test_bag_features, f)

		with open(args.model_weights_path[:-14] + save_type + '_bag_attention_feature.pickle', 'wb') as f:
			pickle.dump(attention_test_bag_feature, f)

		with open(args.model_weights_path[:-14] + save_type + '_bag_attention_patch.pickle', 'wb') as f:
			pickle.dump(attention_test_bag_patch, f)

	auc_value = roc_auc_score(test_labels, test_predictions)
	
	test_predictions = (test_predictions>0.5)*1
	bag_score = 0
	
	bag_score_classwise = np.zeros(args.num_classes)
	bag_number_classwise = np.zeros(args.num_classes)
	
	for i in range(0, len(test_list)):
		bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score       
		
		bag_score_classwise[test_labels[i]] += np.array_equal(test_labels[i], test_predictions[i])
		bag_number_classwise[test_labels[i]] += 1
		
	avg_score = bag_score / len(test_list)
	bag_score_classwise = bag_score_classwise/bag_number_classwise
	
	if save is True:
		with open(args.model_weights_path[:-14] + save_type + '_class_prediction_bag.pickle', 'wb') as f:
			pickle.dump(test_predictions, f)
	
	return total_loss / len(test_list), avg_score, auc_value, bag_score_classwise

def main():
	parser = argparse.ArgumentParser(description='Train ABMIL on 5x patch features learned by dino vit-small')
	parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
	parser.add_argument('--gpu_index', type=int, nargs='+', default=(1,), help='GPU ID(s) [0]')
	
	
	# path for PathExpert features
	parser.add_argument('--dataset_split_path', default='', type=str, help='Dataset folder name')
	parser.add_argument('--dataset_split_path_test', default='', type=str, help='Dataset folder name')
	parser.add_argument('--features_path', default='', type=str, help='Dataset folder name')

	# path for Deep features
	parser.add_argument('--dataset_split_deep_path_test', default='', type=str, help='Dataset folder name')
	parser.add_argument('--features_deep_path_test', default='', type=str, help='Dataset folder name')

	parser.add_argument('--model_weights_path', default='MIL_experiment/', type=str, help='Dataset folder name')
	
	parser.add_argument('--use_additive', default='yes', type=str, help=' no, yes. for grouping in wandb')
	parser.add_argument('--stop_gradient', default='no', type=str, help=' no, yes. for grouping in wandb')
	parser.add_argument('--no_projection', default='yes', type=str, help=' no, yes. for grouping in wandb')
	parser.add_argument('--top_k', default=20, type=int, help='')
	parser.add_argument('--temperature', default=3.0, type=float, help='')
	parser.add_argument('--percentile', default=0.75, type=float, help='')

	args = parser.parse_args()
	
	
	gpu_ids = tuple(args.gpu_index)
	os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
	
	
	with open(args.dataset_split_path, 'rb') as f:
		dataset_split_dict = pickle.load(f)
	
	
	with open(args.dataset_split_path_test, 'rb') as f:
		dataset_split_dict_test = pickle.load(f)
	

	with open(args.dataset_split_deep_path_test, 'rb') as f:
		dataset_split_deep_dict_test = pickle.load(f)

	
	features_deep_array_test = torch.load(args.features_deep_path_test)
	
	
	features_array = pd.read_csv(args.features_path)

	features_array = features_array.set_index('Unnamed: 0')
	
	features_array = np.array(features_array)
	
	feats_size = features_array.shape[1]
	feats_size_deep = features_deep_array_test.shape[1]
		
	criterion = nn.BCELoss()
	
	
	
	import si_mil as mil
	
	b_classifier = mil.BClassifier(input_size=feats_size, input_size_deep=feats_size_deep, output_class=args.num_classes, stop_gradient=args.stop_gradient, no_projection=args.no_projection, top_k=args.top_k, temperature=args.temperature, percentile=args.percentile).cuda()
	
	milnet = mil.MILNet(b_classifier).cuda()

	state_dict_weights = torch.load(args.model_weights_path)
	milnet.load_state_dict(state_dict_weights, strict=True)
	
	bags_path = list(dataset_split_dict.keys())

	test_path = list(dataset_split_dict_test.keys())
	
	train_index_select_list = []
	for i in bags_path:  # from train and val both
		train_index_select_list.extend(list(np.arange(dataset_split_dict[i][0], dataset_split_dict[i][1])))


	for i in range(features_array.shape[1]):
		non_minusone_indices = np.where(features_array[:,i] != -1)[0]
		non_minusone_indices_in_train = np.intersect1d(train_index_select_list, non_minusone_indices)

		features_array[non_minusone_indices, i] = (features_array[non_minusone_indices, i] - features_array[non_minusone_indices_in_train, i].mean()) / (features_array[non_minusone_indices_in_train, i].std() + 1e-6)
	
	# print(len(test_path))
	
	test_loss_bag, avg_score, aucs, bag_score_classwise = test(test_path, dataset_split_dict_test, dataset_split_deep_dict_test, features_array, features_deep_array_test, milnet, criterion, feats_size, feats_size_deep, args)
	
	print('\r test loss: %.4f, average score: %.4f, AUC: %.4f' % 
				(test_loss_bag, avg_score, aucs)) 
	print(bag_score_classwise)
	

if __name__ == '__main__':
	main()
	
