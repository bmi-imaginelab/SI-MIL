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
from tqdm import tqdm
import pickle
import json
torch.manual_seed(0)
import torch.nn.functional as F
from sklearn.model_selection import KFold
import wandb

def train(train_list, dataset_split_dict, dataset_split_deep_dict, features_array, features_deep_array, milnet, criterion, optimizer, feats_size, feats_size_deep, args, epoch):
	milnet.train()
	total_loss = 0
	total_loss_mse = 0
	total_loss_attnreg=0

	bc = 0
	Tensor = torch.cuda.FloatTensor
	for i in range(len(train_list)):
		optimizer.zero_grad()

		split_info = dataset_split_dict[train_list[i]]
		if split_info[1]-split_info[0] == 0:
			print(train_list[i])

		label = split_info[-1]

		feats = features_array[split_info[0]:split_info[1]]
		# feats = np.array(feats)
		
		split_info_deep = dataset_split_deep_dict[train_list[i]]
		feats_deep = features_deep_array[split_info_deep[0]:split_info_deep[1]]
		feats_deep = np.array(feats_deep)
		
		feats, feats_deep = dropout_patches(feats, feats_deep, args.dropout_patch)
		
		bag_feats_deep = Variable(Tensor([feats_deep]))
		bag_feats_deep = bag_feats_deep.view(-1, feats_size_deep)

		assert feats_deep.shape[0] == feats.shape[0]
		
		bag_label = Variable(Tensor([label]))
		
		bag_feats = Variable(Tensor([feats]))
		bag_feats = bag_feats.view(-1, feats_size)
					
		

		bag_prediction, bag_prediction_deep, A_patch, bag_features, bag_features_deep, A_feat = milnet(bag_feats, bag_feats_deep, training='yes')
			

		loss_class = criterion(bag_prediction, bag_label) + criterion(bag_prediction_deep, bag_label)

		loss = loss_class
		
		if 'Lmse' in args.feat_type:
			loss_mse = F.mse_loss(bag_prediction, bag_prediction_deep.clone().detach())
			total_loss_mse = total_loss_mse + 20*loss_mse.item()
			
			loss += 20*loss_mse
		

		
		loss_attnreg = torch.norm(A_patch, p=2)
		total_loss_attnreg = total_loss_attnreg + 0.05*loss_attnreg.item()
		loss += 0.05*loss_attnreg
		
		loss.backward()
		optimizer.step()
		total_loss = total_loss + loss_class.item()
		
		sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_list), loss.item()))
	return total_loss / len(train_list), total_loss_mse / len(train_list), total_loss_attnreg / len(train_list)

def dropout_patches(feats, feats_deep, p):
	idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0]*(1-p)), replace=False)
	sampled_feats = np.take(feats, idx, axis=0)
	sampled_feats_deep = np.take(feats_deep, idx, axis=0)
	pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0]*p), replace=False)
	pad_feats = np.take(sampled_feats, pad_idx, axis=0)
	pad_feats_deep = np.take(sampled_feats_deep, pad_idx, axis=0)
	sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
	sampled_feats_deep = np.concatenate((sampled_feats_deep, pad_feats_deep), axis=0)
	return sampled_feats, sampled_feats_deep


def test(test_list, dataset_split_dict, dataset_split_deep_dict, features_array, features_deep_array, milnet, criterion, feats_size, feats_size_deep, args):
	milnet.eval()
	total_loss = 0
	test_labels = []
	test_predictions = []
	test_bag_features = dict()
	
	Tensor = torch.cuda.FloatTensor
	with torch.no_grad():
		for i in range(len(test_list)):
			
			
			split_info = dataset_split_dict[test_list[i]]
			# label = [split_info[-1]]*args.num_classes

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
			# bag_prediction, _, bag_features = milnet(bag_feats, bag_feats_deep)
			
			bag_prediction, bag_prediction_deep, A_patch, bag_features, bag_features_deep, A_feat = milnet(bag_feats, bag_feats_deep)
						
			# print(bag_features.shape)
			
			loss = criterion(bag_prediction, bag_label)
			total_loss = total_loss + loss.item()
			sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_list), loss.item()))
			test_labels.append(label)
			test_predictions.append(bag_prediction.squeeze().cpu().numpy())
			
			test_bag_features[i] = bag_features.squeeze(0).squeeze(0)
			
	test_labels = np.array(test_labels)
	test_predictions = np.array(test_predictions)
	
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
	
	return total_loss / len(test_list), avg_score, auc_value, bag_score_classwise

	
def val(test_list, dataset_split_dict, dataset_split_deep_dict, features_array, features_deep_array, milnet, criterion, feats_size, feats_size_deep, args):
	milnet.eval()
	total_loss = 0
	test_labels = []
	test_predictions = []
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
			# bag_prediction, _, _ = milnet(bag_feats, bag_feats_deep)
			
			bag_prediction, bag_prediction_deep, A_patch, bag_features, bag_features_deep, A_feat = milnet(bag_feats, bag_feats_deep)
			
			loss = criterion(bag_prediction, bag_label)
			total_loss = total_loss + loss.item()
			sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_list), loss.item()))
			test_labels.append(label)
			test_predictions.append(bag_prediction.squeeze().cpu().numpy())
			
	test_labels = np.array(test_labels)
	test_predictions = np.array(test_predictions)
	
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
	
	return total_loss / len(test_list), avg_score, auc_value, bag_score_classwise
	



def main():
	parser = argparse.ArgumentParser(description='Train ABMIL on 5x patch features learned by encoder')
	parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
	parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
	parser.add_argument('--num_epochs', default=40, type=int, help='Number of total training epochs [40|200]')
	parser.add_argument('--gpu_index', type=int, default=1, help='GPU ID(s) [0]')
	parser.add_argument('--weight_decay', default=5e-2, type=float, help='Weight decay [5e-3]')
	
	# path for PathExpert features
	parser.add_argument('--dataset_split_path', default='', type=str, help='Dataset folder name')
	parser.add_argument('--dataset_split_path_test', default='', type=str, help='Dataset folder name')
	parser.add_argument('--features_path', default='', type=str, help='Dataset folder name')

	# path for Deep features
	parser.add_argument('--dataset_split_deep_path', default='', type=str, help='Dataset folder name')
	parser.add_argument('--dataset_split_deep_path_test', default='', type=str, help='Dataset folder name')
	parser.add_argument('--features_deep_path', default='', type=str, help='Dataset folder name')
	parser.add_argument('--features_deep_path_test', default='', type=str, help='Dataset folder name')

	parser.add_argument('--save_path', default='', type=str, help='Dataset folder name')
	
	parser.add_argument('--bins', default=10, type=int, help='Number of bins for binarization')
	parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
	parser.add_argument('--model', default='abmil', type=str, help='MIL model [abmil]')
	parser.add_argument('--dropout_patch', default=0.0, type=float, help='Patch dropout rate [0]')
	parser.add_argument('--dropout_node', default=0.0, type=float, help='Bag classifier dropout rate [0]')
	parser.add_argument('--organ', default='lung', type=str, help='for grouping in wandb')
	parser.add_argument('--torch_seed', default=-1, type=int, help='')
	parser.add_argument('--feat_type', default='', type=str, help='for grouping in wandb')

	parser.add_argument('--use_additive', default='yes', type=str, help=' no, yes. for grouping in wandb')
	parser.add_argument('--stop_gradient', default='no', type=str, help=' no, yes. for grouping in wandb')
	parser.add_argument('--no_projection', default='yes', type=str, help=' no, yes. for grouping in wandb')
	parser.add_argument('--cross_val_fold', default=0, type=int, help='')
	parser.add_argument('--temperature', default=3.0, type=float, help='')
	parser.add_argument('--percentile', default=0.75, type=float, help='')
	parser.add_argument('--top_k', default=20, type=int, help='')

	args = parser.parse_args()
	if args.torch_seed >= 0:
		torch.manual_seed(args.torch_seed)
			
	save_path = args.save_path
	if not os.path.isdir(save_path):
		try:
			os.mkdir(save_path)
		except:
			print('just made by different multiprocessing file')
			
	feat_type = args.save_path.split('/')[-3]  + '_' + args.feat_type
		
	device = torch.device(f'cuda:{args.gpu_index}')
	
	with open(args.dataset_split_path, 'rb') as f:
		dataset_split_dict = pickle.load(f)
	
	with open(args.dataset_split_path_test, 'rb') as f:
		dataset_split_dict_test = pickle.load(f)
	
	with open(args.dataset_split_deep_path, 'rb') as f:
		dataset_split_deep_dict = pickle.load(f)
	
	with open(args.dataset_split_deep_path_test, 'rb') as f:
		dataset_split_deep_dict_test = pickle.load(f)

	
	features_deep_array = torch.load(args.features_deep_path)
	features_deep_array_test = torch.load(args.features_deep_path_test)
	
	print(features_deep_array.shape, features_deep_array_test.shape)
	
	features_array = pd.read_csv(args.features_path)

	features_array = features_array.set_index('Unnamed: 0')
	
	features_array = np.array(features_array)
		
	print(features_array.shape)
	
	feats_size = features_array.shape[1]
	feats_size_deep = features_deep_array.shape[1]

	import si_mil as mil	
	
	b_classifier = mil.BClassifier(input_size=feats_size, input_size_deep=feats_size_deep, output_class=args.num_classes, stop_gradient=args.stop_gradient, no_projection=args.no_projection, top_k=args.top_k, temperature=args.temperature, percentile=args.percentile, dropout_v=args.dropout_node).to(device)
	
	milnet = mil.MILNet(b_classifier).to(device)

	criterion = nn.BCELoss()
	
	optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
	
	save_path += '/' + args.organ + '_' + feat_type +'_top' + str(args.top_k)  + '_stp_gr_' + args.stop_gradient + '_cross_val_fold_' + str(args.cross_val_fold) + '_no_proj_' + args.no_projection + '_temp' + str(args.temperature) + '_perc' + str(args.percentile) + '_feat_size_' + str(feats_size) + '_ep' + str(args.num_epochs) + '_lr0p' + str(args.lr).split('.')[-1] + '_wd0p' + str(args.weight_decay).split('.')[-1] + '_dropout0p' + str(args.dropout_patch).split('.')[-1]
	
	if not os.path.isdir(save_path):
		os.mkdir(save_path)
		
	name='ABMIL: '+ args.organ + '_' + feat_type + '_top_k_' + str(args.top_k)  + '_additive_' + args.use_additive  + '_stop_gradient_' + args.stop_gradient + '_no_projection_' + args.no_projection + '_temperature_' + str(args.temperature) + '_percentile_' + str(args.percentile) + '_' + 'feat_size_' + str(feats_size) + '_ep' + str(args.num_epochs) + '_lr0p' + str(args.lr).split('.')[-1] + '_wd0p' + str(args.weight_decay).split('.')[-1] + '_classes' + str(args.num_classes) + '_dropout0p' + str(args.dropout_patch).split('.')[-1] + '_dropoutnode0p' + str(args.dropout_node).split('.')[-1] + '_bins' + str(args.bins)
	
	wandb.init(
		# set the wandb project where this run will be logged
		project="Interpretable_MIL",


		name='ABMIL: '+ args.organ + '_' + feat_type + '_top_k_' + str(args.top_k) + '_additive_' + args.use_additive + '_cross_val_fold_' + str(args.cross_val_fold)  + '_stop_gradient_' + args.stop_gradient + '_no_projection_' + args.no_projection + '_temperature_' + str(args.temperature) + '_percentile_' + str(args.percentile) + '_' + 'feat_size_' + str(feats_size) + '_ep' + str(args.num_epochs) + '_lr0p' + str(args.lr).split('.')[-1] + '_wd0p' + str(args.weight_decay).split('.')[-1] + '_classes' + str(args.num_classes) + '_dropout0p' + str(args.dropout_patch).split('.')[-1] + '_dropoutnode0p' + str(args.dropout_node).split('.')[-1] + '_bins' + str(args.bins),
		# track hyperparameters and run metadata
		config={
		"MIL architecture": 'ABMIL',
		"use_additive": args.use_additive,
		"no_projection": args.no_projection,
		"stop_gradient": args.stop_gradient,
		"feat_type": feat_type,
		"feats_size": feats_size,
		"learning_rate": args.lr,
		"epochs": args.num_epochs,
		"weight_decay": args.weight_decay,
		"number_of_classes": args.num_classes,
		"patch_dropout": args.dropout_patch,
		"node_dropout": args.dropout_node,
		"organ":args.organ,
		"bins":args.bins,
		"cross_val_fold":args.cross_val_fold,
		"run_name":name,
		"temperature":args.temperature,
		"percentile":args.percentile,
		"top_k": args.top_k,
		},
		
		settings=wandb.Settings(_service_wait=300)
	)

	with open(save_path + '/' + 'commandline_args.txt', 'w') as f:
		json.dump(args.__dict__, f, indent=2)   
		
	bags_path = list(dataset_split_dict.keys())
	np.random.seed(0)
	np.random.shuffle(bags_path)
	
	kf = KFold(n_splits=5)

	c = 0
	for train_index, val_index in kf.split(bags_path):
		if c == args.cross_val_fold:
			train_path = [bags_path[i] for i in train_index]
			val_path = [bags_path[i] for i in val_index]
		c += 1
	
	best_score = 0
	test_path = list(dataset_split_dict_test.keys())
	
	train_index_select_list = []
	for i in bags_path:  # from train and val both
		train_index_select_list.extend(list(np.arange(dataset_split_dict[i][0], dataset_split_dict[i][1])))
		
	print(features_array.shape, features_array[train_index_select_list].shape)


	for i in range(features_array.shape[1]):
		non_minusone_indices = np.where(features_array[:,i] != -1)[0] # -1 when feature is None i.e. can't be calculated
		non_minusone_indices_in_train = np.intersect1d(train_index_select_list, non_minusone_indices)

		features_array[non_minusone_indices, i] = (features_array[non_minusone_indices, i] - features_array[non_minusone_indices_in_train, i].mean()) / (features_array[non_minusone_indices_in_train, i].std() + 1e-6)
		
		# features_array[minusone_indices, i] = 0

	print(len(train_path), len(val_path), len(test_path))
	
	for epoch in range(1, args.num_epochs):
		np.random.shuffle(train_path)       
		train_loss_bag, train_loss_bag_mse, train_loss_bag_attnreg = train(train_path, dataset_split_dict, dataset_split_deep_dict, features_array, features_deep_array, milnet, criterion, optimizer, feats_size, feats_size_deep, args, epoch) # iterate all bags
		
		val_loss_bag, avg_score_val, aucs_val, bag_score_classwise_val = val(val_path, dataset_split_dict, dataset_split_deep_dict, features_array, features_deep_array, milnet, criterion, feats_size, feats_size_deep, args)
		
		test_loss_bag, avg_score, aucs, bag_score_classwise = test(test_path, dataset_split_dict_test, dataset_split_deep_dict_test, features_array, features_deep_array_test, milnet, criterion, feats_size, feats_size_deep, args)
		
		
		print('\r Epoch [%d/%d] train loss: %.4f val loss: %.4f, average score: %.4f, AUC: %.4f' % 
			  (epoch, args.num_epochs, train_loss_bag, val_loss_bag, avg_score_val, aucs_val)) 
		
		print(bag_score_classwise_val)
		print('\r test loss: %.4f, average score: %.4f, AUC: %.4f' % 
			  (test_loss_bag, avg_score, aucs)) 
		print(bag_score_classwise)
		
		
		if 'Lmse' in args.feat_type: 
			wandb.log({"train loss": train_loss_bag, "train loss attnreg": train_loss_bag_attnreg, "train loss mse": train_loss_bag_mse, "val loss": val_loss_bag, 
				   "test loss": test_loss_bag, "val score": avg_score_val,
				   "test score": avg_score}, step=epoch)
		else:
			wandb.log({"train loss": train_loss_bag, "train loss attnreg": train_loss_bag_attnreg, "val loss": val_loss_bag, 
				   "test loss": test_loss_bag, "val score": avg_score_val,
				   "test score": avg_score}, step=epoch)
		
		for class_index in range(args.num_classes):
			wandb.log({ "Validation AUC Class " + str(class_index):aucs_val, "Test AUC Class " + str(class_index):aucs, "Validation score Class " + str(class_index):bag_score_classwise_val[class_index], "Test score Class " + str(class_index):bag_score_classwise[class_index] }, step=epoch)
			
		scheduler.step()
		
		current_score = (aucs_val + avg_score_val)/2
		
		if current_score >= best_score:
			best_score = current_score
			save_name = os.path.join(save_path, 'checkpoint_best.pth')
			torch.save(milnet.state_dict(), save_name)
			print('Best model saved at: ' + save_name)

			
		save_name = os.path.join(save_path, 'checkpoint.pth')
		torch.save(milnet.state_dict(), save_name)
		

	wandb.run.summary["final_average_score"] = avg_score
	wandb.run.summary["final_average_auc"] = aucs
	
	wandb.finish()


if __name__ == '__main__':
	main()
	
