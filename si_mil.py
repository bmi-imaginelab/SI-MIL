import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from perturbedtopk import PerturbedTopK


class Attn_Net_Gated(nn.Module):
	def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
		super(Attn_Net_Gated, self).__init__()
		self.attention_a = [
			nn.Linear(L, D),
			nn.Tanh()]
		
		self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
		if dropout:
			self.attention_a.append(nn.Dropout(0.25))
			self.attention_b.append(nn.Dropout(0.25))

		self.attention_a = nn.Sequential(*self.attention_a)
		self.attention_b = nn.Sequential(*self.attention_b)
		self.attention_c = nn.Linear(D, n_classes)

	def forward(self, x):
		a = self.attention_a(x)
		b = self.attention_b(x)
		A = a.mul(b)
		A = self.attention_c(A)  # N x n_classes
		return A

	
	
class MLPMixerLayer(nn.Module):
	def __init__(self, num_tokens, dim, hidden_dim):
		super(MLPMixerLayer, self).__init__()
		
		# Token mixing (across the token/sequence dimension)
		self.token_mlp = nn.Sequential(
			nn.Linear(num_tokens, hidden_dim),
			nn.GELU(),
			nn.Linear(hidden_dim, num_tokens)
		)
		
		# Channel mixing (across the feature dimension)
		self.channel_mlp = nn.Sequential(
			nn.Linear(dim, hidden_dim),
			nn.GELU(),
			nn.Linear(hidden_dim, dim)
		)

	def forward(self, x):
		# Token mixing
		x = x + self.token_mlp(x.permute(0, 2, 1)).permute(0, 2, 1)
		# Channel mixing
		x = x + self.channel_mlp(x)
		return x

class MLPMixer(nn.Module):
	def __init__(self, num_tokens, dim, hidden_dim, num_layers=2):
		super(MLPMixer, self).__init__()
		
		self.layers = nn.ModuleList([MLPMixerLayer(num_tokens, dim, hidden_dim) for _ in range(num_layers)])
		
	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x
	

class BClassifier(nn.Module):
	def __init__(self, input_size, input_size_deep, output_class, stop_gradient='no', no_projection='yes', top_k=20, temperature = 3, percentile=0.75, dropout_v=0.0, mlp_layers=4): # K, L, N
		super(BClassifier, self).__init__()
		
		self.attention_deep = Attn_Net_Gated(L=input_size_deep)  
		
		self.projection_head_deep = nn.Sequential(
			nn.Linear(input_size_deep, input_size_deep),
			nn.ReLU(),
		)   

		self.output_class = output_class

		# head for Deep features
		self.classification_head_deep = nn.Sequential(
			nn.Linear(input_size_deep, 1)
		)
		
		# head for PathExpert features
		self.classification_head = nn.Sequential(
			nn.Linear(input_size, 1)
		)
		
		self.top_k = top_k
		self.k_sigma = 0.002
		
		self.aux_ga = Attn_Net_Gated(L=self.top_k, D=128)
		
		self.mlp_mixer = MLPMixer(num_tokens=input_size, dim=top_k, hidden_dim=128, num_layers=mlp_layers) # changed from 128, 2
		self.temperature = temperature
		
		self.percentile = percentile
		
		self.stop_gradient = stop_gradient
		
	def forward(self, feats, feats_deep, training='no'): # N x K, N x C
		device = feats_deep.device
		

		V_deep = self.projection_head_deep(feats_deep) # N x D

		A_deep = self.attention_deep(V_deep) # N x 1
		

		A_deep = F.softmax(A_deep, 0)  # N x 1
		

		B_deep = A_deep*V_deep   #  Final output (B) - NxD

		B_deep = self.classification_head_deep(B_deep) # N x 1

		_, topk_indices = torch.sort(A_deep.clone().detach(), 0, descending=True)
			
		if topk_indices.shape[0]<self.top_k:  # repeat necessary to make the model run as we need to fix features input
			repeat_factor = int(self.top_k//topk_indices.shape[0]) + 1
			topk_indices = topk_indices.repeat(repeat_factor, 1)
			
		topk_feats = ((feats[topk_indices[:self.top_k, :]]).clone()).permute(1, 2, 0)  # 1 x D x topk
		
		topk_feats = self.mlp_mixer(topk_feats)  # 1 x D x topk

		A_aux = self.aux_ga(topk_feats).squeeze(-1)   # input: 1 x D x topk, output: 1 x D

		if self.percentile != 0:
			percentile = torch.quantile(A_aux, self.percentile, dim=1, keepdim=True)   
			std_values = torch.std(A_aux, dim=1, keepdim=True)

			A_aux = (A_aux - percentile) / (std_values + 1e-6)
		
		A_aux = F.sigmoid(A_aux*self.temperature)          # 1 x D      
		
		if feats.shape[0] > self.top_k:
			if self.stop_gradient == 'no' and training=='yes':
				topk_selector = PerturbedTopK(k=self.top_k, num_samples=100, sigma=self.k_sigma)
				topk_indices = topk_selector(A_deep.transpose(0, 1)).squeeze() # feed 1xN to get output of size top_k X N
				perturbed_topk_feats = torch.einsum('kn,nd->kd', topk_indices, feats) # top_k x D
			else:	
				_, topk_indices = torch.sort(A_deep.clone().detach(), 0, descending=True)
				perturbed_topk_feats = ((feats[topk_indices[:self.top_k, :]]).clone()).squeeze()  # top_k x D
		else:
			perturbed_topk_feats = feats

		B = perturbed_topk_feats * A_aux 	# top_k/N x D   # / when self.top_k>feats.shape[0] 

		B = self.classification_head(B) # top_k/N x 1

		C = F.sigmoid(B.sum(0)) # 1
		
		C_deep = F.sigmoid(B_deep.sum(0)) # 1
		
		return C, C_deep, A_deep, B, B_deep, A_aux
		
		
		
class MILNet(nn.Module):
	def __init__(self, b_classifier):
		super(MILNet, self).__init__()
		self.b_classifier = b_classifier
		
	def forward(self, x, x_deep, training='no'):
		
		prediction_bag, prediction_bag_deep, A, B, B_deep, A_aux = self.b_classifier(x, x_deep, training=training)

		return prediction_bag, prediction_bag_deep, A, B, B_deep, A_aux
