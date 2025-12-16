
import torch
import torch.nn.functional as F

# change it with respect to the original model
from config import LlamaConfig
from llama import load_pretrained
from tokenizer import Tokenizer

class LlamaZeroShotClassifier(torch.nn.Module):
	def __init__(self, config: LlamaConfig, tokenizer: Tokenizer, label_names: list[str]):
		super(LlamaZeroShotClassifier, self).__init__()
		self.num_labels = config.num_labels
		self.llama = load_pretrained(config.pretrained_model_path)
		# Zero-shot classification does not require updating llama paramters.
		for param in self.llama.parameters():
			param.requires_grad = False
		assert len(label_names) == self.num_labels
		self.tokenizer = tokenizer
		self.label_name_ids = [tokenizer.encode(label, bos=False, eos=False) for label in label_names]


	def forward(self, input_ids):
		# compute the completion probability of each label string
		logits, _ = self.llama(input_ids)
		log_probabilities = F.log_softmax(logits, dim=-1)
		label_probabilities = torch.zeros((log_probabilities.shape[0], self.num_labels), device=log_probabilities.device)
		for i, label_token_ids in enumerate(self.label_name_ids):
			total_log_prob = torch.sum(log_probabilities[:, :, label_token_ids], axis=-1)
			label_probabilities[:, i] = total_log_prob[:, 0]
		return label_probabilities

class LlamaEmbeddingClassifier(torch.nn.Module):
	def __init__(self, config):
		super(LlamaEmbeddingClassifier, self).__init__()
		self.num_labels = config.num_labels
		self.llama = load_pretrained(config.pretrained_model_path)

        self.pooling_method = getattr(config, 'pooling_method', 'last_token')  # 'last_token', 'cls_token', 'mean_pooling', 'max_pooling', 'attention_pooling'
		# If we use pretrain mode, we freeze Llama parameters.
		for param in self.llama.parameters():
			if config.option == 'pretrain':
				param.requires_grad = False
			elif config.option == 'finetune':
				param.requires_grad = True

        # attention poolingのための重みパラメータ
        if self.pooling_method == 'attention_pooling':
            self.attention_weights = torch.nn.Linear(self.llama.config.dim, 1)

		self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
		self.classifier_head = torch.nn.Linear(self.llama.config.dim, self.num_labels)

	def create_attention_weights(self, input_ids, pad_token_id):
		return (input_ids != pad_token_id).float()

	def forward(self, input_ids):
		'''
		1) Find the hidden state after the final token of the input sequence
		2) Apply dropout (self.dropout) to the hidden state at training time to mitigate
		   overfitting.
		2) Pass this through the classifier head (self.classifier_head), which will return
		   logits (unnormalized probabilities) over all classes.
		3) Take the log-softmax of the logits and return log-probabilities over all classes.
		'''
		# todo
		# 1) LLaMAモデルで隠れ状態を取得
		logits, hidden_states = self.llama(input_ids)
		
		# 最終トークンの隠れ状態を取得
		# hidden_states shape: (batch_size, seq_len, hidden_dim)
		if self.pooling_method == 'last_token':
			pooled_output = hidden_states[:, -1, :]  # (batch_size, hidden_dim)
		elif self.pooling_method == 'cls_token':
			pooled_output = hidden_states[:, 0, :]  # (batch_size, hidden_dim)
		elif self.pooling_method == 'mean_pooling':
			attention_mask = self.create_attention_mask(input_ids, self.tokenizer.pad_id)
			attention_mask = attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
			
			# マスクされた平均プーリング
			masked_hidden_states = hidden_states * attention_mask
			sum_hidden_states = torch.sum(masked_hidden_states, dim=1)
			sum_mask = torch.sum(attention_mask, dim=1)
			pooled_output = sum_hidden_states / sum_mask
		elif self.pooling_method == 'max_pooling':
			# パディングマスクを作成
			attention_mask = self.create_attention_mask(input_ids, self.tokenizer.pad_id)
			attention_mask = attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
			
			# パディング部分を非常に小さな値に設定
			masked_hidden_states = hidden_states.masked_fill(attention_mask == 0, float('-inf'))
			pooled_output, _ = torch.max(masked_hidden_states, dim=1)
		elif self.pooling_method == 'attention_pooling':
			pooled_output = self.attention_weights(hidden_states)  
			attention_weights = torch.nn.functional.softmax(pooled_output, dim=-1)
			pooled_output = torch.sum(hidden_states * attention_weights, dim=1)  # (batch_size, hidden_dim)
		else:
			raise ValueError(f"Invalid pooling method: {self.pooling_method}")
		
		# 2) ドロップアウトを適用（訓練時のみ）
		pooled_output = self.dropout(pooled_output)
		
		# 3) 分類ヘッドを通してロジットを計算
		logits = self.classifier_head(pooled_output)  # (batch_size, num_labels)
		
		# 4) log-softmaxを適用してlog確率を返す
		log_probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
		
		return log_probabilities