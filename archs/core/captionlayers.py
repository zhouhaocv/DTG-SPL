import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.config import config

def batch_semantic_sim_score(candidate_corpus, references_corpus,word_vectors):
    """
    revised version for sentence-level metrics

    """
    batch_size,anchor_num,_ = candidate_corpus.size()
    cls_num = config.DTG_SPL.SEMANTIC.PARAMS.vocab_size
    w2v_dim = word_vectors.size(-1)
    word_vectors = word_vectors.to(candidate_corpus.get_device())

    candidate_num = (candidate_corpus>2).sum(-1)
    references_num = (references_corpus>2).sum(-1)

    candidate_cls = word_vectors[candidate_corpus].sum(2)/candidate_num[:,:,None]
    references_cls = word_vectors[references_corpus].sum(2)/references_num[:,:,None]

    semantic_sim_scores = F.cosine_similarity(candidate_cls,references_cls, dim=-1)

    return semantic_sim_scores

class DecoderRNN(nn.Module):
	"""
	Provides functionality for decoding in a seq2seq framework, with an option for attention.
	"""
	def __init__(self, cfg):
		super(DecoderRNN, self).__init__()

		if cfg.cell_type.lower() == 'lstm':
			self.rnn_cell = nn.LSTM
		elif cfg.cell_type.lower() == 'gru':
			self.rnn_cell = nn.GRU
		else:
			raise ValueError("Unsupported RNN Cell: {0}".format(cfg.cell_type))

		self.bidirectional_encoder = cfg.bidirectional
		self.hidden_size = cfg.hidden_dim
		self.output_size = cfg.vocab_size
		self.use_attention = cfg.use_attention
		self.eos_id = cfg.eos_id
		self.sos_id = cfg.sos_id
		self.num_layers = cfg.num_layers
		self.input_dropout = nn.Dropout(p=cfg.input_dropout_p)
		if cfg.trans_type == "word":
			self.max_length = cfg.max_word + 1
		else:
			self.max_length = cfg.max_query + 1

		self.rnn = self.rnn_cell(self.hidden_size, self.hidden_size//2 if self.bidirectional_encoder else self.hidden_size, 
			num_layers=self.num_layers,bidirectional=self.bidirectional_encoder, batch_first=True, dropout=cfg.dropout_p)
		self.embedding = nn.Embedding(self.output_size, self.hidden_size)
		if cfg.use_attention:
			self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
		self.out = nn.Linear(self.hidden_size, self.output_size)

	def forward_step(self, input_var, hidden, encoder_hidden, function):

		batch_size = input_var.size(0)
		embedded = self.input_dropout(self.embedding(input_var))
		if self.use_attention:
			mix = encoder_hidden.contiguous().view(batch_size,-1).unsqueeze(1)
			combined = torch.cat((mix, embedded), dim=2)
			embedded = torch.relu(self.attn_combine(combined.view(-1, 2 * self.hidden_size))).view(batch_size, -1, self.hidden_size)

		output, hidden = self.rnn(embedded, hidden)
		predicted_softmax = function(self.out(output.contiguous().view(batch_size, -1)), dim=1)
		return predicted_softmax, hidden

	def forward(self, encoder_hidden,function=F.log_softmax):

		inputs, decoder_hidden,batch_size = self._init_state(encoder_hidden)

		ret_dict = dict()
		decoder_outputs = []
		sequence_symbols = []
		lengths = np.array([self.max_length] * batch_size)

		def decode(step, step_output):
			symbols = step_output.topk(1)[1]
			decoder_outputs.append(step_output.view(encoder_hidden.size(0),encoder_hidden.size(1),-1))
			sequence_symbols.append(symbols.view(encoder_hidden.size(0),encoder_hidden.size(1),-1))

			eos_batches = symbols.data.eq(self.eos_id)
			if eos_batches.dim() > 0:
				eos_batches = eos_batches.cpu().view(-1).numpy()
				update_idx = ((lengths > step) & eos_batches) != 0
				# TODO: fisrt <eos> should be regarded end idx 
				lengths[update_idx] = len(sequence_symbols) -1
			return symbols

		decoder_input = inputs[:, 0].unsqueeze(1)
		# print(inputs.size(),encoder_hidden.size())
		for di in range(self.max_length):
			decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, encoder_hidden,
																	 function=function)
			symbols = decode(di, decoder_output)
			decoder_input = symbols

		ret_dict['sequence'] = sequence_symbols
		ret_dict['length'] = lengths.reshape((encoder_hidden.size(0),encoder_hidden.size(1))).tolist()

		return decoder_outputs, decoder_hidden, ret_dict

	def compute_similarity(self, encoder_hidden):

		inputs, decoder_hidden,batch_size = self._init_state(encoder_hidden)

		ret_dict = dict()
		sequence_symbols = []

		decoder_input = inputs[:, 0].unsqueeze(1)
		for di in range(self.max_length):
			decoder_input, decoder_hidden = self.forward_step2(decoder_input, decoder_hidden, encoder_hidden)
			sequence_symbols.append(decoder_input.view(encoder_hidden.size(0),encoder_hidden.size(1),-1))

		ret_dict['sequence'] = sequence_symbols

		return ret_dict

	def forward_step2(self, input_var, hidden, encoder_hidden):

		batch_size = input_var.size(0)
		embedded = self.input_dropout(self.embedding(input_var))
		if self.use_attention:
			mix = encoder_hidden.contiguous().view(batch_size,-1).unsqueeze(1)
			embedded = torch.cat((mix, embedded), dim=2)
			embedded = torch.relu(self.attn_combine(embedded.view(-1, 2 * self.hidden_size))).view(batch_size, -1, self.hidden_size)

		output, hidden = self.rnn(embedded, hidden)
		output = self.out(output.contiguous().view(batch_size, -1)).topk(1)[1]
		return output, hidden

	def _init_state(self, encoder_hidden):

		batch_size = encoder_hidden.size(1) * encoder_hidden.size(0)

		inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
		if torch.cuda.is_available():
			inputs = inputs.cuda()

		encoder_hidden = encoder_hidden.contiguous().view(batch_size,self.hidden_size).unsqueeze(0)
		if self.bidirectional_encoder:
			encoder_hidden = encoder_hidden.view(2,batch_size,-1)
		encoder_hidden = encoder_hidden.repeat(self.num_layers,1,1)

		return inputs,encoder_hidden,batch_size
