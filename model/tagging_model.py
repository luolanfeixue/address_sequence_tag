import numpy as np
import os
import tensorflow as tf

from .data_core import *
from .general_utils import Progbar

from tensorflow.python import debug as tf_debug

# from tensorflow.keras.utils import Progbar


class Model():
	
	def __init__(self, config):
		self.config = config
		self.logger = config.logger
		self.id_to_tag = {id: tag for tag, id in self.config.tag_to_id.items()}
		# self.batch_size = config.batch_size
		self.debug_see = None
		self.sess = None
		self.saver = None
	
	def build(self):
		# shape = (batch_size, sentence length)
		self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name='input_data')
		# shape = (batch_size, sentence length)
		self.labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
		# shape = (batch_size) 里面存的是每个元素是sequence的长度
		self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
		
		# hyper parameters
		self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
		self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
		
		with tf.variable_scope("words"):
			_word_embeddings = tf.Variable(self.config.embeddings,
			                                   name="_word_embeddings",
			                                   dtype=tf.float32,
			                                   trainable=self.config.train_embeddings)
			self.word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids, name='word_embeddings')
		with tf.variable_scope('bi-lstm'):
			self.cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size_lstm)
			self.cell_bw = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size_lstm)
			(output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, self.word_embeddings,
			                                                sequence_length=self.sequence_lengths, dtype=tf.float32)
			output = tf.concat([output_fw, output_bw], axis = -1)
			output = tf.nn.dropout(output, self.dropout)
			
		with tf.variable_scope('proj'):
			W = tf.get_variable("W", dtype=tf.float32, shape=[2*self.config.hidden_size_lstm, self.config.ntags],
			                    initializer=tf.truncated_normal_initializer)
			b = tf.get_variable("b", dtype=tf.float32, shape=[self.config.ntags], initializer=tf.zeros_initializer)
			nsteps = tf.shape(output)[1]
			output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
			pred = tf.matmul(output, W) + b
			self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])

			# CRF Loss
			log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.labels,
			                                                                 self.sequence_lengths)
			self.trans_params = trans_params
			self.loss = tf.reduce_mean(-log_likelihood)

		self.add_train_op(self.config.lr_method, self.lr, self.loss, self.config.clip)
		self.initialize_session()
		
	def add_train_op(self, lr_method, lr, loss, clip):
		_lr_m = lr_method.lower()
		with tf.variable_scope('train_step'):
			if _lr_m =='adam':
				optimizer = tf.train.AdamOptimizer(lr)
			elif _lr_m == 'adagrad':
				optimizer = tf.train.AdagradDAOptimizer(lr)
			elif _lr_m == 'sgd':
				optimizer = tf.train.GradientDescentOptimizer(lr)
			elif _lr_m == 'rmsprob':
				optimizer = tf.train.RMSPropOptimizer()
			else:
				raise NotImplementedError('Unknown methond {}'.format(_lr_m))
			
			if clip > 0 :
				grads, vs = zip(*optimizer.compute_gradients(loss))
				grads, gnorm = tf.clip_by_global_norm(grads, clip)
				self.train_op = optimizer.apply_gradients(zip(grads, vs))
			else:
				self.train_op = optimizer.minimize(loss)
				
				
	def initialize_session(self):
		
	
		# config = tf.ConfigProto()
		# config.gpu_options.per_process_gpu_memory_fraction = 0.5  # maximun alloc gpu50% of MEM
		# config.gpu_options.allow_growth = True  # allocate dynamically
		self.logger.info("Initializing tf session")
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		self.debug_sess = tf_debug.LocalCLIDebugWrapperSession(sess=self.sess)
		self.saver = tf.train.Saver()
	

	
	def train(self, train, dev):
		"""
		使用early_stopping 和 lr 指数衰减
		:param train: yield tuple(句子，tags）
		:param dev:
		:return:
		"""
		best_score = 0
		nepoch_no_imprv = 0  # for early_stopping
		
		# TODO add_summary() # for tensorboard
		for epoch in range(self.config.nepochs):
			self.logger.info("Epoch {:} out ot {:}".format(epoch + 1, self.config.nepochs))
			score = self.run_epoch(train, dev)
			self.config.lr *= self.config.lr_decay
			if score >= best_score:
				nepoch_no_imprv = 0
				self.save_session()
				best_score = score
				self.logger.info("--new best score")
			else:
				nepoch_no_imprv += 1
				if nepoch_no_imprv >= self.config.nepoch_no_imprv:
					self.logger.info('-- early stopping {} epochs without improvement'.format(nepoch_no_imprv))
	
	def run_epoch(self, train, dev):
		"""
		训练一次epoch，训练完一次就用dev评估一次准确率。以便上面做early stop
		"""
		
		batch_size = self.config.batch_size
		nbatches = (len(train) + batch_size - 1) // batch_size
		prog = Progbar(target=nbatches)
		
		for i, (sentences, labels) in enumerate(minibatches(train, batch_size)):
			fd, _ = self.get_feed_dict(sentences, labels, self.config.lr, self.config.dropout)
			_, train_loss = self.sess.run([self.train_op, self.loss], feed_dict=fd)
			prog.update(i + 1, [("train loss", train_loss)])
		metrics = self.run_evaluate(dev)
		msg = " - ".join(["{} {:04.2f}".format(k, v)
		                  for k, v in metrics.items()])
		self.logger.info(msg)
		
		return metrics['f1']
	
	def run_evaluate(self, test):
		"""
		评估在测试集的表现
		:param test:
		:return:
		"""
		accs = []
		correct_preds, total_correct, total_preds = 0., 0., 0.,
		for words, labels in minibatches(test, self.config.batch_size):
			labels_pred, sequence_lengths = self.predict_batch(words)
			for lab, lab_pred, length in zip(labels, labels_pred,
			                                 sequence_lengths):
				lab = lab[:length]
				lab_pred = lab_pred[:length]
				accs += [a == b for (a, b) in zip(lab, lab_pred)]
				
				lab_chunks = set(get_chunks(lab, self.config.tag_to_id))
				lab_pred_chunks = set(get_chunks(lab_pred, self.config.tag_to_id))
				
				correct_preds += len(lab_chunks & lab_pred_chunks)
				total_preds += len(lab_pred_chunks)
				total_correct += len(lab_chunks)
		
		p = correct_preds / total_preds if correct_preds > 0 else 0
		r = correct_preds / total_correct if correct_preds > 0 else 0
		f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
		acc = np.mean(accs)
		return {"acc": 100 * acc, "f1": 100 * f1}
	
	
	def predict_batch(self, words):
		"""
		Args:
			words: list of sentences

		Returns:
			labels_pred: list of labels for each sentence
			sequence_length

		"""
		fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)
		
		viterbi_sequences = []
		logits, trans_params = self.sess.run(
			[self.logits, self.trans_params], feed_dict=fd)
		
		# iterate over the sentences because no batching in vitervi_decode
		for logit, sequence_length in zip(logits, sequence_lengths):
			logit = logit[:sequence_length]  # keep only the valid steps
			viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
				logit, trans_params)
			viterbi_sequences += [viterbi_seq]
		return viterbi_sequences, sequence_lengths
	
	
	def get_feed_dict(self, sentences, labels=None, lr=None, dropout=None):
		"""
		:param sentences: [[word_id1,word_id2],[word_id3,word_id4]]
		:param labels: [[tag_id1,tag_id2],[tag_id3,tag_id4]]
		:param lr:
		:param dropout:·
		:return:
		"""
		
		sequence_padded, sequence_lengths = self.pad_sequences(sentences, 0)
		
		feed = {
			self.word_ids: sequence_padded,
			self.sequence_lengths: sequence_lengths
		}
		
		if labels is not None:
			labels, _ = self.pad_sequences(labels, 0)
			feed[self.labels] = labels
			
		if lr is not None:
			feed[self.lr] = lr
		
		if dropout is not None:
			feed[self.dropout] = dropout
		
		return feed,sequence_lengths
	
	def pad_sequences(self, sequences, pad_tok):
		"""
		sequence_padded 为 sequences中每个个sequence最大长度
		sequence_lengths 为原来每个sequence的实际长度
		:param sequences:[[1,2],[3,4,5],[6,7,8,9]]
		:param pad_tok:
		:param max_length:
		:return: sequence_padded:[[1,2,0,0],[3,4,5,0],[6,7,8,9]]
				 sequence_lengths:[2,3,4]
		"""
		max_length = max(map(lambda x: len(x), sequences))
		# max_length = 26  for debug
		sequence_padded, sequence_lengths = [], []
		for seq in sequences:
			seq = list(seq)
			seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
			sequence_padded += [seq_]
			sequence_lengths += [min(len(seq), max_length)]
		
		return sequence_padded, sequence_lengths
	
	
	def save_session(self):
		"""Saves session = weights"""
		if not os.path.exists(self.config.dir_model):
			os.makedirs(self.config.dir_model)
		self.saver.save(self.sess, self.config.dir_model)