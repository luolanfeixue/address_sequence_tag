from model.config import Config
from model.data_core import AddressingDataset,get_processing,\
	get_vocabs,get_embedding_vocab,UNK,write_vocab,load_vocab,save_used_embedding

if __name__ == '__main__':
	
	config = Config(load=False)
	processing = get_processing()
	dev = AddressingDataset(config.filename_dev)
	train = AddressingDataset(config.filename_dev)
	# test = AddressingDataset(config.filename_dev)
	# Build Word and Tag vocab
	vocab_words, vocab_tags = get_vocabs([train, dev])
	vocab_embedding = get_embedding_vocab(config.filename_embedding)
	
	vocab_all = vocab_words & vocab_embedding
	vocab_all.add(UNK)
	
	# Save vocab
	write_vocab(vocab_all, config.filename_words)
	write_vocab(vocab_tags, config.filename_tags)
	
	word_to_idx = load_vocab(config.filename_words)
	save_used_embedding(word_to_idx,config.filename_embedding,config.filename_used_embedding,config.dim_word)
	
	# print(word_to_idx)
	