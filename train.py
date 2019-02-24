from model.config import Config
from model.data_core import AddressingDataset
from model.tagging_model import Model
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
	
	
	config = Config(load=True)
	dev = AddressingDataset(config.filename_dev, config.processing_word, config.processing_tag)
	train = AddressingDataset(config.filename_train, config.processing_word, config.processing_tag)
	
	model = Model(config)
	model.build()
	model.train(train, dev)