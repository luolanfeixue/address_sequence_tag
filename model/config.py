import os

from .general_utils import get_logger
from .data_core import *

class Config():


    def __init__(self, load=True):
        """
        初始化参数并且加载语料

        参数：
        load_embedding
        """

        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        if load:
            self.load()

    def load(self):
        
        self.word_to_id = load_vocab(self.filename_words)
        self.tag_to_id = load_vocab(self.filename_tags)
        
        self.nwords = len(self.word_to_id)
        self.ntags = len(self.tag_to_id)
        
        self.processing_word = get_processing(self.word_to_id)
        self.processing_tag = get_processing(self.tag_to_id,allow_unk=False)

        self.embeddings = load_used_embedding(self.filename_used_embedding)


    # geneal_config
    dir_output = 'result/'
    path_log = dir_output + "log.txt"
    dir_model = dir_output + "model.weights/"

    dim_word = 200

    #dataset
    filename_dev = 'data/raw_data/test_data'
    filename_test = 'data/raw_data/test_data'
    filename_train = 'data/raw_data/train_data'
    filename_embedding = 'data/wordembedding/Tencent_AILab_ChineseEmbedding.txt'
    filename_used_embedding = 'data/wordembedding/used_embedding.npz'

    # dataset test
    filename_dev = filename_test = filename_train = 'data/raw_data/test.txt'
    
    filename_words = 'data/words.txt'
    filename_tags = 'data/tags.txt'

    max_iter = None  # if not None, max number of examples in Dataset

    use_crf = True

    # training
    train_embeddings = False
    nepochs          = 15
    dropout          = 0.5
    batch_size       = 32
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1  # if negative, no clipping
    nepoch_no_imprv  = 3

    hidden_size_lstm = 300  # lstm on word embeddings

