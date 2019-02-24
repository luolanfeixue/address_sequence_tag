import numpy as np
import os


UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"


# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
        ERROR: Unable to locate file {}.
        
        FIX: Have you tried running python build_data.py first?
        This will build vocab file from your train, test and dev sets and
        trimm your word vectors.
        """.format(filename)
        super(MyIOError, self).__init__(message)


class AddressingDataset(object):

    """
    地址数据的迭代生成器类
    
    __iter__ yield a tuple( words, tags)
        words:list of raw words
        tags: list of raw tags
   Example:
    ```python
    data = AddressingDataset(filename)
    for sentence, tags in data:
        pass
    ```
    """
    
    def __init__(self, filename, processing_word=None, processing_tag=None):
        """
        
        :param filename: 文件路径
        :param processing_word: 处理字
        :param processing_tag: 处理tag
        :param max_iter: yield 的最大句字数
        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        # self.max_iter = max_iter
        self.length = None
    
    def __iter__(self):
        """
        一次迭代返回一个句子，遇到len(line) == 0（空行）就返回一次，然后接着该行继续往下读。
        :return:
        """
        niter = 0
        with open(self.filename) as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    if len(words) != 0:
                        niter += 1
                        # if self.max_iter is not None and niter > self.max_iter:
                        # 	break
                        yield words, tags
                        words, tags = [], []
                else:
                    ls = line.split(' ')
                    # print(ls)
                    if len(ls) != 2:
                        print(line)
                        continue
                    word, tag = ls[0], ls[1]
                    if self.processing_word is not None:
                        word = self.processing_word(word)  # 如果有字典库，则按照字典库的编号返回编号。没有则返回字本身
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)  # 如果有tag库，则按照tag库的编号返回编号。没有则返回tag本身
                    words += [word]
                    tags += [tag]
    
    def __len__(self):
        """
        :return: 迭代器AddressingDataset的长度，本质就是所读数据有多少个句子
        """
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1
        return self.length


def get_processing(vocab=None, allow_unk=True):
    
    """
    返回一个处理函数。如果存在字（tag）库，则返回该字（tag）的编号，如果不存在字（tag）库则返回字（tag）本身
    :param vocab: dict[word] = word_idx或者 dict[tag] = tag_idx
    :param allow_unk:
    :return: 如果指定vocab， f('京') = (1234) f('B_PRO') = (1) 否则  f('京') = '京'
    """
    def f(word):
        if vocab is not None:
            if word in vocab:
                word = vocab[word]
            else:
                if allow_unk:
                    word = vocab[UNK]
                else:
                    raise Exception("key {}  不存在，请检查字（tag）库 是否正确 ".format(word))
        else:
            word = word.upper()
        return word
    
    return f


def get_vocabs(datasets):
    """
    通过datasets 对象的迭代建立字典库
    :param datasets: list of dataset object
    :return: all words and all tags
    """
    
    print("建立字典库")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, tags in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)
    print('字典库建立完成，{} 个字,{} 个tag'.format(len(vocab_words), len(vocab_tags)))
    return vocab_words, vocab_tags

def get_embedding_vocab(filename):
    """
    load embedding vocab
    :param filename: path to embedding vector
    :return: set() of chinese word
    """
    print("建立 embeding 词典")
    vocab = set()
    with open(filename) as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print('embeding字典库建立完成，{} 个字'.format(len(vocab)))
    return vocab

def write_vocab(vocab, filename):

    """
    写入字典，一行一个字，将来行数就是字的编码
    :param vocab:
    :param filename:
    :return:
    """
    print('写入库....')
    with open(filename,'w') as f:
        for i,word in enumerate(vocab):
            if i !=len(vocab)-1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print('写入完成，共 {} 个'.format(len(vocab)))

def load_vocab(filename):
    
    """
    加载字库
    :param filename:
    :return:
    """
    try:
        word_to_id = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()
                word_to_id[word] = idx
    except:
        raise MyIOError(filename)
    return word_to_id

def save_used_embedding(word_to_id,filename_embedding,filename_used_embedding,dim_word):
    
    """
    将能够用到的embedding压缩保存下来
    :param word_to_id:
    :param filename_embedding:
    :param filename_used_embedding:
    :param dim_word:
    :return:
    """
    
    used_embeddings = np.zeros([len(word_to_id),dim_word])
    with open(filename_embedding) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in word_to_id:
                idx = word_to_id[word]
                used_embeddings[idx] = np.asarray(embedding)
    np.savez_compressed(filename_used_embedding,embeddings=used_embeddings)
    
    
def load_used_embedding(filename):
    
    """
    将压缩的embedding load到np中
    :param filename:
    :return:
    """
    
    try:
        with np.load(filename) as data:
            return data['embeddings']
    except IOError:
        raise MyIOError(filename)
    
    
def minibatches(data, mini_batch_size):
    """
    每次迭代产生一个batch
    :param data:
    :param mini_batch_size:
    :return:x_batch 一个batch的句子,  [[word_id1, word_id2],[word_id3, word_id4]]
            y_batch 一个bathc的label [[tag_id1,  tag_id2 ],[tag_id3,  tag_id4 ]]
    """
    x_batch, y_batch = [], []
    for (x_list, y_list) in data:
        if len(x_batch) == mini_batch_size:
            yield x_batch, y_batch
            x_batch , y_batch = [], []
        x_batch += [x_list]
        y_batch += [y_list]
    if (len(x_batch)) != 0 :
        yield x_batch, y_batch

def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position398

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks

def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type