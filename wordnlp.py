import gensim
import jieba
import os

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line, line.split()
 
sentences = MySentences('./nlp-txt') # a memory-friendly iterator
for line, split in sentences:
    seg_list = jieba.cut(line, cut_all=False)
    a = map( lambda x: print(x), seg_list )
    words = [x for x in seg_list]
    print(words)
#model = gensim.models.Word2Vec(sentences)