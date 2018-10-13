import gensim
import jieba
import os

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            if fname == '.DS_Store':
                continue
            #path = os.path.join(self.dirname, fname)
            #print(path)
            #print(gensim.models.Doc2Vec(path))
            #break
            for line in open(os.path.join(self.dirname, fname), encoding='utf-8'):
                yield line

#创建停用词list
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath,'r',encoding='utf-8').readlines()]
    return stopwords

'''
def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist('stop_words_zh.txt') #加载停用词
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word !='\t':
                outstr +=word
                outstr +=''
    return outstr

inputs = open('cc.txt','rb')
outputs = open('tt.txt','w')

for line in inputs:
    line_seg = seg_sentence(line)
    outputs.write(line_seg+'\n')
outputs.close()
inputs.close()
'''

 
sentences = MySentences('./nlp-txt') # a memory-friendly iterator
for line in sentences:
    seg_list = jieba.cut(line, cut_all=False)
    words = [x for x in seg_list]
    print(words)
#model = gensim.models.Word2Vec(sentences)