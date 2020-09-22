
import json
from tqdm import tqdm
import jieba
from gensim import corpora
import re
import gensim

def load_data():
    """
    @description: 读取爬取好的微博数据
    @param
    @return: 读取到的数据
    """
    with open('./data.json') as file:
       data = file.read()
    data = json.loads(data)
    return data

def fenci_data():
        '''
        @description: 微博数据分词
        @param 
        @return: text处理好的数据，格式是[['xx','xx'],['xx','xx'],...,['xx','xx']]
        '''
        text = []
        data = load_data()
        with open('./hit_stopwords.txt') as file:
            stop_word_list = file.read()
        for weibo_item in tqdm(data):
            tmp = []
            sentence=''.join(re.findall(r'[\u4e00-\u9fa5]+',weibo_item['weibo_cont']))
            for word in jieba.lcut(sentence):
                if word not in stop_word_list:
                    tmp.append(word)
            text.append(tmp)
        return text

def weibo_lda():
        '''
        @description: 生成dictionary, corpus
        @param 
        @return: 生成的字典，dictionary：{序号：单词}比如{0:你好}
        @return: 生成的词袋，corpus ： {序号：单词的频数}比如{0:3}就是0号单词（你好）出现了3次 
        '''
        text = fenci_data()
        dictionary = corpora.Dictionary(text)
        corpus = [dictionary.doc2bow(tmp) for tmp in text]
        return dictionary, corpus

def choose_topic():
        '''
        @description: 生成模型
        @param 
        @return: 生成主题数分别为1-15的LDA主题模型，并保存起来。
        '''
        dictionary, corpus = weibo_lda()
        texts = fenci_data()
        for i in range(1,3):
            print('目前的topic个数:{}'.format(i))
            print('目前的数据量:{}'.format(len(texts)))
            temp = 'lda_{}_{}'.format(i,len(texts))
            tmp = gensim.models.ldamodel.LdaModel(corpus, num_topics=i, id2word=dictionary, passes=20)
            file_path = './{}.model'.format(temp)
            tmp.save(file_path)
            print('------------------')


choose_topic()

