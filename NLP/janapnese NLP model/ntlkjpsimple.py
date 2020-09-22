import nltk


jp_sent_tokenizer = nltk.RegexpTokenizer(u'[^ 「」!?。．）]*[!?。]')
jp_chartype_tokenizer = nltk.RegexpTokenizer(u'([ぁ-んー]+|[ァ-ンー]+|[\u4e00-\u9FFF]+|[^ぁ-んァ-ンー\u4e00-\u9FFF]+)')


print (jp_chartype_tokenizer.tokenize(u'の各宣言を実行しておく必要があることに注意しよう。これ以下の節では、各スクリプト例の前にこれらがすでに宣言されていることを前提とする。'))
