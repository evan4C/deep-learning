import json
import re
from tqdm import tqdm
import time

with open('./data.json') as file:
   data = file.read()

data = json.loads(data)


for i in tqdm(data):
	sentence=''.join(re.findall(r'[\u4e00-\u9fa5]+',i['weibo_cont']))
	print(sentence)
	time.sleep(1)
