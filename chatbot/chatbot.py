import os
import sys

import constant_api
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain import llms

os.environ["OPENAI_API_KEY"] = constant_api.APIKEY

query = sys.argv[1]

loader = TextLoader('./data.txt')

index = VectorstoreIndexCreator().from_loaders([loader])
print(query)
print(index.query(query, llm=llms.OpenAIChat()))
