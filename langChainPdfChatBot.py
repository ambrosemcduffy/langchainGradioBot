import os
import argparse

import textwrap
import requests

from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# adding in my huggingFace API Key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
llm=HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature":0, "max_length":512})
chain = load_qa_chain(llm, chain_type="stuff")

# Loading in the Text
loader = TextLoader('./data.txt')
documents = loader.load()


def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text


# Text Splitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embeddings)
query = "what is offramp?"
docs = db.similarity_search(query)
answer = chain.run(input_documents=docs, question=query)
print(answer)

query = "If I wanted to get my jobs and sort by priority how would I do that?"
docs = db.similarity_search(query)
answer = chain.run(input_documents=docs, question=query)
print(answer)

query = "How do I find jobs that are mine?"
docs = db.similarity_search(query)
answer = chain.run(input_documents=docs, question=query)
print(answer)






# query = "What did the president say about economy?"
# docs = db.similarity_search(query)
# print(chain.run(input_documents=docs, question=query))


from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator





pdf_doc = "paper.pdf"

loaders = UnstructuredPDFLoader(pdf_doc)
#print(loaders)

index = VectorstoreIndexCreator(
embedding=HuggingFaceEmbeddings(),
text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
).from_loaders([loaders])

llm=HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0, "max_length":512})

from langchain.chains import RetrievalQA
chain = RetrievalQA.from_chain_type(llm=llm, 
                                     chain_type="stuff", 
                                     retriever=index.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":10}), 
                                     input_key="question")



query = 'Who are the authors?'
docs = db.similarity_search(query, k=20)
print(chain.run(input_documents = docs, question=query))

# query = 'Who are the authors of the paper?'
# docs = db.similarity_search(query, k=10)
# print(chain.run(input_documents = docs, question=query))


# query = 'What is a summary of the paper?'
# docs = db.similarity_search(query, k=20)
# print(chain.run(input_documents = docs, question=query))


# end = time.time()

# print((end-start)*100)