import os

from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


llm=HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":512})
chain = load_qa_chain(llm, chain_type="stuff")



import requests

url = "https://raw.githubusercontent.com/hwchase17/langchain/master/docs/modules/state_of_the_union.txt"
res = requests.get(url)
with open("state_of_the_union.txt", "w") as f:
  f.write(res.text)

  # Document Loader
from langchain.document_loaders import TextLoader
loader = TextLoader('./state_of_the_union.txt')
documents = loader.load()


import textwrap

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


# Text Splitter
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


print(wrap_text_preserve_newlines(str(documents[0])))

# Embeddings
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()
# Vectorstore: https://python.langchain.com/en/latest/modules/indexes/vectorstores.html
from langchain.vectorstores import FAISS

db = FAISS.from_documents(docs, embeddings)


query = "What did the president say about the Supreme Court"
docs = db.similarity_search(query)

print(wrap_text_preserve_newlines(str(docs[0].page_content)))

query = "What did the president say about the Supreme Court"
docs = db.similarity_search(query)
print(chain.run(input_documents=docs, question=query))

query = "What did the president say about economy?"
docs = db.similarity_search(query)
chain.run(input_documents=docs, question=query)


from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator

pdf_doc = "paper.pdf"

loaders = UnstructuredPDFLoader(pdf_doc)
print(loaders)

index = VectorstoreIndexCreator(
embedding=HuggingFaceEmbeddings(),
text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)).from_loaders([loaders])

llm=HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":512})

from langchain.chains import RetrievalQA
chain = RetrievalQA.from_chain_type(llm=llm, 
                                    chain_type="stuff", 
                                    retriever=index.vectorstore.as_retriever(), 
                                    input_key="question")


print(chain.run('what is a summary on Kernel-Predicting Convolutional Networks for Denoising Monte Carlo Renderings?'))
