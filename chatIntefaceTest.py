import os
from typing import Optional, Tuple

import gradio as gr
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.llms import OpenAI
from langchain.schema import AIMessage, HumanMessage


import os
import time
import sys
import torch
import gradio as gr
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
    TextStreamer,
)
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import gc


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


flush()
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
device = (
    f"cuda:{torch.cuda.current_device()}"
    if torch.cuda.is_available()
    else "cpu"
)  # Determine the device

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto",
    torch_dtype=torch.bfloat16
    )

model.tie_weights()
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)
from transformers import TextIteratorStreamer

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

stop_list = ["\nHuman:", "\n```\n"]
stop_token_ids = [tokenizer(x)["input_ids"] for x in stop_list]
stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

from transformers import StoppingCriteria, StoppingCriteriaList


# ----- Define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        stop_ids = [29, 0]
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                return True
        return False


stopping_criteria = StoppingCriteriaList([StopOnTokens()])


# Move inputs to the correct device
def move_inputs_to_device(inputs, device):
    for key in inputs:
        inputs[key] = inputs[key].to(device)
    return inputs

streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)


pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=False,
    use_cache=True,
    device_map="auto",
    max_new_tokens=2046,
    max_length=2000,
    do_sample=True,
    # temperature=0.01,
    streamer = streamer,
    top_k=1000,
    num_beams=1,
    return_full_text=True,
    temperature=1.0,
    num_return_sequences=1,
    # eos_token_id=tokenizer.eos_token_id,
    # pad_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.1,
    stopping_criteria=stopping_criteria,
)

llm = HuggingFacePipeline(pipeline=pipe)

    
from langchain.prompts import ChatPromptTemplate
from threading import Thread

def getLLMChain():
    from langchain import PromptTemplate
    from langchain import LLMChain
    
    # Create a ConversationBufferMemory instance
    memory = ConversationBufferMemory(memory_key="chat_history",
                                      return_messages=True)
    
    template = """
    <s>[INST] You are an AI providing detailed and clear answers. Always answer the question. Adhere to these guidelines.
    Previous Interaction:
    {chat_history}
    Question: {question}
    Answer:"""
    CHAIN_PROMPT = ChatPromptTemplate.from_template(template)
    
    # CHAIN_PROMPT = PromptTemplate(
    #     input_variables=["question"],
    #     template=template
    # )
    
    print("Using basic Chain")
    llm_chain = LLMChain(prompt=CHAIN_PROMPT,
                         llm=llm,
                         memory=memory,
                         verbose=True)
    return llm_chain
chain = getLLMChain()

def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    t = Thread(target=chain({"question": message})["text"])
    t.start()
    partial_message  = ""
    for new_token in streamer:
        if new_token != '<':
            partial_message += new_token
            yield partial_message
    
block = gr.Blocks()

with block:
    
    chatbot = gr.ChatInterface(
    fn=predict,
    title="Chatbot")

block.queue().launch()