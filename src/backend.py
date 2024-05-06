import os
import torch

import gradio as gr

from langchain import PromptTemplate, LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import (
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredPDFLoader,
    WebBaseLoader,
    NewsURLLoader,
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.llms.base import LLM
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.globals import set_debug

from threading import Thread
from typing import Optional

from transformers import TextIteratorStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer

import warnings
import gradio as gr
import json
import const
import gc

warnings.filterwarnings("ignore", category=UserWarning)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

set_debug(True)


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# checking for devices for acceleration
def getDevice():
    """This function finds the correct device to place the model on."""
    if torch.cuda.is_available():
        print(f"Detected cuda device User Nvidia GPU: {torch.cuda.get_device_name()}")
        # if Using for cuda devices
        device = f"cuda:{torch.cuda.current_device()}"

    elif torch.backends.mps.is_available():
        print("Detected MPS device use MacOS GPU METAL:")
        device = "mps"
    else:
        device = "cpu"
    return device


# Initialize our LLM
device = getDevice()
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
global llm


class CustomLLM(LLM):
    """This is a custom LLM class similar to pipeline in HuggingFace"""

    streamer: Optional[TextIteratorStreamer] = None

    def _call(self, prompt, stop=None, run_manager=None) -> str:
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, Timeout=5
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = move_inputs_to_device(
            inputs, device
        )  # Move inputs to the correct device
        kwargs = dict(
            input_ids=inputs["input_ids"],
            streamer=self.streamer,
            max_new_tokens=512,
        )
        thread = Thread(target=getHuggingFaceModel().generate(), kwargs=kwargs)
        thread.start()
        return ""

    @property
    def _llm_type(self) -> str:
        return "custom"


def getLLM():
    if device != "mps":
        return CustomLLM(LLM)
    return getLlamaCppModel()


def getLlamaCppModel():
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain.llms import LlamaCpp

    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    n_gpu_layers = (
        -1
    )  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
    n_batch = (
        512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    )

    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        #model_path="./models/llama-2-7b-arguments.Q8_0.gguf",
        model_path="./models/mistral-7b-instruct-v0.1.Q8_0.gguf",
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        max_tokens=900,
        n_ctx=4096,
        temperature=0.3,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )
    return llm


def getHuggingFaceModel():
    llm = AutoModelForCausalLM.from_pretrained(
        model_id, device_map=device, torch_dtype=torch.float16
    )
    llm.tie_weights()
    llm.eval()
    return llm


def move_inputs_to_device(inputs, device):
    """This function moves the our inputs to the correct device."""
    for key in inputs:
        inputs[key] = inputs[key].to(device)
    return inputs


def getVectorDB(path, useChroma=False, useFaiss=True):
    """Gets vector db for our documents."""

    loader = getLoader(path)
    # Data loader and splitting of data
    data_split = getDataSplit(loader)
    embeddings = getModelEmbeddings()
    print("This is the embeddings..\n\n")
    print(embeddings)
    print("\n\n")

    ### This code below allows for Chromadb
    if useChroma:
        return Chroma.from_documents(data_split, embedding=embeddings)
    return FAISS.from_documents(data_split, embeddings)


def getModelEmbeddings():
    """Gets Model Embeddings"""
    model_name = "BAAI/bge-large-en"
    encode_kwargs = {"normalize_embeddings": True}
    model_kwargs = {"device": "mps"}
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)


def getDataSplit(loader):
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
    data_split = text_splitter.split_documents(pages)
    print("This is the data split..\n\n")
    print(data_split)
    print("\n\n")
    return data_split


def getLoader(path):
    if path[-3:] == "txt":
        loader = TextLoader(path)
    elif path[-3:] == "pdf":
        loader = UnstructuredPDFLoader(path)
        print("using pdf")
    elif path[-4:] == "xlsx":
        loader = UnstructuredExcelLoader(
            path,
        )
    elif (path[:4] == "http") or (path[-5:] == ".com/" or path[-4:] == ".com"):
        # print("in webloader")
        # print(path)
        # loader = WebBaseLoader(path)
        loader = NewsURLLoader(urls=path)
    print("This is the loader..\n\n")
    print(loader)
    print("\n\n")
    return loader


def getLLMChain(llm, template):
    from operator import itemgetter
    from langchain.memory import ConversationBufferMemory
    from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

    prompt = PromptTemplate(
        input_variables=["context", "question", "history"], template=template
    )
    # memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    memory = ConversationBufferWindowMemory(
        memory_key="history", k=4, return_messages=True
    )
    chain = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt
        | llm
    )
    return chain, memory


def getRagChain(llm, vectordb, template):
    from operator import itemgetter
    from langchain.memory import ConversationBufferMemory
    from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question", "history"], template=template
    )
    memory = ConversationBufferWindowMemory(
        memory_key="history", k=4, return_messages=True
    )
    rag_chain = (
        {
            "context": vectordb.as_retriever(search_kwargs={"k": 3}),
            "question": RunnablePassthrough(),
            "history": RunnableLambda(memory.load_memory_variables)
            | itemgetter("history"),
        }
        | QA_CHAIN_PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain, memory


def getNewAgentChain(llm):
    from langchain.agents import load_tools, initialize_agent
    from langchain.agents import AgentType

    PREFIX = """<<SYS>> You are an AI ChatBot AGent Designed to help students with Research. Follow these guidelines:
    1. Adopt a formal tone throughout the interaction.
    2. Provide detailed, step-by-step explanations, ensuring your responses are thorough and considerate.
    3. If you know the answer, provide it directly and clearly. If the answer is unknown or uncertain, state "I don't know" or "I'm unsure" and explain why, without fabricating details.
    4. When presenting code, ensure it is clean, well-commented, and easy for users to understand.
    5. Rephrase and clarify the user's question if necessary to ensure accurate understanding and responses.
    6. Aim to provide a direct answer to every question. If a direct answer isn't possible, guide the user on how they might find the answer or suggest alternative approaches to address their inquiry.
    7. Ensure your responses are accurate, clear, and detailed, helping the user gain a comprehensive understanding of the topic.
    8. When detailed information is not available, guide the user on how to find reliable sources or suggest potential avenues for further inquiry.
    9. In cases involving recent developments or current events, advise users to check the latest information, as situations can evolve rapidly. 
    10. Provide information bullet points and be detailed oriented. .<</SYS>>\n"""

    tools = load_tools(["wikipedia"], llm=llm)
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=10,
        agent_kwargs={
            "prefix": PREFIX,
            #         'format_instructions': FORMAT_INSTRUCTIONS,
            #         'suffix': SUFFIX
        },
    )
    return agent


def vote(data: gr.LikeData):
    file_path = "chat_data.json"
    # Check if the JSON file exists, if not, create one with an empty list
    if not os.path.exists(file_path):
        with open(file_path, "w") as file:
            json.dump([], file)

    # Load existing data from the JSON file
    with open(file_path, "r") as file:
        existing_data = json.load(file)

    # Update data with the new vote
    vote_data = {"liked": data.liked, "value": data.value}
    existing_data.append(vote_data)

    # Write updated data back to the JSON file
    with open(file_path, "w") as file:
        json.dump(existing_data, file, indent=4)  # indent for better formatting

    # Print feedback to the user
    if data.liked:
        print(f"You upvoted this response: {data.value}")
    else:
        print(f"You downvoted this response: {data.value}")


llm = getLLM()
