import os
import torch
import gradio as gr
from langchain import PromptTemplate, LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline, TextStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms.base import LLM
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from llama_cpp import Llama

from threading import Thread
from typing import Optional
from transformers import TextIteratorStreamer
from transformers import StoppingCriteria, StoppingCriteriaList

import warnings
import gradio as gr
import json
import const
import gc

warnings.filterwarnings("ignore", category=UserWarning)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


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

if device == "mps":
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.llms import LlamaCpp

    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
    n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path="./models/llama-2-7b-arguments.Q8_0.gguf",
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        max_tokens=2046,
        n_ctx=8192,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )
else:
    
    # model_id = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
    # model_id = "google/gemma-7b-it"
    # model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.float16
    )
    model.tie_weights()
    model.eval()


tokenizer = AutoTokenizer.from_pretrained(model_id)
stop_list = ["\nHuman:", "\n```\n"]
stop_token_ids = [tokenizer(x)["input_ids"] for x in stop_list]
stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]


# ----- Define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                return True
        return False


stopping_criteria = StoppingCriteriaList([StopOnTokens()])


def move_inputs_to_device(inputs, device):
    """This function moves the our inputs to the correct device."""
    for key in inputs:
        inputs[key] = inputs[key].to(device)
    return inputs


def getVectorDB(path):
    """Gets vector db for our documents."""
    loader = TextLoader(path)
    # loader = DirectoryLoader("/home/ambrosemcduffy/Downloads/documents", glob="./*/*.pdf", loader_cls=PyPDFLoader)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    embeddings = getModelEmbeddings()
    data_split = text_splitter.split_documents(pages)
    return FAISS.from_documents(data_split, embeddings)


def getModelEmbeddings():
    """Gets Model Embeddings"""
    model_name = "BAAI/bge-large-en"
    encode_kwargs = {"normalize_embeddings": True}
    model_kwargs = {"device": "mps"}
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

if device != "mps":
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    
    class CustomLLM(LLM):
        """This is a custom LLM class similar to pipeline in HuggingFace"""

        streamer: Optional[TextIteratorStreamer] = None

        def _call(self, prompt, stop=None, run_manager=None) -> str:
            self.streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, Timeout=5)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = move_inputs_to_device(
                inputs, device
            )  # Move inputs to the correct device
            kwargs = dict(
                input_ids=inputs["input_ids"],
                streamer=self.streamer,
                max_new_tokens=512,
            )
            thread = Thread(target=llm.stream, kwargs=kwargs)
            thread.start()
            return ""

        @property
        def _llm_type(self) -> str:
            return "custom"
    
    
    llm = CustomLLM()
    

def getLLMChain():
    """This function is a basic LLMChain to be simple
    it's an non-query data llm, so a basic llm model.
    """
    from langchain import PromptTemplate
    from langchain import LLMChain

    memory = ConversationBufferMemory(
        memory_key="chat_history", input_key="question", return_messages=True
    )
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", k=5, input_key="question", return_messages=True
    )

    CHAIN_PROMPT = PromptTemplate(
        input_variables=["question"], template=const.llmChainTemplate
    )

    print("Using basic Chain")
    llm_chain = LLMChain(prompt=CHAIN_PROMPT, llm=llm, memory=memory, verbose=True)
    return llm_chain


def getRagChain(vectordb):
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"], template=const.documentQueryTemplate
    )

    rag_chain = (
        {
            "context": vectordb.as_retriever(search_kwargs={"k": 6}),
            "question": RunnablePassthrough(),
        }
        | QA_CHAIN_PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain


def getQARetreiverChain(vectordb):
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"], template=const.qaTemplate
    )
    memory = ConversationBufferWindowMemory(
        memory_key="history", k=3, input_key="question", return_messages=True
    )
    memory = ConversationBufferMemory(
        memory_key="history", input_key="question", return_messages=True
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 10}),
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": QA_CHAIN_PROMPT,
            "verbose": True,
            "memory": memory,
        },
    )
    return qa_chain

def getNewLLMChain():
    from operator import itemgetter
    from langchain.memory import ConversationBufferMemory
    from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
    prompt = PromptTemplate(
    input_variables=["context", "question", "history"], template=const.llmChainTemplate1
    )
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    chain = (
    RunnablePassthrough.assign(
        history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
    )
    | prompt
    | llm
    )
    return chain, memory

def getSequentialChain():
    """Sequential Chain these are multiple LLMS being queried one after another."""
    from langchain.prompts.chat import ChatPromptTemplate
    from langchain.chains import SequentialChain

    first_prompt = """
    <s>[INST] Your are a helpful AI chatbot use to help answer the users question you will adhere to these guidelines:
    1. Use a formal tone
    2. Explain step by step be thorough and thoughtful.
    3. If unsure about the answer, state "I don't know" without fabricating details.
    4. For code, ensure it's clean and commented for easy understanding.
    5. Rephrase my question for better querying.
    6. Always provide an answer, even if brief, for answer unsure say "I'm unsure about this one" or "I don't know".
    7. Always prioritize accuracy and clarity in your answers. [/INST]</s>
    Previous Interaction:
    {chat_history}
    [INST] Question: What is the best way to answer this question think about it before answering this question:  {question} [/INST]</s>
    Answer:
    """

    # Chain 1

    CHAIN_PROMPT1 = ChatPromptTemplate.from_template(first_prompt)
    chain_one = LLMChain(llm=llm, prompt=CHAIN_PROMPT1, output_key="firstAnswer")

    second_prompt = """
    <s>[INST] Your are a helpful AI chatbot use to help answer the users question you will adhere to these guidelines:
    1. Use a formal tone
    2. Explain step by step be thorough and thoughtful.
    3. If unsure about the answer, state "I don't know" without fabricating details.
    4. For code, ensure it's clean and commented for easy understanding.
    5. Rephrase my question for better querying.
    6. Always provide an answer, even if brief, for answer unsure say "I'm unsure about this one" or "I don't know".
    7. Always prioritize accuracy and clarity in your answers. [/INST]</s>
    [INST] Question: Write a poem from this answer{firstAnswer} [/INST]</s>
    Poem:
    """
    CHAIN_PROMPT2 = ChatPromptTemplate.from_template(second_prompt)

    # chain 2
    chain_two = LLMChain(llm=llm, prompt=CHAIN_PROMPT2, output_key="poem")

    overall_simple_chain = SequentialChain(
        chains=[chain_one, chain_two],
        input_variables=["question", "chat_history"],
        output_variables=["firstAnswer", "poem"],
        verbose=True,
    )
    return overall_simple_chain


def getAgent():
    from langchain.agents import load_tools, initialize_agent
    from langchain.agents import AgentType

    tools = load_tools(["llm-math", "wikipedia"], llm=llm)
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True,
    )
    return agent


# Tested this with langchain '0.0.179'
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


def pause_text_stream():
    global pause_streaming
    pause_streaming = True


def resume_text_stream():
    global pause_streaming
    pause_streaming = False
