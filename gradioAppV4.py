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
    temperature=0.01,
    return_full_text=True,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.1,
    stopping_criteria=stopping_criteria,
)

llm = HuggingFacePipeline(pipeline=pipe)


def getModelEmbeddings():
    model_name = "BAAI/bge-large-en"
    encode_kwargs = {"normalize_embeddings": True}
    model_kwargs = {"device": "cuda"}
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)


def getVectorDB(path):
    loader = TextLoader(path)
    # loader = DirectoryLoader("/home/ambrosemcduffy/Downloads/documents", glob="./*/*.pdf", loader_cls=PyPDFLoader)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0
    )
    embeddings = getModelEmbeddings()
    data_split = text_splitter.split_documents(pages)
    return FAISS.from_documents(data_split, embeddings)


# --------- I'm using the Class below to have streaming text -----
from langchain.llms.base import LLM
from threading import Thread
from typing import Optional
from transformers import TextIteratorStreamer


class CustomLLM(LLM):
    streamer: Optional[TextIteratorStreamer] = None

    def _call(self, prompt, stop=None, run_manager=None) -> str:
        self.streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, Timeout=5
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = move_inputs_to_device(
            inputs, device
        )  # Move inputs to the correct device
        kwargs = dict(
            input_ids=inputs["input_ids"],
            streamer=self.streamer,
            max_new_tokens=512,
        )
        thread = Thread(target=model.generate, kwargs=kwargs)
        thread.start()
        return ""

    @property
    def _llm_type(self) -> str:
        return "custom"


# -------- Memory with a window of 4 ---------------

llm = CustomLLM()

from langchain.chains import LLMSummarizationCheckerChain


# --------- If you don't want to Query Data -------

    
system_message = "\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

template = """
You are Morbo(from Futurama TV Show) A Pixar AI assistant! 🌟 We’re embarking on a journey of discovery. Guidelines for dialogue:
- Always assist with care, respect, and truth.
- Respond with utility yet securely.
- Avoid harmful, unethical, prejudiced, or negative content.
- Promote fairness and positivity.
- If you don't know the answer do not make anything up. Say " I don't know "
- Think through the problem step by step then give the final answer
- Add a Confidence score to the answer

<ctx>
Always answer in one sentence.
You possess additional context related to Pixar:

Within each movie unit, there are several prods. Prods are containers (directories) storing any combination of data (shots, models, textures, scripts, etc).

Types of Prod:
- Personal: Prefix "pp" (e.g., ppdoc, pppu, ppmhedman). Used by a login user for experimentation.
- Mod: Prefix "mod" (e.g., modprops, modchars, modpu). Collections of similar models or assets. (modshared - Data shared among assets)
- Sequence: Unique letter based on the show. (e.g., e100, r175, t122). A series of shots in the movie.
- Dev: Prefix "dev" (e.g., devfx, devgt, devchars). For development - early look dev, test shots, or work on other shows.

Where Prods Live:
Prods are stored in /usr/anim/<UNIT>/<PROD>/ or /shows/<UNIT>/master/<PROD>/. People tend to work in the first directory.
A turnadoodle is a rendered animation of a model from a fixed viewpoint, rotating as if on a turntable.

More additional context possibly related to the question:
{context}
</ctx>

<inst>
<s>[INST] Human: {question}[/INST]</s>
</inst>

Morbo Response [confidence score]:
"""


# --------- This retriever will Query our Data -------


from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


def getRagChain(vectordb):
    QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template)
    
    rag_chain = (
        {"context": vectordb.as_retriever(search_kwargs={"k": 6}), "question": RunnablePassthrough()}
        | QA_CHAIN_PROMPT 
        | llm
        | StrOutputParser() 
    )
    return rag_chain


from langchain.memory import ConversationBufferMemory
def getSequentialChain():
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
    
    CHAIN_PROMPT1 =  ChatPromptTemplate.from_template(first_prompt)
    chain_one = LLMChain(llm=llm, prompt=CHAIN_PROMPT1,output_key="firstAnswer")
    
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
    
    overall_simple_chain = SequentialChain(chains=[chain_one, chain_two],
                                           input_variables=["question", "chat_history"],
                                           output_variables=["firstAnswer", "poem"],
                                           verbose=True)
    return overall_simple_chain

    
def getAgent():
    from langchain.agents import load_tools, initialize_agent
    from langchain.agents import AgentType
    
    tools = load_tools(["llm-math","wikipedia"], llm=llm)
    agent= initialize_agent(
        tools, 
        llm, 
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose = True)
    return agent

def getQARetreiverChain(vectordb):
    template = """
    You are Morbo(from Futurama TV Show) A Pixar AI assistant! 🌟 We’re embarking on a journey of discovery. Guidelines for dialogue:
    - Always assist with care, respect, and truth.
    - Respond with utility yet securely.
    - Avoid harmful, unethical, prejudiced, or negative content.
    - Promote fairness and positivity.
    - If you don't know the answer do not make anything up. Say " I don't know "
    - Think through the problem step by step then give the final answer
    - Add a Confidence score to the answer

    <ctx>
    Always answer in one sentence.
    You possess additional context related to Pixar:

    Within each movie unit, there are several prods. Prods are containers (directories) storing any combination of data (shots, models, textures, scripts, etc).

    Types of Prod:
    - Personal: Prefix "pp" (e.g., ppdoc, pppu, ppmhedman). Used by a login user for experimentation.
    - Mod: Prefix "mod" (e.g., modprops, modchars, modpu). Collections of similar models or assets. (modshared - Data shared among assets)
    - Sequence: Unique letter based on the show. (e.g., e100, r175, t122). A series of shots in the movie.
    - Dev: Prefix "dev" (e.g., devfx, devgt, devchars). For development - early look dev, test shots, or work on other shows.

    Where Prods Live:
    Prods are stored in /usr/anim/<UNIT>/<PROD>/ or /shows/<UNIT>/master/<PROD>/. People tend to work in the first directory.
    A turnadoodle is a rendered animation of a model from a fixed viewpoint, rotating as if on a turntable.

    More additional context possibly related to the question:
    {context}
    </ctx>
    <hst>
    {history}
    </hst>
    <inst>
    <s>[INST] Human: {question}[/INST]</s>
    </inst>

    Morbo Response [confidence score]:
    """
    QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template)
    memory = ConversationBufferWindowMemory(memory_key="history",
                                            k=3,
                                            input_key="question",
                                            return_messages=True)
    # memory = ConversationBufferMemory(memory_key="history",
    #                                   input_key="question", 
    #                                   return_messages=True
    #                                   )
    
    qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 10}),
    return_source_documents=True,
    chain_type="stuff",
    chain_type_kwargs={
        "prompt": QA_CHAIN_PROMPT,
        "verbose": True,
        "memory": memory})
    return qa_chain

def getLLMChain():
    from langchain import PromptTemplate
    from langchain import LLMChain
    # memory = ConversationBufferMemory(memory_key="chat_history",
    #                                         input_key="question",
    #                                         return_messages=True)
    memory = ConversationBufferWindowMemory(memory_key="chat_history",
                                            k=3,
                                            input_key="question",
                                            return_messages=True)

    template = """
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
    [INST] Question: {question} [/INST]</s>
    Answer:"""

    CHAIN_PROMPT = PromptTemplate(
        input_variables=["question"],
        template=template
    )
    
    print("Using basic Chain")
    llm_chain = LLMChain(prompt=CHAIN_PROMPT,
                         llm=llm,
                         memory=memory,
                         verbose=True)
    return llm_chain

import gradio as gr
import json

dark_theme_css = """
body {
    background-color: #2e2e2e;
    color: white;
}
.interface-box {
    background-color: #3c3c3c;
}
.textbox, .button {
    background-color: #4a4a4a;
    color: white;
}
.message-bubble {
    background-color: #565656;
    color: white;
}
"""

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

pause_streaming = False

def pause_text_stream():
    global pause_streaming
    pause_streaming = True

def resume_text_stream():
    global pause_streaming
    pause_streaming = False


from theme_dropdown import create_theme_dropdown  # noqa: F401
import gradio as gr

choices = ["LLM Chain", "QA Chain", "RAG Chain", "Seq Chain"]
chainSelected = 0
filePath = "/home/ambrosemcduffy/chatBotPDF/assetData.txt"

def select_chain(selected_item):
    global chainSelected
    print(f" currently selected {choices[selected_item]}")
    chainSelected = selected_item

vectordb = getVectorDB(filePath)
rag_chain = getRagChain(vectordb)
qa_chain = getQARetreiverChain(vectordb)
llm_chain = getLLMChain()

seqChain = getSequentialChain()
import re

dropdown, js = create_theme_dropdown()
with gr.Blocks(css=dark_theme_css, theme='gradio/default') as demo:
    with gr.Row().style(equal_height=True):
        with gr.Column(scale=5):
            gr.Markdown(
                f"""
                # Mistral-7B-v0.1 AI Pixar Chatbot
                This is an Example Utilizing Langchain and the open-source Large Language Model (LLM) Mistral-7B-v0.1 can significantly enhance analyzing in-house documentation.
                Langchain, a framework designed for developing language model-powered applications, facilitates the connection to various data sources like text files, PDFs, and HTML pages, making it effective for managing and querying documentation.
                Mistral-7B-v0.1, on the other hand, is a 7.3 billion parameter LLM capable of generating coherent text and performing diverse natural language processing tasks.
                It can be deployed locally, offering flexibility for various in-house setups.
                Together, Langchain and Mistral-7B-v0.1 provide a robust solution for intelligently analyzing in-house documentation
                """
            )
        with gr.Column(scale=3):
            with gr.Box():
                dropdown.render()
                toggle_dark = gr.Button(value="Toggle Dark").style(
                    full_width=True
                )
                chainDropdown = gr.Dropdown(choices=choices, label="Select Chain", interactive=True, type="index")
                btn_select = gr.Button(value='Get Chain')
    dropdown.change(None, dropdown, None, _js=js)
    toggle_dark.click(
        None,
        _js="""
        () => {
            document.body.classList.toggle('dark');
            document.querySelector('gradio-app').style.backgroundColor = 'var(--color-background-primary)'
        }
        """,
    )
        
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                avatar_images=("human.png", "Morbo.png"), bubble_full_width=False
            )
            chatbot.like(vote, None, None)
            
            def user(user_message, history):
                return "", history + [[user_message, None]]

            def bot(history):
                global pause_streaming
                if history[-1][0] is None:
                    history[-1][0] = ""     
                
                if choices[chainSelected] == "RAG Chain":
                    flush()
                    rag_chain.invoke(history[-1][0])
                
                if choices[chainSelected] == "QA Chain":
                    flush()
                    chat_history = []
                    qa_chain({"query": history[-1][0],"chat_history":chat_history})
                
                if choices[chainSelected] == "LLM Chain":
                    flush()
                    chat_history = []
                    llm_chain({"question": history[-1][0], "chat_history":chat_history})
                
                if choices[chainSelected] == "seqChain":
                    flush()
                    seqChain({"question": history[-1][0], "chat_history":chat_history})
                
                history[-1][1] = ""
                for chunk in llm.streamer:
                    if pause_streaming:
                            break
                    chunk = chunk.lstrip('\n')  # Remove leading newlines
                    chunk = re.sub('\n+', '\n', chunk)  # Replace consecutive newlines with a single newline
                    
                    for character in chunk:
                        print(character)
                        history[-1][1] += character
                        yield history
                print(history[-1][1])
                resume_text_stream()
           

            with gr.Row(scale=1):
                msg = gr.Textbox(scale=7, max_height=500)
                msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
                
                submit_button = gr.Button(label="Submit", variant="primary")
                submit_button.style(size="sm")
                submit_button.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
                
                pause_button = gr.Button("Pause Streaming", variant="secondary").style(size="sm")
                pause_button.click(pause_text_stream, None, chatbot, queue=False)
                
                clear = gr.Button("Clear", variant="secondary").style(size="sm")
                clear.click(lambda: None, None, chatbot, queue=False)
                btn_select.click(fn=select_chain, inputs=[chainDropdown], outputs=[])
            
            gr.Examples(["What is offramp?",
                            "What is a Prod?",
                            "What is a Unit?",
                            "What are types of Prods?",
                            "how to get a stage in presto with python?",
                            "How to get a job that is mine using offramp?",
                            "What is Local and Global space?",
                            "whats the best way to get a python code review?",
                            "Whats a turnadoodle?",], inputs=[msg])


def same_auth(username, password):
    return username == password

if __name__ == "__main__":
    demo.queue().launch(share=True)
