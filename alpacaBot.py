import torch

from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

tokenizer = LlamaTokenizer.from_pretrained("chavinlo/alpaca-native")

alpaca_llm = LlamaForCausalLM.from_pretrained(
    "chavinlo/alpaca-native",
    load_in_8bit=False,
    device_map='auto',
    offload="offload"
)
pipe = pipeline(
    "text-generation",
    model=alpaca_llm, 
    tokenizer=tokenizer, 
    max_length=256,
    temperature=0.6,
    top_p=0.95,
    repetition_penalty=1.2
)
local_model = HuggingFacePipeline(pipeline=pipe)

template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction: 
{instruction}
Answer:"""

prompt = PromptTemplate(template=template, input_variables=["instruction"])
llm_chain = LLMChain(prompt=prompt, llm=local_model)
qn = "What is the water boiling temperature?"
print(llm_chain.run(qn))


# Chat with LangChain memory
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

window_memory = ConversationBufferWindowMemory(k=4)
convo = ConversationChain(
    llm=local_model,
    verbose=True,
    memory=window_memory
)
convo.prompt.template = """The below is a conversation between a human and Alpaca, an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
Current conversation:
{history}
Human: {input}
Alpaca:"""

convo.predict(input="Hey! I am Cedric")
convo.predict(input="What's your name?")