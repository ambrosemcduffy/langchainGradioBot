import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

def main():
    tokenizer = LlamaTokenizer.from_pretrained("chavinlo/alpaca-native")
    alpaca_llm = LlamaForCausalLM.from_pretrained("chavinlo/alpaca-native", load_in_8bit=True, device_map='auto')
    pipe = pipeline(
        "text-generation",
        model=alpaca_llm, 
        tokenizer=tokenizer, 
        max_length=1024,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.2
    )
    local_model = HuggingFacePipeline(pipeline=pipe)

    template = """The below is a conversation between a human and Alpaca, an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    Current conversation:
    {history}
    Human: {input}
    Alpaca: """

    prompt = PromptTemplate(template=template, input_variables=["history", "input"])
    convo = ConversationChain(
        llm=local_model,
        verbose=False,
        memory=ConversationBufferWindowMemory(k=4),
        prompt=prompt
    )

    while True:
        user_input = input("You: ")
        response = convo.predict(input=user_input)
        print("Alpaca:", response)

        if user_input.lower() == "exit":
            break


if __name__ == "__main__":
    main()

