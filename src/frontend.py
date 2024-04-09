import const
import backend

import gradio as gr
import json


from theme_dropdown import create_theme_dropdown  # noqa: F401
import gradio as gr

choices = ["LLM Chain", "Agent", "RAG"]
chainSelected = 0

filePath = "./documents/assetData.txt"


def select_chain(selected_item, fileDisplay):
    global chainSelected
    print(f" currently selected {choices[selected_item]}")
    chainSelected = selected_item
    print(chainSelected)
    if choices[selected_item] == "RAG":
        print("in rag chain")
        fileDisplay.visible=False

global vectordb
# vectordb = backend.getVectorDB(filePath)

# ragChain, memoryRAG = backend.getRagChain(vectordb)
# qa_chain = backend.getQARetreiverChain(vectordb)
llm_chain, memory = backend.getNewLLMChain(backend.llm)
seqChain = backend.getSequentialChain()
agentChain = backend.getNewAgentChain(backend.llm)

import re
import time
pause_streaming = False

def streamChain(chain, history):
    history[-1][1] = ""
    for chunk in chain.stream(history[-1][0]):
        if pause_streaming:
            break
        if len(chunk["text"]) != 0:
            chunk = chunk["text"].lstrip("\n")
            # chunk = re.sub(
            #  "\n+", "\n", chunk
            #  )
            for char in chunk:
                history[-1][1]  = history[-1][1]  + char
                yield history

dropdown, js = create_theme_dropdown()
import threading
class StreamController:
    def __init__(self):
        self.paused = False
        self.condition = threading.Condition()

    def pause(self):
        with self.condition:
            self.paused = True

    def resume(self):
        with self.condition:
            self.paused = False
            self.condition.notify_all()

    def wait_while_paused(self):
        with self.condition:
            while self.paused:
                self.condition.wait()

def update_chains(llm, vectordb=None, isRag=False):
    global ragChain, memoryRAG, llm_chain, memory
    if isRag:
        ragChain, memoryRAG = backend.getRagChain(llm, vectordb)
    else:
        llm_chain, memory = backend.getNewLLMChain(llm)

def handleTemperatureChange(temperatureValue, state):
    llm = backend.llm
    llm.temperature = temperatureValue
    update_chains(llm)
    return state

def handle_uploaded_file(file_obj):
    if file_obj is not None:
        filePath = file_obj.name
        # Update vectordb with the new file
        llm = backend.llm
        vectordb = backend.getVectorDB(filePath)
        # Update the chains with the new vectordb
        update_chains(llm, vectordb, isRag=True)
        return filePath
    return "No file uploaded"

with gr.Blocks(css=const.dark_theme_css, theme="gradio/default") as demo:
    controller = StreamController()
    with gr.Row().style(equal_height=True):
        with gr.Column(scale=5):
            gr.Markdown(
                f"""
                # LLAMA-2.7B AI Chatbot
                This is an example utilizing Langchain and the open-source Large Language Model (LLM) LLAMA-2.7B can significantly enhance analyzing in-house documentation.
                Langchain, a framework designed for developing language model-powered applications, facilitates the connection to various data sources like text files, PDFs, and HTML pages, making it effective for managing and querying documentation.
                LLAMA-2.7B, on the other hand, is a 7.3 billion parameter LLM capable of generating coherent text and performing diverse natural language processing tasks.
                It can be deployed locally, offering flexibility for various in-house setups.
                Together, Langchain and LLAMA-2.7B provide a robust solution for intelligently analyzing in-house documentation
                """
            )
        with gr.Column(scale=3):
            with gr.Group():
                with gr.Tab("Chain Selections"):
                    # dropdown.render()
                    # toggle_dark = gr.Button(value="Toggle Dark").style(full_width=True)
                    chainDropdown = gr.Dropdown(
                        choices=choices,
                        label="Select Chain",
                        interactive=True,
                        value=choices[0],
                        type="index",
                    )
                    btn_select = gr.Button(value="Get Chain")
                with gr.Tab("Parameters"):
                    with gr.Row():
                        with gr.Column():
                            tempSlider = gr.Slider(minimum=0.0, maximum=1.0, randomize=False, label="Temp")
                            tempSlider.value = backend.llm.temperature
                            state = gr.State(value=0)
                            tempSlider.release(handleTemperatureChange, inputs=[tempSlider, state], outputs=[state])
                            with gr.Accordion("Prompt", open=False):
                                prompt = gr.Markdown(const.llmChainTemplate)
                                newPrompt = gr.Textbox(label="New Prompt", visible=True)
                        
                with gr.Tab("Document upload"):
                    with gr.Row():
                        with gr.Column():
                            uploaded_file = gr.File(label="Upload your text file", type="file", file_types=["txt"], visible=True)
                            file_path_display = gr.Textbox(label="File Path", visible=True)
                            uploaded_file.change(handle_uploaded_file, inputs=[uploaded_file], outputs=[file_path_display])
                        
    
    dropdown.change(None, dropdown, None, _js=js)
    # toggle_dark.click(
    #     None,
    #     _js="""
    #     () => {
    #         document.body.classList.toggle('dark');
    #         document.querySelector('gradio-app').style.backgroundColor = 'var(--color-background-primary)'
    #     }
    #     """,
    # )

    with gr.Row():
        
        with gr.Column(scale=4):
            with gr.Row():  
                chatbot = gr.Chatbot(
                    avatar_images=(const.imagePathHuman, "/Users/ambrosemcduffy/Documents/langchainGradioBot/images/bot.png"),
                    bubble_full_width=False,
                )
                chatbot.like(backend.vote, None, None)
                
                    


            
            def user(user_message, history):
                return "", history + [[user_message, None]]

            def bot(history):
                global pause_streaming
                global current_chain
                print(backend.llm.temperature)
                if history[-1][0] is None:
                    history[-1][0] = ""
                
                if choices[chainSelected] == "Agent":
                    history[-1][1] = ""
                    question = {"input":history[-1][0]}
                    chain = agentChain
                
                if choices[chainSelected] == "RAG":
                    # if file_path_display.visible != True:
                    #     file_path_display.visible = True
                    history[-1][1] = ""
                    question = history[-1][0]
                    chain = ragChain

                if choices[chainSelected] == "LLM Chain":
                    history[-1][1] = ""
                    question = {"question": history[-1][0]}
                    chain = llm_chain

                for chunk in chain.stream(question):
                    if type(chunk) == dict:
                        if "text" in chunk.keys():
                            chunk = chunk["text"]
                        elif "output" in chunk.keys():
                            chunk = chunk["output"]
                    if len(chunk) != 0:
                        for char in chunk:
                            history[-1][1]  = history[-1][1]  + char
                            yield history
                    else:
                        yield history
                        break
                    controller.wait_while_paused()
                
                if controller.paused:
                    print("It's paused")
                if chainSelected != "RAG":
                    memory.save_context({"input": history[-1][0]}, {"output": history[-1][1]})
                else:
                    memoryRAG.save_context({"input": history[-1][0]}, {"output": history[-1][1]})
                return history
                    
            msg = gr.Textbox(scale=7, max_height=500)
            with gr.Row(scale=1):
                msg_clickEvent = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                    bot, chatbot, chatbot
                )
                
                submit_button = gr.Button(label="Submit", variant="primary")
                submit_button.style(size="sm")
                click_event = submit_button.click(
                    user, [msg, chatbot], [msg, chatbot], queue=False
                ).then(bot, chatbot, chatbot)
                
                stop_btn = gr.Button("Stop Streaming", variant="secondary").style(
                    size="sm"
                )
                
                btn_selectClickEvent = btn_select.click(fn=select_chain, inputs=[chainDropdown, uploaded_file], outputs=[]).then()

                stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[click_event, msg_clickEvent])
            with gr.Row(scale=1):
                clear = gr.Button("Clear", variant="secondary").style(size="sm")
                clear.click(lambda: None, None, chatbot, queue=False)

            gr.Examples(
                [
                    "What are the major themes of the paper?",
                    "Breakdown the paper.. Provide Subheadings.",
                    "Provide a Summary for the paper.",
                ],
                inputs=[msg],
            )


def same_auth(username, password):
    return username == password


if __name__ == "__main__":
    demo.queue().launch(share=True)
