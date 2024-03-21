import const
import backend

import gradio as gr
import json


from theme_dropdown import create_theme_dropdown  # noqa: F401
import gradio as gr

choices = ["LLM Chain"]
chainSelected = 0
filePath = "./documents/assetData.txt"


def select_chain(selected_item):
    global chainSelected
    print(f" currently selected {choices[selected_item]}")
    chainSelected = selected_item


vectordb = backend.getVectorDB(filePath)
rag_chain = backend.getRagChain(vectordb)
qa_chain = backend.getQARetreiverChain(vectordb)
llm_chain, memory = backend.getNewLLMChain()
seqChain = backend.getSequentialChain()
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
                

with gr.Blocks(css=const.dark_theme_css, theme="gradio/default") as demo:
    controller = StreamController()

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
                toggle_dark = gr.Button(value="Toggle Dark").style(full_width=True)
                chainDropdown = gr.Dropdown(
                    choices=choices,
                    label="Select Chain",
                    interactive=True,
                    type="index",
                )
                btn_select = gr.Button(value="Get Chain")
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
                avatar_images=(const.imagePathHuman, "/Users/ambrosemcduffy/Documents/langchainGradioBot/images/bot.png"),
                bubble_full_width=False,
            )
            chatbot.like(backend.vote, None, None)

            def user(user_message, history):
                return "", history + [[user_message, None]]

            def bot(history):
                global pause_streaming
                global current_chain
                if history[-1][0] is None:
                    history[-1][0] = ""

                if choices[chainSelected] == "LLM Chain":
                    history[-1][1] = ""
                    for chunk in llm_chain.stream({"question": history[-1][0]}):
                        if type(chunk) == dict:
                            if chunk["text"] != None:
                                chunk = chunk["text"]
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
                    memory.save_context({"input": history[-1][0]}, {"output": history[-1][1]})
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
                
                
                btn_select.click(fn=select_chain, inputs=[chainDropdown], outputs=[])
                stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[click_event, msg_clickEvent])
            with gr.Row(scale=1):
                clear = gr.Button("Clear", variant="secondary").style(size="sm")
                clear.click(lambda: None, None, chatbot, queue=False)

            gr.Examples(
                [
                    "What is offramp?",
                    "What is a Prod?",
                    "What is a Unit?",
                    "What are types of Prods?",
                    "how to get a stage in presto with python?",
                    "How to get a job that is mine using offramp?",
                    "What is Local and Global space?",
                    "whats the best way to get a python code review?",
                    "Whats a turnadoodle?",
                ],
                inputs=[msg],
            )


def same_auth(username, password):
    return username == password


if __name__ == "__main__":
    demo.queue().launch(share=True)
