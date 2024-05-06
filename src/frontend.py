import const
import backend

import gradio as gr

from theme_dropdown import create_theme_dropdown  # noqa: F401
import gradio as gr

global chosenTemplate
choices = ["LLM Chain", "Agent", "RAG"]
promptTemplatesChoices = ["Basic Prompt", "Email", "Article Summary", "Document Summary", "Script Template", "Social Media Template"]
chainSelected = 0
promptSelected = 0

filePath = "./documents/assetData.txt"
chosenTemplate = const.llmChainTemplate
chosenTemplateRAG = const.llmChainTemplateForRag


def select_prompt(selected_item):
    global promptSelected
    print(f" currently selected {promptTemplatesChoices[selected_item]}")
    promptSelected = selected_item

    if promptTemplatesChoices[selected_item] == "Email":
        updatePrompt(const.emailTemplate)
    elif promptTemplatesChoices[selected_item] == "Article Summary":
        updatePrompt(const.summaryTemplate)
    elif promptTemplatesChoices[selected_item] == "Document Summary":
        updatePrompt(const.llmChainTemplateForRag)
    elif promptTemplatesChoices[selected_item] == "Script Template":
        updatePrompt(const.scriptTemplate)
    elif promptTemplatesChoices[selected_item] == "Social Media Template":
        updatePrompt(const.socialMediaTemplate)
    else:
        updatePrompt(const.llmChainTemplate)

def select_chain(selected_item):
    global chainSelected
    print(f" currently selected {choices[selected_item]}")
    chainSelected = selected_item
    print(chainSelected)


global vectordb
vectordb = backend.getVectorDB(filePath)
ragChain, memoryRAG = backend.getRagChain(backend.llm, vectordb, const.llmChainTemplateForRag)
llm_chain, memory = backend.getLLMChain(backend.llm, const.llmChainTemplate)
agentChain = backend.getNewAgentChain(backend.llm)

pause_streaming = False
dropdown, js = create_theme_dropdown()
import threading

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

def wrap_in_code_block(text):
    """
    Wraps the text in backticks to create an inline code block in Markdown.
    This prevents Markdown processing within the text.

    Parameters:
    text (str): The text to be wrapped.

    Returns:
    str: The text wrapped in an inline code block.
    """
    # Wrap the text in backticks
    # Use triple backticks to ensure it's interpreted as a code block
    return f"```\n{text}\n```"

def update_chains(llm, template, vectordb=None, isRag=False):
    global ragChain, memoryRAG, llm_chain, memory

    print(llm.temperature, llm.max_tokens)
    if isRag:
        ragChain, memoryRAG = backend.getRagChain(llm, vectordb, const.llmChainTemplateForRag)
    else:
        llm_chain, memory = backend.getLLMChain(llm, template)

def handleTemperatureChange(temperatureValue, state):
    llm = backend.llm
    llm.temperature = temperatureValue
    update_chains(llm, chosenTemplate)
    return state

def handleMaxTokensChange(maxTokensValue, state):
    llm = backend.llm
    llm.max_tokens = maxTokensValue
    update_chains(llm, chosenTemplate)
    return state

def handleTopKChange(topKValue, state):
    llm = backend.llm
    llm.top_k = topKValue
    update_chains(llm, chosenTemplate)
    return state

def handleTopPChange(topPValue, state):
    llm = backend.llm
    llm.top_p = topPValue
    update_chains(llm, chosenTemplate)
    return state

def updatePrompt(newPrompt):
     chosenTemplate = newPrompt
     update_chains(backend.llm, newPrompt)
     return wrap_in_code_block(newPrompt)

def handle_uploaded_file(file_obj):
    if file_obj is not None:
        filePath = file_obj.name
        # Update vectordb with the new file
        llm = backend.llm
        vectordb = backend.getVectorDB(filePath)
        # Update the chains with the new vectordb
        update_chains(llm, const.llmChainTemplate, vectordb, isRag=True)
        return filePath
    return "No file uploaded"

def handle_urlInsert(urlPath):
        print("uploading url path")
        print(urlPath)
        # Update vectordb with the new file
        llm = backend.llm
        vectordb = backend.getVectorDB(urlPath)
        # Update the chains with the new vectordb
        update_chains(llm, const.llmChainTemplate, vectordb, isRag=True)

def getSelectedChain(history):
    if history[-1][0] is None:
        history[-1][0] = ""
                
    if choices[chainSelected] == "Agent":
        history[-1][1] = ""
        question = {"input":history[-1][0]}
        chain = agentChain
    
    if choices[chainSelected] == "RAG":
        # if file_path_display.visible != True:
        #     file_path_display.visible = True
        print("Inside raaaaag!!!!!")
        history[-1][1] = ""
        question = history[-1][0]
        chain = ragChain

    if choices[chainSelected] == "LLM Chain":
        history[-1][1] = ""
        question = {"question": history[-1][0]}
        chain = llm_chain        
    return chain, history, question

def clean_text(hash_map):
    # Get the output text and strip leading/trailing whitespaces and newlines
    cleaned_output = hash_map["text"].strip()
    # Replace carriage returns and newlines within the string
    cleaned_output = cleaned_output.replace('\r', '').replace('\n', ' ')
    # Update the hash map with the cleaned output
    hash_map["text"] = cleaned_output
    return hash_map
 
def runGradioChatApp():
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
                        chainDropdown = gr.Dropdown(
                            choices=choices,
                            label="Select Chain",
                            interactive=True,
                            value=choices[0],
                            type="index",
                        )
                        chainDropdown.change(select_chain, inputs=chainDropdown, outputs=[])
                    with gr.Tab("Parameters"):
                        with gr.Row():
                            with gr.Column():
                                tempSlider = gr.Slider(minimum=0.0, maximum=1.0, randomize=False, label="Temp")
                                tempSlider.value = backend.llm.temperature
                                state = gr.State(value=0)
                                tempSlider.release(handleTemperatureChange, inputs=[tempSlider, state], outputs=[state])

                                maxTokensSlider = gr.Slider(minimum=50, maximum=4096, randomize=False, label="Max Tokens")
                                maxTokensSlider.value = backend.llm.max_tokens
                                maxTokensState = gr.State(value=0)
                                maxTokensSlider.release(handleMaxTokensChange, inputs=[maxTokensSlider, maxTokensState], outputs=[maxTokensState])

                                topKSlider = gr.Slider(minimum=1, maximum=50, randomize=False, label="Top K")
                                topKSlider.value = backend.llm.top_k
                                topKState = gr.State(value=0)
                                topKSlider.release(handleTopKChange, inputs=[topKSlider, topKState], outputs=[topKState])

                                topPSlider = gr.Slider(minimum=0.1, maximum=1.0, randomize=False, label="Top P (Nucleus sampling)")
                                topPSlider.value = backend.llm.top_p
                                topPState = gr.State(value=0)
                                topPSlider.release(handleTopPChange, inputs=[topPSlider, topPState], outputs=[topPState])

                                with gr.Accordion("Prompt Template Selection", open=False):
                                    chainDropdownPrompt = gr.Dropdown(
                                    choices=promptTemplatesChoices,
                                    label="Select PromptTemplate",
                                    interactive=True,
                                    value=promptTemplatesChoices[0],
                                    type="index",
                                )
                                chainDropdownPrompt.change(select_prompt, inputs=chainDropdownPrompt, outputs=[])
                                with gr.Accordion("Custom Prompt", open=False):
                                    promptMarkdown = gr.Markdown(wrap_in_code_block(const.llmChainTemplate))
                                    newPrompt = gr.TextArea(label="New Prompt", visible=True, min_width=400, scale=5, lines=10)
                                    newPrompt.change(fn=updatePrompt, inputs=newPrompt, outputs=promptMarkdown)

                            
                    with gr.Tab("Document upload"):
                        with gr.Row():
                            with gr.Column():
                                uploaded_file = gr.File(label="Upload your text file", type="file", file_types=["txt"], visible=True)
                                file_path_display = gr.Textbox(label="File Path", visible=True)
                                uploaded_file.change(handle_uploaded_file, inputs=[uploaded_file], outputs=[file_path_display])
                                urlPath = gr.Textbox(label="URL Path", visible=True)
                                urlPath.change(handle_urlInsert, inputs=[urlPath], outputs=[])
                            
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
                    
                    chain, history, question = getSelectedChain(history)
                    
                    firstChunk = True

                    # Sometimes Gradio doesn't input my questions so this is a safety for NULL, or Empty questions.
                    if question["question"] == "":
                        question["question"] = "The user did not input any data please respond requesting to do so."
                    
                    for chunk in chain.stream(question):
                        if type(chunk) == dict:
                            if "text" in chunk.keys():
                                chunk = chunk["text"]
                            elif "output" in chunk.keys():
                                chunk = chunk["output"]
                        # Gradio doesn't do well with leading space and newlines so this is strip them.
                        if firstChunk:
                            chunk = chunk.lstrip()
                            firstChunk = False
                        
                        if len(chunk) != 0:
                            for char in chunk:
                                history[-1][1]  = history[-1][1]  + char
                                yield history
                    
                    # This is to capture memory for RAG
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
    return demo
