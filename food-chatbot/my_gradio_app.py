import gradio as gr
from huggingface_hub import login
import torch
from torch.version import cuda
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from guidance import models, gen
import json
import re
from json import dumps
import emoji
import string
import random
import gradio as gr

# Formatting function for the English message and chatbot history to construct a prompt to generate a response for the user
def format_message_eng(message: str, history: list, user_info, memory_limit: int = 1):
    """
    Formats the message and history to costruct a prompt to generate a response for the user.

    Parameters:
        message (str): The message to which the user is expecting a reply.
        history (list): Past conversation history between the user and the system.
        user_info (dictionary): The Json with the user’s information and preferences.
        memory_limit (int): Limit on how many past interactions between the chatbot and the user to consider.

    Returns:
        str: Formatted message string.
    """

    missing_info = []
    for key in user_info:
      if user_info[key] is None:
        missing_info.append(key)

    system_message = """You are a useful food recommender system equipped with knowledge about the user stored in a JSON file. Your responses are as short as possible. Your answers follow a persuasive style and highlight aspects like healthiness and sustainability."""
    if missing_info != []:
      system_message += """Do not ask the user's name if they already provided it. Be sure to start the sentence by asking the user about their """+missing_info[0]+""". In the subsequent turns inquire about any specific information you need from them, both about demographics(age, gender and so on) and preferences/restrictions/goals (allergies, favorite and disliked ingredients,whether the user wants to lose weight or if they have any dietary restriction or disease), taking into account the json structure and inquiring only about the fields that are deemed as null one or two at the time."""
    system_message += """The content of the JSON file you must reference when providing recommendation is the following: """+dumps(user_info)
    prompt_template = f'''[INST] <<SYS>>
    {system_message}
    <</SYS>>'''

    # always keep len(history) <= memory_limit
    if len(history) > memory_limit:
        history = history[-memory_limit:]

    if len(history) == 0:
        return system_message + f"{message} [/INST]"

    formatted_message = system_message + f"{history[0][0]} [/INST] {history[0][1]} </s>"
    # Handle conversation history
    for user_msg, model_answer in history[1:]:
        formatted_message += f"<s>[INST] {user_msg} [/INST] {model_answer} </s>"
    # Handle the current message
    formatted_message += f"<s>[INST] {message} [/INST]"

    return formatted_message

# Formatting function for the English message and chatbot history to construct a prompt to generate the user Json
def format_json_message_eng(message: str, history: list, user_info):
    """
    Formats the message and history to costruct a prompt to generate the user Json.

    Parameters:
        message (str): The latest message that was submitted by the user.
        history (list): Past conversation history between the user and the system.
        user_info (dictionary): The Json with the user’s information and preferences.
    Returns:
        str: Formatted message string.
    """

    model_previous_answer=""
    system_message=""
    if len(history) > 0:
        history = history[-1:]
        for user_msg, model_answer in history[0:]:
          model_previous_answer = emoji.replace_emoji(model_answer)
          #model_previous_answer = f"""\The last model answer is the following '{model_answer}'. Use it alongside the current user message to better extract user information."""

    system_message = """The following are the known user's information in json format: """ + dumps(
        user_info) + """ To save the user's personal information modify the initial json if any new information has been provided exclusively in the user message. Only the fields for which a new information is available must be modified, the other fields are copied from the initial json. If no new personal information is provided in the user message, the initial json stays the same.Write only the new json without further instructions or messages."""
    json_formatted_message = f"""\{system_message} The last model answer is the following '{model_previous_answer}'. Use it alongside the current user message to better extract user information. The user message is the following: '{message}'. The following is the new user profile in JSON format.
        It has fields 'name', 'age', 'gender', 'food allergies', 'favorite ingredients',  'disliked ingredients', 'goal weight' (null by default, one between null, gain, mantain or lose), 'diseases' (one or more between null, heart disease, obesity, diabete), 'dietary restrictions'(null by default, one or more between null, vegan, vegetarian, celiac, dairy-free). The information have to be extracted exclusively from the user's message if present, otherwise the considered fields are kept the same ```json"""
    return json_formatted_message


if __name__ == "__main__":

    print(torch.cuda.is_available())
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    #access_token_read = "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"
    #login(token=access_token_read)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", padding_side='left')
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", quantization_config=bnb_config,pad_token_id=0)
    dictionary = {"name": None, "age": None, "gender": None, "food allergies": None, "favorite ingredients": None, "disliked ingredients": None, "weight goal": None, "diseases": None,                   "dietary restrictions": None}
    
    # Define the pipeline
    llama_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )
    # load a model (could be Transformers, LlamaCpp, VertexAI, OpenAI...)
    llama2 = models.Transformers(model, tokenizer)
    
    callback = gr.CSVLogger()
    with gr.Blocks() as FoodLLM:
        with gr.Row():
          history = gr.Chatbot(scale=3,height=460)
          instructions = gr.Textbox(value="Start the conversation by greeting the chatbot and introducing yourself by stating some personal information. The chatbot can ask you further questions about your health status, your allergies and your food preferences. \n What you can do: \n- ask questions to the system about dishes and ingredients\n- ask for recipe suggestions\n- ask to explain the reasons behind a certain recommendation\n- ask the system to persuade you to try a certain recipe\n- ask to compare recipes with one another\n- ask to suggest alternative recipes similar to a specific dish\nN.B.: \n- make sure to provide the system as much information about you as possible\n- make sure to enrich your messages to the system with your feedback about the system's responses\n- the response time of the system may vary, be patient",label="Chatbot instructions",scale=1)
        with gr.Row():
          message = gr.Textbox(label="Write your message here!",scale=3)
          username = gr.Textbox(label="User's ID (be sure to note it down for later)",interactive=False,scale=1)
        json_file = gr.JSON(value=dictionary,visible=False)
        latest_interaction = gr.Textbox(visible=False)
        callback.setup([latest_interaction, username], "English_chatbot_log")
        def get_llama_response(message: str, history: list,json_file,username,latest_interaction):
          """
          Generates a conversational response from the Llama model and updates
          the JSON with the user's personal information.
    
          Parameters:
              message (str): User's input message.
              history (list): Past conversation history.
    
          Returns:
              str: Generated response from the Llama model.
          """
          if(username==""):
            username = ''.join(random.choices(string.ascii_uppercase + string.digits, k=12))
          query = format_message_eng(message, history, json_file)
          json_query = format_json_message_eng(message, history, json_file)
          response = ""
    
          with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            sequences = llama_pipeline(
                query,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512
            )
          torch.cuda.empty_cache()
          generated_text = sequences[0]['generated_text']
          response = generated_text[len(query):]  # Remove the prompt from the output
          with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            lm = llama2 + json_query + gen(name="json", temperature=0.01, stop='```')
            json_file = json.loads(lm["json"])
            torch.cuda.empty_cache()
          history.append(("User: "+message, "Chatbot: "+response.strip()))
          latest_interaction = "User: "+message+" Chatbot: "+response.strip()
          return "", history, json_file,username,latest_interaction
    
        message.submit(get_llama_response, [message, history, json_file,username,latest_interaction], [message, history, json_file,username,latest_interaction])
        history.change(lambda *args: callback.flag(args), [latest_interaction,username], None, preprocess=False)
    FoodLLM.launch(share=True)