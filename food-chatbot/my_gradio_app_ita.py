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
def format_message_ita(message: str, history: list, user_info, memory_limit: int = 1):
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

    system_message = " [INST]<<SYS>>\n" \
         "Sei un assistente disponibile, rispettoso e onesto e un recommender di ricette esperto in salute e sostenibilità. " \
         "Se l'utente ti chiede chi sei, rispondi che sei un assistente virtuale che suggerisce ricette e dà consigli sulla salute e la sostenibilità. " \
         "Rispondi sempre nel modo piu' utile possibile, pur essendo sicuro. Lo stile delle tue risposte è persuasivo. " \
         "Le risposte non devono includere contenuti dannosi, non etici, razzisti, sessisti, tossici, pericolosi o illegali. " \
         "Assicurati che le tue risposte siano socialmente imparziali e positive. " \
         "Se una domanda non ha senso o non e' coerente con i fatti, spiegane il motivo invece di rispondere in modo non corretto. " \
         "Se non conosci la risposta a una domanda, non condividere informazioni false.\n" \
         "Le informazioni attualmente note sull'utente sono le seguenti: "+dumps(user_info)+"Fa all'utente domande sul suo nome, le sue allergie, le sue restrizioni alimentari"\
         "e i suoi ingredienti preferiti per conoscerlo meglio"\
         "<</SYS>>\n\n" \

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

# Formatting function for the Italian json query
def format_json_message_ita(message: str, history: list, user_info):
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

    system_message = """Di seguito sono riportate le informazioni conosciute sull'utente in formato json: """ + dumps(user_info)+""" Per memorizzare le informazioni personali dell        'utente, modificare il json iniziale se sono state fornite nuove informazioni esclusivamente nel messaggio dell'utente. Solo i campi per i quali è disponibile una nuova               informazione devono essere modificati, gli altri campi vengono copiati dal json iniziale. Se nel messaggio dell'utente non vengono fornite nuove informazioni personali, il json       iniziale rimane invariato. Scrivere solo il nuovo json senza ulteriori altre frasi o istruzioni."""
    json_formatted_message = f"""\{system_message} L'ultima risposta del modello è la seguente: '{model_previous_answer}'. Utilizzala solo per capire il contesto, assieme al messaggio    utente corrente, per estrarre meglio le informazioni sull'utente. Il messaggio dell'utente da cui estrarre le nuove informazioni è il seguente: '{message}'. Il seguente è il          nuovo profilo utente in formato JSON.
    Il json ha i campi 'nome' (rappresenta il nome dell'utente), 'età'(può assumere un valore numerico in base a quanto dichiarato), 'sesso'(se l'utente è un uomo o una donna o altro)    , 'allergie alimentari', 'ingredienti preferiti'(ingredienti preferiti dall'utente), 'ingredienti non graditi', 'obiettivo di peso' (null di default, può assumere valori tra null,    guadagnare, mantenere o perdere), 'malattie' (può assumere valori uno o più tra null, malattie cardiache, obesità, diabete), 'restrizioni alimentari' (Il valore di default è null,    può assumere valori uno o più tra null, vegano, vegetariano, celiaco, senza latticini). Per costruirlo estrai le informazioni riguardo l'utente solo ed esclusivamente dal             messaggio dell'utente se presenti, altrimenti i valori dei campi considerati rimane invariato. Se per esempio l'utente dicesse 'il mio $campo_json è $valore_campo_json',              aggiornare il json con il valore specificato dall'utente. Se l'utente manda un messaggio tipo 'ciao' o qualsiasi altro messaggio che non contenga nuove informazioni, non              aggiornare il json con informazioni fasulle e inventate. Fa in modo di generare un json corretto. Il json aggiornato è il seguente: ```json"""
    return json_formatted_message


if __name__ == "__main__":
    print(torch.cuda.is_available())
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained("swap-uniba/LLaMAntino-2-chat-7b-hf-UltraChat-ITA")
    model = AutoModelForCausalLM.from_pretrained("swap-uniba/LLaMAntino-2-chat-7b-hf-UltraChat-ITA",
                                                     quantization_config=bnb_config)
    dictionary = {"nome": None, "età": None, "sesso": None, "allergie alimentari": None, "ingredienti preferiti": None, "ingredienti non graditi": None, "obiettivo di peso": None,        "malattie": None, "restrizioni alimentari": None}

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
          instructions = gr.Textbox(value="Comincia la conversazione con il chatbot salutandolo e presentandoti fornendogli qualche informazione personale su di te. Il chatbot può porre ulteriori domande riguardo il tuo stato di salute, le tue allergie e le tue preferenze alimentari. \nCosa si può fare: \n- fare domande al sistema su piatti e ingredienti \n- chiedere di suggerire ricette \n- chiedere di spiegare le ragioni dietro un certo consiglio \n- chiedere al sistema di convincervi a provare una certa ricetta \n- chiedere di confrontare ricette tra loro \n- chiedere di suggerire ricette alternative simili a un piatto specifico \nN.B.: \n- assicurati di fornire al sistema il maggior numero di informazioni possibili su di te \n- assicurati di arricchire i tuoi messaggi con il tuo feedback riguardo le risposte del sistema \n- il tempo di risposta del sistema può variare, si prega di portare pazienza",label="Istruzioni del chatbot",scale=1)
      with gr.Row():
        message = gr.Textbox(label="Scrivi il tuo messaggio per il chatbot qui!",scale=3)
        username = gr.Textbox(label="ID Utente (fare in modo di annotarselo)",interactive=False,scale=1)
      json_file = gr.JSON(value=dictionary,visible=False)
      latest_interaction = gr.Textbox(visible=False)
      callback.setup([latest_interaction, username], "Italian_chatbot_log")
      def get_llamantino_response(message: str, history: list,json_file,username,latest_interaction):
        """
        Generates a conversational response from the Llamantino model and updates
        the JSON with the user's personal information.
  
        Parameters:
            message (str): User's input message.
            history (list): Past conversation history.
  
        Returns:
            str: Generated response from the Llamantino model.
        """
        if(username==""):
          username = ''.join(random.choices(string.ascii_uppercase + string.digits, k=12))
        query = format_message_ita(message, history, json_file)
        json_query = format_json_message_ita(message, history, json_file)
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
        history.append(("Utente: "+message, "Chatbot: "+response.strip()))
        latest_interaction = "Utente: "+message+" Chatbot: "+response.strip()
        return "", history, json_file,username,latest_interaction
  
      message.submit(get_llamantino_response, [message, history, json_file,username,latest_interaction], [message, history, json_file,username,latest_interaction])
      history.change(lambda *args: callback.flag(args), [latest_interaction,username], None, preprocess=False)
    FoodLLM.launch(share=True)