import os
from pathlib import Path
import multiprocessing
from itertools import takewhile
import operator
import threading
import numpy as np
import queue
import time
import tiktoken
import re

from ._model import Chat, Instruct
from ._remote import Remote

try:
    # TODO: can we eliminate the torch requirement for llama.cpp by using numpy in the caller instead?
    import openai as openai_package
    is_openai = True
except ImportError:
    is_openai = False

chat_model_pattern = r'^(ft:)?(gpt-3\.5-turbo|gpt-4)(?:(?!-instruct$)(-\w+)+)?(:[\w-]+(?:[:\w-]+)*)?(::\w+)?$'

class OpenAI(Remote):
    def __init__(self, model, tokenizer=None, echo=True, caching=True, api_key=None,
                 temperature=0.0, top_p=1.0, max_streaming_tokens=1000, **kwargs):
        '''Build a new OpenAI model object that represents a model in a given state.

        This class automatically subclasses itself into the appropriate OpenAIChat, OpenAIInstruct,
        or OpenAICompletion subclass based on the model name.
        
        Parameters
        ----------
        model : str
            The name of the OpenAI model to use (e.g. gpt-3.5-turbo).
        tokenizer : None or tiktoken.Encoding
            The tokenizer to use for the given model. If set to None we use `tiktoken.encoding_for_model(model)`.
        echo : bool
            If true the final result of creating this model state will be displayed (as HTML in a notebook).
        api_key : None or str
            The OpenAI API key to use for remote requests, passed directly to the `openai.OpenAI` constructor.
        temperature : float
            The default temperature to use for generation requests. Note this default value may be overridden by the
            grammars that are executed.
        top_p : float
            The default top_p to use for generation requests. Note this default value may be overridden by the
            grammars that are executed.
        max_streaming_tokens : int
            The maximum number of tokens we allow this model to generate in a single stream. Normally this is set very
            high and we rely either on early stopping on the remote side, or on the grammar terminating causing the
            stream loop to break on the local side. This number needs to be longer than the longest stream you want
            to generate.
        **kwargs : 
            All extra keyword arguments are passed directly to the `openai.OpenAI` constructor. Commonly used argument
            names include `base_url` and `organization`
        '''

        if not is_openai or not hasattr(openai_package, "OpenAI"):
            raise Exception("Please install the openai package version >= 1 using `pip install openai -U` in order to use guidance.models.OpenAI!")
        
        # if we are called directly (as opposed to through super()) then we convert ourselves to a more specific subclass if possible
        if self.__class__ is OpenAI:
            found_subclass = None

            # chat
            if re.match(chat_model_pattern, model):
                found_subclass = OpenAIChat

            # instruct
            elif "instruct" in model:
                found_subclass = OpenAIInstruct

            # regular completion
            else:
                found_subclass = OpenAICompletion
            
            # convert to any found subclass
            self.__class__ = found_subclass
            found_subclass.__init__(self, model, tokenizer=tokenizer, echo=echo, caching=caching, api_key=api_key, temperature=temperature, max_streaming_tokens=max_streaming_tokens, **kwargs)
            return # we return since we just ran init above and don't need to run again

        self.client = openai_package.OpenAI(api_key=api_key, **kwargs)
        self.model_name = model
        self.top_p = top_p

        if tokenizer is None:
            tokenizer = tiktoken.encoding_for_model(model)

        super().__init__(
            model, tokenizer=tokenizer, echo=echo,
            caching=caching, temperature=temperature,
            max_streaming_tokens=max_streaming_tokens
        )

class OAICompletionMixin(Instruct):

    def _generator(self, prompt, temperature):
        
        self._reset_shared_data(prompt, temperature) # update our shared data state

        try:
            generator = self.client.completions.create(
                model=self.model_name,
                prompt=prompt.decode("utf8"), 
                max_tokens=self.max_streaming_tokens, 
                n=1, 
                top_p=self.top_p, 
                temperature=temperature, 
                stream=True
            )
        except Exception as e: # TODO: add retry logic
            raise e

        for part in generator:
            if len(part.choices) > 0:
                chunk = part.choices[0].text or ""
            else:
                chunk = ""
            yield chunk.encode("utf8")

class OAIInstructMixin(Instruct):
    def get_role_start(self, name):
        return ""
    
    def get_role_end(self, name):
        if name == "instruction":
            return "<|endofprompt|>"
        else:
            raise Exception(f"The OpenAIInstruct model does not know about the {name} role type!")

    def _generator(self, prompt, temperature):
        # start the new stream
        prompt_end = prompt.find(b'<|endofprompt|>')
        if prompt_end >= 0:
            stripped_prompt = prompt[:prompt_end]
        else:
            raise Exception("This model cannot handle prompts that don't match the instruct format!")
        
        # make sure you don't try and instruct the same model twice
        if b'<|endofprompt|>' in prompt[prompt_end + len(b'<|endofprompt|>'):]:
            raise Exception("This model has been given two separate instruct blocks, but this is not allowed!")
        
        # update our shared data state
        self._reset_shared_data(stripped_prompt + b'<|endofprompt|>', temperature)

        try:
            generator = self.client.completions.create(
                model=self.model_name,
                prompt=self._shared_state["data"].decode("utf8"), 
                max_tokens=self.max_streaming_tokens, 
                n=1, 
                top_p=self.top_p, 
                temperature=temperature, 
                stream=True
            )
        except Exception as e: # TODO: add retry logic
            raise e

        for part in generator:
            if len(part.choices) > 0:
                chunk = part.choices[0].text or ""
            else:
                chunk = ""
            yield chunk.encode("utf8")

class OAIChatMixin(Chat):
    def _generator(self, prompt, temperature):

        # find the role tags
        pos = 0
        role_end = b'<|im_end|>'
        messages = []
        found = True
        while found:

            # find the role text blocks
            found = False
            for role_name,start_bytes in (("system", b'<|im_start|>system\n'), ("user", b'<|im_start|>user\n'), ("assistant", b'<|im_start|>assistant\n')):
                if prompt[pos:].startswith(start_bytes):
                    pos += len(start_bytes)
                    end_pos = prompt[pos:].find(role_end)
                    if end_pos < 0:
                        assert role_name == "assistant", "Bad chat format! Last role before gen needs to be assistant!"
                        break
                    btext = prompt[pos:pos+end_pos]
                    pos += end_pos + len(role_end)
                    messages.append({"role": role_name, "content": btext.decode("utf8")})
                    found = True
                    break
        
        
        
        # Add nice exception if no role tags were used in the prompt.
        # TODO: Move this somewhere more general for all chat models?
        if messages == []:
            raise ValueError(f"The OpenAI model {self.model_name} is a Chat-based model and requires role tags in the prompt! \
            Make sure you are using guidance context managers like `with system():`, `with user():` and `with assistant():` \
            to appropriately format your guidance program for this type of model.")
        
        # update our shared data state
        self._reset_shared_data(prompt[:pos], temperature)

        try:
                
            generator = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_streaming_tokens,
                n=1,
                top_p=self.top_p,
                temperature=temperature,
                stream=True
            )
        except Exception as e: # TODO: add retry logic
            raise e
        for part in generator:
            if len(part.choices) > 0:
                chunk = part.choices[0].delta.content or ""
            else:
                chunk = ""
            yield chunk.encode("utf8")

class OpenAICompletion(OpenAI, OAICompletionMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class OpenAIInstruct(OpenAI, OAIInstructMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class OpenAIChat(OpenAI, OAIChatMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)