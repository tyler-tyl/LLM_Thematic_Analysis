"""
message = [
{'role': 'system','content': '''You are a research and writing assistant. You write in a concise, academic, straightforward tone.'''},
{'role': 'user','content': ''''''},
{'role': 'user','content': ''''''},
#{'role': 'assistant','content': ''''''},
#{'role': 'user','content': ''''''},
#{'role': 'assistant','content': ''''''},
#{'role': 'user','content': ''''''},
#{'role': 'assistant','content': ''''''},
#{'role': 'user','content': ''''''},
#{'role': 'assistant','content': ''''''},
#{'role': 'user','content': ''''''},
#{'role': 'assistant','content': ''''''},
#{'role': 'user','content': ''''''},
#{'role': 'assistant','content': ''''''},
#{'role': 'user','content': ''''''},
#{'role': 'assistant','content': ''''''},
]
"""

import json
import os

# api_key = os.environ["OPENAI_API_KEY"]

project_key = os.environ["TMAI_API_KEY"]

from openai import OpenAI

client = OpenAI(api_key=project_key)

import tiktoken

span = '_' * 120
red = '\033[91m'
green = '\033[92m'
yellow = '\033[93m'
blue = '\033[94m'
pink = '\033[95m'
teal = '\033[96m'
grey = '\033[97m'
black = '\033[90m'
defaultcolor = '\033[99m'

class st:
    reset = '\033[0m'
    bold = '\033[01m'
    disable = '\033[02m'
    underline = '\033[04m'
    reverse = '\033[07m'
    strikethrough = '\033[09m'
    invisible = '\033[08m'

class fg:
    black = '\033[30m'
    red = '\033[31m'
    green = '\033[32m'
    orange = '\033[33m'
    blue = '\033[34m'
    purple = '\033[35m'
    cyan = '\033[36m'
    lightgrey = '\033[37m'
    darkgrey = '\033[90m'
    lightred = '\033[91m'
    lightgreen = '\033[92m'
    yellow = '\033[93m'
    lightblue = '\033[94m'
    pink = '\033[95m'
    lightcyan = '\033[96m'

class bg:
    black = '\033[40m'
    red = '\033[41m'
    green = '\033[42m'
    orange = '\033[43m'
    blue = '\033[44m'
    purple = '\033[45m'
    cyan = '\033[46m'
    lightgrey = '\033[47m'

class GPT:
    def __init__(self, SYSTEM_PROMPT='Answer concisely', BASE_CHAIN=None, MODEL='gpt-4o', TEMP=0.1, FREQ_PENALTY=0,
                 PRES_PENALTY=0, ):
        if BASE_CHAIN is None: BASE_CHAIN = []
        self.system_prompt = SYSTEM_PROMPT
        BASE_CHAIN = [BASE_CHAIN] if isinstance(BASE_CHAIN, dict) else BASE_CHAIN
        self.base_chain = BASE_CHAIN
        self.cur_chain = BASE_CHAIN

        self.config = {
            'model': MODEL,
            'temp': TEMP,
            'freq_pen': FREQ_PENALTY,
            'pres_pen': PRES_PENALTY,
        }

        self.latest_output_text = None
        self.latest_output_JSON = None

    @staticmethod
    def printChain(INPUT_CHAIN):
        print(f"{fg.black+bg.lightgrey}\n\nPRINTING NEW CHAIN\n\n")
        for i in INPUT_CHAIN: GPT.printSingleMessage(i)
        tokens = GPT.countTokens(INPUT_CHAIN, VERBOSE=True)
        GPT.tokenCost(tokens, IS_INPUT=True, VERBOSE=True)

    @staticmethod
    def printSingleMessage(INPUT_MESSAGE, IS_INPUT=True, SPAN=span):
        if not IS_INPUT:
            print(f"{fg.black+bg.blue}OUTPUT:\n{INPUT_MESSAGE}\n{st.reset}\n")
            return
        elif INPUT_MESSAGE['role'] == 'system':
            color = fg.black+bg.red
        elif INPUT_MESSAGE['role'] == 'user':
            color = fg.black+bg.green
        else:
            color = fg.black+bg.purple
        print(f"{color}{INPUT_MESSAGE['role'].upper()}:\n{st.reverse}{INPUT_MESSAGE['content']}{st.reset}\n")

    @staticmethod
    def countTokens(INPUT, MODEL='gpt-4o', VERBOSE=False):
        input_combined = '\n'.join([x['role'].upper() + ': ' + x['content'] for x in INPUT]) if isinstance(INPUT, (
            dict, list)) else INPUT
        encoder = tiktoken.encoding_for_model(MODEL)
        encoding = encoder.encode(input_combined)
        tokens = len(encoding)
        if VERBOSE: print(f'{fg.lightgrey+bg.black}MODEL: {MODEL}\nTOKEN COUNT: {tokens}{st.reset}')
        return tokens

    @staticmethod
    def tokenCost(TOKENCOUNT, IS_INPUT=True, INPUT_COST_PER_MILLION=5, VERBOSE=False):

        cost = TOKENCOUNT * INPUT_COST_PER_MILLION / 1000000 if IS_INPUT else TOKENCOUNT * INPUT_COST_PER_MILLION / 1000000
        if VERBOSE: print(f'{fg.lightgrey+bg.black}COST: {cost}{st.reset}')
        return cost

    def run(self, OUTPUT_JSON=False, ADD_USER_MESSAGE=None, VERBOSE=True):

        responseFormat = {"type": "json_object"} if OUTPUT_JSON else None

        if ADD_USER_MESSAGE != None: self.cur_chain += [{'role': 'user', 'content': ADD_USER_MESSAGE}]

        chain_to_submit = [{'role': 'system', 'content': self.system_prompt}] + self.cur_chain

        if VERBOSE: GPT.printChain(chain_to_submit)

        completion = client.chat.completions.create(
            messages=chain_to_submit,
            response_format=responseFormat,
            model=self.config['model'],
            temperature=self.config['temp'],
            presence_penalty=self.config['freq_pen'],
            frequency_penalty=self.config['pres_pen'],
        )

        response_text = completion.choices[0].message.content
        self.latest_output_text = response_text
        self.cur_chain.append({'role': 'assistant', 'content': response_text})
        if VERBOSE: GPT.printSingleMessage(response_text, IS_INPUT=False)
        if VERBOSE: print('Total Cost: ' + str(round(
            GPT.tokenCost(GPT.countTokens(chain_to_submit), IS_INPUT=True, VERBOSE=False) + GPT.tokenCost(
                GPT.countTokens(response_text), IS_INPUT=False, VERBOSE=False), 4)))

        if OUTPUT_JSON:
            response_JSON = json.loads(response_text)
            # self.latest_output_text = response_text unneeded cause check a few lines above
            self.latest_output_JSON = response_JSON
            return response_JSON

        return response_text

    def correction(self, OUTPUT_JSON,
                   ERROR_MSG='There was an error with your last message, please rerun it.'):  # Passes error output, but doesn't keep subsequent output (if error) and reruns with error message
        temperature_cache = self.config['temp']
        self.config['temp'] = 1
        self.run(ADD_USER_MESSAGE=ERROR_MSG, OUTPUT_JSON=OUTPUT_JSON)
        self.config['temp'] = temperature_cache
        self.cur_chain = self.cur_chain[:(len(self.cur_chain) - 3)]
        self.cur_chain.append({'role': 'assistant', 'content': self.latest_output_text})

    def rerunLatest(self, OUTPUT_JSON):  # Passes and Keeps error output and reruns without error message
        temperature_cache = self.config['temp']
        self.config['temp'] = 0.8
        del self.cur_chain[-1]
        self.run(OUTPUT_JSON=OUTPUT_JSON)
        self.config['temp'] = temperature_cache

# test = GPT('You generate JSON objects',[{'role': 'user','content': '''Please write a poem.'''},])

# test.run(ADD_USER_MESSAGE='Generate a small random json object',OUTPUT_JSON=True)