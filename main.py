import json
from transcripts import *
import nltk
import math
from GPT_May_2024 import *



class Transcript:
    format_description = '''# Interview Transcript Information

- Below is information about the transcripts that are the center of analysis in this project 
- Transcripts consist of interviews between 2 people:
    - The interviewer is a researcher
    - The interviewee is an experienced physiotherapist
- Transcripts contain errors
    - In the transcription of the interview's audio recording
    - In the labelling of the speakers
    - There should be enough contextual clues in the transcript to allow you to ignore these errors

**Transcript Format**: Transcripts are organized into three hierarchical levels: Transcript, Speech Turn, and Sentence. Each level has a unique ID for precise referencing.

## Levels of Organization

1. **Transcript**: The entire conversation, identified by a unique Transcript ID (e.g., "A" for transcript 'A').
2. **Speech Turn**: A segment where a single speaker talks continuously, identified by a unique Speech Turn ID (e.g., "A.1.2" for speech turn 1 of transcript 'A' which is spoken by speaker 2).
3. **Sentence**: Individual sentences within a Speech Turn, identified by a unique Sentence ID (e.g., "A.1.2.0" for the first sentence of speech turn 1 spoken by speaker 2 in transcript 'A').

## IDs Explanation

- **Transcript ID**: At the start of every ID; identifies the transcript from which the content originates (e.g., "A").
- **Speech Turn ID**: Combines Transcript ID, turn number, and speaker number (e.g., "A.1.2").
- **Sentence ID**: Extends Speech Turn ID with a sentence number (e.g., "A.1.2.0").
'''
    def __init__(self, parentProject, id, rawInput):
        self.parentProject = parentProject
        self.id = id
        self.rawTranscript = rawInput
        self.turns = {}
        self.sentences = {}

        rawTranscript2Lines = self.rawTranscript.split('\n')

        turnNum = 0
        for i in rawTranscript2Lines:
            if i.lower().startswith('speaker'):
                if turnNum != 0: SpeechTurn(self, turnNum, turnSpeaker, turnContent)
                # reset
                turnNum += 1
                turnContent = ''
                turnSpeaker = i.lower().lstrip('speaker').strip()
                if turnSpeaker == '' or turnSpeaker.isspace(): print('missing speaker')
            else:
                turnContent += i

        SpeechTurn(self, turnNum, turnSpeaker, turnContent)

    def __repr__(self,INCL_INFO=True):
        if INCL_INFO:
            OUT = f'{Transcript.format_description}\n\n# Transcript {self.id} Content\n\nTranscript {self.id} is provided below:\n\n```\n'
        else:
            OUT = f'# Transcript ID: {self.id}\n\n```\n'
        OUT += ''.join([repr(self.turns[i]) for i in self.turns]) + '\n```\n'
        return OUT

    def divide(self,tokens_per_section = 2000):
        num_divisions = math.ceil(GPT.countTokens(self.__repr__()) / tokens_per_section)
        system_prompt = f'# Your Task\n\nThe transcript is too long to be reviewed all at once, your task is to divide it into around {num_divisions} sections that can be provided one at a time to a different reviewer. Each section should not end abruptly or cut off a train of thought.' + '''

## Task Output Format

Your output should be a json list of speech turn IDs where you would like to divide the transcript. A division in the transcript will be created AFTER the speech turns matching the IDs you provide; this means that each section includes the speech turns you identify as the last speech turn in that section.

```
{
    "sections":[
        'M.16.1',  
        'M.34.2', 
        ...
        'M.60.1', 
    ]
}
```

In this example, 'M.16.1' represents the ID of last speech turn in section 1, 'M.34.2' represents the ID of last speech turn in section 2, 'M.60.1' represents the ID of last speech turn in the last section; this should be the last speech turn in the entire transcript
'''

        dividerGPT = GPT(self.parentProject.info + system_prompt)
        dividerGPT.run(ADD_USER_MESSAGE=self.__repr__(INCL_INFO=True), OUTPUT_JSON=True)
        self.section_div_ids = dividerGPT.latest_output_JSON['sections']
        if len(self.section_div_ids) != num_divisions: assert 'Divider Mishap'
        last_turn = list(self.turns)[-1]
        if self.section_div_ids[-1] != last_turn:
            self.section_div_ids.append(last_turn)

        self.sections = []
        self.section_ranges = []
        start_range = True
        cur_section = []
        for i in self.turns:
            if start_range:
                cur_section_range = i + ' to '
                start_range = False

            cur_section.append(self.turns[i])
            if i in self.section_div_ids:
                self.sections.append(cur_section)
                cur_section = []

                cur_section_range += i
                self.section_ranges.append(cur_section_range)
                start_range = True

        print(self.section_div_ids)


    def getSents(self, SELECTION):

        if '-' not in SELECTION:
            if SELECTION not in self.sentences: return f'ERROR: The sentence {SELECTION} was not found in this transcript.'
            return self.sentences[SELECTION]

        selection_split = SELECTION.split('-')
        range_start = selection_split[0].strip()
        range_end = selection_split[1].strip()
        if range_start not in self.sentences: return f'ERROR: The starting sentence ID "{range_start}" of the sentence range "{SELECTION}" was not found in this transcript.'
        if range_end not in self.sentences: return f'ERROR: The ending sentence ID "{range_end}" of the sentence range "{SELECTION}" was not found in this transcript.'

        OUTPUT_range = []
        in_range = False
        for i in self.sentences:
            if i == str(range_start):
                in_range = True
            if in_range:
                OUTPUT_range.append(self.sentences[i])
            if i == str(range_end): # INCLUDES THE RANGE END
                in_range = False

        return OUTPUT_range

class SpeechTurn:
    def __init__(self, parentTranscript, num, speaker, content):
        self.parentTranscript = parentTranscript
        self.num = num
        self.speaker = speaker
        self.content = content

        self.id = f'{self.parentTranscript.id}.{self.num}.{self.speaker}'
        self.sentences = {}

        sentNum = 0
        for sentContent in nltk.sent_tokenize(self.content):
            sentence = Sentence(self, sentNum, sentContent)
            self.sentences[sentence.id] = sentence
            sentNum += 1

        self.parentTranscript.turns[self.id] = self  # add self to transcript parent

    def __repr__(self,LOD='full'):
        OUT = f'### Speech Turn ID: {self.id}, Speaker: {self.speaker}\n'
        if LOD == 'full': OUT += ''.join([repr(self.sentences[i]) for i in self.sentences]) + '\n'
        return OUT

class Sentence:
    def __init__(self, parentTurn, num, content):
        self.parentTurn = parentTurn
        self.num = num
        self.content = content

        self.id = f'{self.parentTurn.id}.{self.num}'

        self.parentTurn.parentTranscript.sentences[self.id] = self  # add self to transcript parent

    def __repr__(self):
        return f'{self.id}: {self.content}\n'

class Project:
    def __init__(self, id, research_question, data):
        self.research_question = research_question
        self.info = f'''# Project Information
- This project is a qualitative thematic analysis of {data}
- This thematic analysis will involve first identifying codes across multiple transcripts, grouping similar codes into categories, and finally grouping one or more categories into themes to ultimately answer the research question.
- The research question of this project is: "{self.research_question}"

'''
        self.id = id
        self.transcripts = {
        }
        self.analysis = {
            "code": {},
            "category": {},
            "theme": {},
        }

    def addLabels(self, INPUT, TYPE):
        for i in INPUT: globals()[TYPE.capitalize()](self, INPUT[i])

    def updateLabels(self, INPUT, TYPE):
        for i in INPUT: self.analysis[TYPE.lower()][i].update(INPUT[i])

    def stringifyLabels(self, TYPE, LOD='', INCL_INFO = True, INITIAL_MSG = None):
        if INCL_INFO: OUT = globals()[TYPE.capitalize()].format_description
        else: OUT = ''

        if INITIAL_MSG != None: OUT += INITIAL_MSG

        OUT += '\n\n```\n' + json.dumps([self.analysis[TYPE][x].__dict__(LOD=LOD) for x in self.analysis[TYPE]],indent=4) + '\n```\n\n'
        return OUT

    def validate_children_ids(self,IDS_TO_CHECK,ID_LIST):
        problem_msg = None

        CHILD_IDS_2_CHECK = []
        for lb in IDS_TO_CHECK: CHILD_IDS_2_CHECK.extend(IDS_TO_CHECK[lb]['children'])
        temp_list =[]
        for id in CHILD_IDS_2_CHECK:
            if '-' in id:
                selection_split = id.split('-')
                temp_list.extend([selection_split[0].strip(),selection_split[1].strip()])
            else: temp_list.append(id)
        CHILD_IDS_2_CHECK = temp_list

        for child_id in CHILD_IDS_2_CHECK:
            if child_id not in ID_LIST:
                if problem_msg == None:
                    problem_msg = f'ERROR: The following ID "{child_id}" was not found.' + '\n'
                else:
                    problem_msg += f'ERROR: The following with ID "{child_id}" was not found.' + '\n'

        return problem_msg
    def validate_repair(self, THINGS_TO_CHECK, VALIDATOR_FUNC, GPT2USE=None, THINGS_REFERENCE=None, CUSTOM_PROBLEM_MSG="Please double check to make sure the IDs you've provided match existing IDs.", RETRY_ATTEMPTS_MAX=3):
        if THINGS_REFERENCE == None: problem_msg = VALIDATOR_FUNC(THINGS_TO_CHECK)
        else: problem_msg = VALIDATOR_FUNC(THINGS_TO_CHECK, THINGS_REFERENCE)

        if problem_msg == None: return THINGS_TO_CHECK

        retry_count = 1

        while problem_msg != None:
            print('Rerun: ', retry_count,'\n',problem_msg)
            if retry_count > RETRY_ATTEMPTS_MAX: raise SystemExit(f'{RETRY_ATTEMPTS_MAX} Retries and codes are still broken')

            problem_msg += CUSTOM_PROBLEM_MSG
            GPT2USE.correction(OUTPUT_JSON=True, ERROR_MSG=problem_msg, )
            # GPT2USE.rerunLatest(OUTPUT_JSON=True)
            if THINGS_REFERENCE == None: problem_msg = VALIDATOR_FUNC(GPT2USE.latest_output_JSON)
            else: problem_msg = VALIDATOR_FUNC(GPT2USE.latest_output_JSON, THINGS_REFERENCE)

            retry_count += 1

        return GPT2USE.latest_output_JSON
        # shouldn't need a return, the while loop either fixes the output or stops the program

    def codeTranscript(self, first_transcript=False, transcript_id2code=None):
        transcript2code = self.transcripts[transcript_id2code]
        transcript2code.divide(tokens_per_section=4000)

        coding_task = f'# Your Task: Transcript Coding\n\nThematic codes should be extracted from the provided transcript. These codes should be relevant to the research question. You have been provided the entirety of transcript {transcript2code.id}, however, I will ask you to search for codes within one section of the transcript at a time rather than across the entire transcript all at once.\n\n'
        full_system_prompt = self.info + transcript2code.__repr__(INCL_INFO=True) + coding_task

        trunc_chain = []
        first_section = True
        counter = 0

        for cur_section_range in transcript2code.section_ranges:

            if first_section: base_chain = []
            else: base_chain = [x for x in trunc_chain]

            codes_LOD = 'no_children'
            if first_transcript and first_section: prev_codes = ''
            elif first_transcript and not first_section: base_chain.append({'role': 'user', 'content': self.stringifyLabels('code',codes_LOD,True, "### Codes So Far\nHere are all the codes we've generated or updated since this conversation began.")})
            elif not first_transcript and first_section: base_chain.append({'role': 'user', 'content': self.stringifyLabels('code', codes_LOD, True, "### Codes So Far\n\nWe have already finished coding some other transcripts in this project. The codes we've found from those transcripts are listed below:")})
            else:  base_chain.append({'role': 'user', 'content': self.stringifyLabels('code', codes_LOD, True, "### Codes So Far\n\nHere are all the codes we've generated in this project so far. This includes codes from the previous transcripts we've coded, as well as the codes we've generated or updated since this conversation began.")})

            code_characteristics = 'In general, any codes you generate should be 1. Unique 2. Clear 3. Relevant to the Research Question 4. Not too broad.'
            if first_transcript and first_section:
                look4first_codes = f"### Coding by Section\n\nWe are going to begin coding the first section: {cur_section_range}.\n\nThis is the first transcript we will code in this project. Remember, codes should be relevant to this project's research question: \"{self.research_question}\".\n\n{code_characteristics}\n\n"

                base_chain.append({'role': 'user', 'content': look4first_codes + Code.first_new_codes_output_format})
                trunc_chain.append({'role': 'user', 'content': look4first_codes})

                coderGPT = GPT(full_system_prompt, BASE_CHAIN=base_chain, )
                coderGPT.run(OUTPUT_JSON=True)
                self.validate_repair(
                    THINGS_TO_CHECK=coderGPT.latest_output_JSON,
                    VALIDATOR_FUNC=self.validate_children_ids,
                    GPT2USE=coderGPT,
                    THINGS_REFERENCE=transcript2code.sentences,
                    CUSTOM_PROBLEM_MSG="Please double check to make sure the sentence IDs you've provided actually match existing sentences in the transcript."
                )
                trunc_chain.append({'role': 'assistant', 'content': coderGPT.latest_output_text})
                self.addLabels(coderGPT.latest_output_JSON,'code')

            else:

                look4updates = f'### Coding by Section\n\nWe are now going to focus on this section of the transcript: {cur_section_range}.\n\nFirst, identify any sentences/excerpts in this section that match the codes we have found previously. Then update those previous codes with the corresponding excerpts of this section and update any of the name, description, or context fields of those codes if needed to incorporate these new excerpts.\n\n{Code.update_codes_output_format}'

                base_chain.append({'role': 'user', 'content': look4updates})

                coderGPT = GPT(full_system_prompt, BASE_CHAIN=base_chain, )  # MODEL='gpt-3.5-turbo' CHANGE MODEL BACK TO GPT-4
                coderGPT.run(OUTPUT_JSON=True, VERBOSE=False)
                self.validate_repair(
                    THINGS_TO_CHECK=coderGPT.latest_output_JSON,
                    VALIDATOR_FUNC=self.validate_children_ids,
                    GPT2USE=coderGPT,
                    THINGS_REFERENCE=transcript2code.sentences,
                    CUSTOM_PROBLEM_MSG="Please double check to make sure the sentence IDs you've provided actually match existing sentences in the transcript."
                )
                self.updateLabels(coderGPT.latest_output_JSON,'code')

                look4new_codes = f'### Coding by Section\n\nFocusing on the same section of the transcript: {cur_section_range}.\n\nIdentify any new codes in this section that can\'t be captured by any of the codes we have found previously. Remember, any codes you find should be relevant to this project\'s research question: \"{self.research_question}\".\n\n{code_characteristics}\n\n'
                trunc_chain.append({'role': 'user', 'content': look4new_codes})

                coderGPT.printChain(trunc_chain)

                coderGPT.run(ADD_USER_MESSAGE=look4new_codes + Code.first_new_codes_output_format, OUTPUT_JSON=True, VERBOSE=False)
                self.validate_repair(
                    THINGS_TO_CHECK=coderGPT.latest_output_JSON,
                    VALIDATOR_FUNC=self.validate_children_ids,
                    GPT2USE=coderGPT,
                    THINGS_REFERENCE=transcript2code.sentences,
                    CUSTOM_PROBLEM_MSG="Please double check to make sure the sentence IDs you've provided actually match existing sentences in the transcript."
                )
                trunc_chain.append({'role': 'assistant', 'content': coderGPT.latest_output_text})
                self.addLabels(coderGPT.latest_output_JSON,'code')

                coderGPT.printChain(trunc_chain)

            first_section = False
            counter += 1

    def createAbstraction(self, TYPE, TYPE_PLURAL, SUB_TYPE, SUB_TYPE_PLURAL, num_type_per_cycle=3, approx_sub_type_per_type=6):

        if TYPE not in self.analysis: self.analysis[TYPE] = {}

        TO_GROUP = self.analysis[SUB_TYPE]

        sub_type_count = len(TO_GROUP) + 1

        num_cycles = math.ceil(sub_type_count / (num_type_per_cycle * approx_sub_type_per_type))

        previously_generated_codes_info = f'''# Previously Generated {SUB_TYPE_PLURAL}

We have previously generated {sub_type_count} thematic {SUB_TYPE_PLURAL} from a collection of transcripts. These {SUB_TYPE_PLURAL} were selected based on their relevance to the project's research question. 

The previously generated {SUB_TYPE_PLURAL} are provided below:
'''
        abstracting_task = f'''# Your Task: Grouping thematic {SUB_TYPE_PLURAL} into {TYPE_PLURAL}

Group these {SUB_TYPE_PLURAL} we previously generated into thematic {TYPE_PLURAL}. Don't generate the {TYPE_PLURAL} all at once. Rather, I will ask you to generate a few {TYPE_PLURAL} at a time.

'''
        full_system_prompt = self.info +  f"\n\n{self.stringifyLabels(TYPE=SUB_TYPE,LOD='no_children',INCL_INFO = True,INITIAL_MSG = previously_generated_codes_info)}\n\n" + abstracting_task + Label.format_description(TYPE, TYPE_PLURAL, SUB_TYPE, SUB_TYPE_PLURAL)

        trunc_chain = []
        for cycle in range(num_cycles):
            if trunc_chain == []:
                base_chain = []
            else:
                base_chain += trunc_chain

            look4categories = f"### Round {cycle + 1} of Searching for Categories\n\nPlease look for similar codes and group them into categories. Limit the number of categories you generate during this round to: {num_type_per_cycle} categories. Remember, categories should be relevant to this project's research question: \"{self.research_question}\".\n\n"
            base_chain.append({'role': 'user', 'content': look4categories + Label.generate_output_format(TYPE, TYPE_PLURAL, SUB_TYPE, SUB_TYPE_PLURAL)})
            trunc_chain.append({'role': 'user', 'content': look4categories})

            abstractorGPT = GPT(full_system_prompt, BASE_CHAIN=base_chain, )
            abstractorGPT.run(OUTPUT_JSON=True)

            self.validate_repair(
                THINGS_TO_CHECK=abstractorGPT.latest_output_JSON,
                VALIDATOR_FUNC=self.validate_children_ids,
                GPT2USE=abstractorGPT,
                THINGS_REFERENCE=TO_GROUP,
                CUSTOM_PROBLEM_MSG="Please double check to make sure the code IDs you've provided actually match existing codes."
            )
            trunc_chain.append({'role': 'assistant', 'content': abstractorGPT.latest_output_text})
            self.addLabels(abstractorGPT.latest_output_JSON, TYPE)

            print("Generate categories. Cycle:", cycle)

class Label:
    def __init__(self, parentProject, INPUT):
        self.parentProject = parentProject
        self.id = INPUT['id']
        self.name = INPUT['name']
        self.description = INPUT['description']
        self.context = INPUT['context']
        self.children = {}

        self.add_children(INPUT['children'])

        self.parentProject.analysis[self.type][self.id] = self

    def dictify_children(self, LOD='full'): return [self.children[x].__dict__(LOD=LOD) for x in self.children]
    def __dict__(self, LOD='full'):

        if LOD == 'full' : children_dictified = self.dictify_children(LOD='full')

        if LOD == 'subtype_children_ids' : children_dictified = self.dictify_children(LOD='children_ids')

        if LOD == 'children_ids' : children_dictified = self.dictify_children(LOD='id')

        if 'children_dictified' in locals():
            OUTPUT = {
                'id': self.id,
                'name': self.name,
                'description': self.description,
                'context': self.context,
                'children': children_dictified
            }

        ## Overwrites the above output if  LOD is the following

        if LOD == 'no_children':
            OUTPUT = {
                'id': self.id,
                'name': self.name,
                'description': self.description,
                'context': self.context,
            }

        if LOD == 'name_id':
            OUTPUT = {
                'id': self.id,
                'name': self.name,
            }

        if LOD == 'id': OUTPUT = self.id

        return OUTPUT
    def add_children(self, child_list):
        for child_id in child_list:
            if self.type == 'code':
                self.children[child_id] = self.parentProject.transcripts[child_id[0]].getSents(child_id)
            else:
                self.children[child_id] = self.parentProject.analysis[self.sub_type][child_id]

    def update(self, INPUT):
        if 'name' in INPUT: self.name = INPUT['name']
        if 'description' in INPUT: self.description = INPUT['description']
        if 'context' in INPUT: self.context = INPUT['context']
        if 'children' in INPUT: self.add_children(INPUT['children'])
        else: print('No excerpts')

    @staticmethod
    def format_description(TYPE, TYPE_PLURAL, SUB_TYPE, SUB_TYPE_PLURAL):
        return f'''### Format of a {TYPE}

Below is the format of a single {TYPE}:

```json
{{
    "id": "", # A short, unique identifier for the {TYPE}. Make sure it does not conflict with existing {TYPE} ids.
    "name": "", # A short, but descriptive name for the {TYPE}. It should only be a few words long.
    "description": "", # A detailed description of the {TYPE}. This should be detailed enough to understand without additional context (aside from the research question)
    "context": "", # Provides additional context to explain why the {SUB_TYPE_PLURAL} below are encapsulated by this {TYPE}. 
    "children": [
        "ID1", # Includes {SUB_TYPE} ID1 within this {TYPE}
        "ID2" # Includes {SUB_TYPE} ID2 within of this {TYPE}
    ]
}}
```
'''

    @staticmethod
    def generate_output_format(TYPE, TYPE_PLURAL, SUB_TYPE, SUB_TYPE_PLURAL):
        return f'''### Output Format

Output category in the following format as a list:

```json
{{
    "id":{{ # {TYPE} 1
        "id": "", # A short, unique identifier for the {TYPE}. Make sure it does not conflict with existing {TYPE} ids.
        "name": "", # A short, but descriptive name for the {TYPE}. It should only be a few words long.
        "description": "", # A detailed description of the {TYPE}. This should be detailed enough to understand without additional context (aside from the research question)
        "context": "", # Provides additional context to explain why the {SUB_TYPE_PLURAL} below are encapsulated by this {TYPE}. 
        "children": [
            "ID1", # Includes {SUB_TYPE} ID1 within this {TYPE}
            "ID2", # Includes {SUB_TYPE} ID2 within of this {TYPE}
            "ID3", # Includes {SUB_TYPE} ID2 within of this {TYPE}
            "ID4", # Includes {SUB_TYPE} ID4 within of this {TYPE}
        ]
    }},
    "id":{{ # {TYPE} 2
        "id": "", 
        "name": "", 
        "description": "", 
        "context": "", 
        "children": [
            "", 
        ]
    }},
    # etc.
}}
```
'''

    @staticmethod
    def update_output_format(TYPE, TYPE_PLURAL, SUB_TYPE, SUB_TYPE_PLURAL):
        return f'''### Output Format

Output {TYPE} updates in the following format as a list (notice, not every field in the {TYPE} needs to be updated). 

```json
{{
    "abc":{{ # updates the {TYPE} with id "abc"
        "name": "", # updates the name of this {TYPE}
        "description": "", # updates the description of this {TYPE}
        "context": "", # updates the context of this {TYPE}
        "children": [
            "ID1", # Adds {SUB_TYPE} ID1 to this {TYPE}
            "ID2", # Adds {SUB_TYPE} ID2 to this {TYPE}
            "ID3", # Adds {SUB_TYPE} ID2 to this {TYPE}
            "ID4", # Adds {SUB_TYPE} ID4 to this {TYPE}
        ]
    }},
    "xyz":{{ # updates the {TYPE} with id "xyz"
        "context": "", # updates the context of this {TYPE}
        "children": [
            "ID7", # Adds {SUB_TYPE} ID7 to this {TYPE}
        ]
    }},
    # etc.
}}
```
'''

class Code(Label):
    format_description = '''### Format of a code

Below is the format of a single code:

```json
{
    "id": "", # A short, unique identifier for the code. Make sure it does not conflict with existing code ids.
    "name": "", # A short, but descriptive name for the code. It should only be a few words long.
    "description": "", # A detailed description of the code. This should be detailed enough to understand without additional context (aside from the research question)
    "context": "", # Provides additional context to explain how the sentences/excerpts below connect to the code. 
    "children": [
        "A.1.1.2", # You can highlight a single sentence. For instance, this particular list item refers to the sentence: A.1.1.2
        "A.0.1.0-A.3.1.3" # You can also highlight a range of sentences in the transcript. For instance, this particular list items refers to all sentences between A.0.1.0 up to (and including) A.3.1.3 
    ]
}
```

Below is an example code from an entirely different project:

```
{
    "id": "TCI",  
    "name": "Teacher-Student Interactions", 
    "description": "This code refers to the theme of teacher-student interactions in the classroom. It includes observations and statements related to communication, engagement, discipline, and feedback between teachers and students. This can encompass both positive and negative interactions, as well as the overall impact on the classroom environment.",  
    "context": "These excerpts highlight different instances where teachers interact with students. They are aimed at understanding the dynamics of these interactions and their effects on learning and classroom atmosphere.", 
    "children": [
        "Z.15.1.2", # "Remember to check your work before submitting,"
        "Z.24.1.0-Z.28.2.3", # "Let's review yesterday's homework." "Can you share your answer, John?" "That's a good effort, but you missed an important part here." "For tomorrow, please complete the next set of problems."
        "Z.30.1.0", # "Please find a partner and get started on the worksheet."
    ]
}
```

'''
    first_new_codes_output_format = '''### Output Format

Output codes in the following format as a list:

```json
{
    "id":{ # code 1. 
        "id", # The id should be a short, unique identifier for the code. Make sure it does not conflict with existing code ids.
        "name": "", # A short, but descriptive name for the code. It should only be a few words long.
        "description": "", # A detailed description of the theme. This should be detailed enough to understand without additional context (aside from the research question)
        "context": "", # Provides additional context to explain how the sentences/excerpts below connect to the code. 
        "children": [
            "A.1.1.2", # You can highlight a single sentence. For instance, this particular list item refers to the sentence: A.1.1.2. Output only the sentence IDs not the sentence themselves. 
            "A.0.1.0-A.3.1.3" # You can also highlight a range of sentences in the transcript. For instance, this particular list items refers to all sentences between A.0.1.0 up to (and including) A.3.1.3. Output only the sentence IDs not the sentence themselves. 
        ]
    },
    "id":{ # code 2
        "id": "", 
        "name": "", 
        "description": "", 
        "context": "", 
        "children": [
            "", 
        ]
    },
    # etc.
}
```
'''
    update_codes_output_format = '''### Output Format

Output code updates in the following format as a list (notice, not every field in the code needs to be updated). Output only the sentence IDs not the sentence themselves:

```json
{
    "abc":{ # updates the code with id "abc"
        "name": "", # updates the name of this code
        "description": "", # updates the description of this code
        "context": "", # updates the context of this code
        "children": [
            "N.35.1.2", # adds the sentence N.35.1.2 to the excerpts included in this code. Output only the sentence IDs not the sentence themselves. 
            "N.37.1.0-N.39.1.3" # adds the range of sentences from N.37.1.0 up to and including N.39.1.3 to the excerpts included in this code. Output only the sentence IDs not the sentence themselves. 
        ]
    },
    "xyz":{ # updates the code with id "xyz"
        "context": "", # updates the context of this code
        "children": [
            "N.16.1.2", # adds the sentence N.16.1.2 to the excerpts included in this code. Output only the sentence IDs not the sentence themselves. 
        ]
    },
    # etc.
}
```
'''
    def __init__(self,parentProject, INPUT):
        self.type = 'code'
        self.sub_type = 'excerpt'
        super().__init__(parentProject, INPUT)

    def dictify_children(self, LOD):
        if LOD == 'full': # more of a stringify than a listify
            children2output = []
            for id in self.children:
                if isinstance(self.children[id], Sentence):
                    children2output.append(str(self.children[id]))
                elif isinstance(self.children[id], list):
                    combined = '\n'.join([str(x) for x in self.children[id]])
                    children2output.append(combined)

        if LOD == 'id': children2output = [x for x in self.children]

        return children2output

class Cateogry(Label):
    def __init__(self, parentProject, INPUT):
        self.type = 'category'
        self.sub_type = 'code'
        super().__init__(parentProject, INPUT)

class Subtheme(Label):
    def __init__(self, parentProject, INPUT):
        self.type = 'subtheme'
        self.sub_type = 'category'
        super().__init__(parentProject, INPUT)

class Theme(Label):
    def __init__(self, parentProject, INPUT):
        self.type = 'theme'
        self.sub_type = 'subtheme'
        super().__init__(parentProject, INPUT)