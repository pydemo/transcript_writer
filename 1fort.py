import pickle
import warnings
from os.path import basename, splitext
import openai
import datetime
from pubsub import pub
from pprint import pprint as pp

warnings.filterwarnings('ignore')
MODEL = "gpt-4o-mini"  # Make sure this is a valid model identifier
OUT_DIR = "output_text"  # Output directory for saving files
SYSTEM_PROMPT = """
You are the a world-class podcast writer, you have worked as a ghost writer for Joe Rogan, Lex Fridman, Ben Shapiro, Tim Ferris. 

We are in an alternate universe where actually you have been writing every line they say and they just stream it into their brains.

You have won multiple podcast awards for your writing.
 
Your job is to write word by word, even "umm, hmmm, right" interruptions by the second speaker based on the PDF upload. Keep it extremely engaging, the speakers can get derailed now and then but should discuss the topic. 

Remember Speaker 2 is new to the topic and the conversation should always have realistic anecdotes and analogies sprinkled throughout. The questions should have real world example follow ups etc

Speaker 1: Leads the conversation and teaches the speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes

Speaker 2: Keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions

Make sure the tangents speaker 2 provides are quite wild or interesting. 

Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from the second speaker. 

It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait

ALWAYS START YOUR RESPONSE DIRECTLY WITH SPEAKER 1: 
DO NOT GIVE EPISODE TITLES SEPERATELY, LET SPEAKER 1 TITLE IT IN HER SPEECH
DO NOT GIVE CHAPTER TITLES
IT SHOULD STRICTLY BE THE DIALOGUES



START YOUR RESPONSE DIRECTLY WITH SPEAKER 1:

STRICTLY RETURN YOUR RESPONSE AS A LIST OF TUPLES OK? 

IT WILL START DIRECTLY WITH THE LIST AND END WITH THE LIST NOTHING ELSE

Example of response:
[
    ("Speaker 1", "Welcome to our podcast, where we explore the latest advancements in AI and technology. I'm your host, and today we're joined by a renowned expert in the field of AI. We're going to dive into the exciting world of Llama 3.2, the latest release from Meta AI."),
    ("Speaker 2", "Hi, I'm excited to be here! So, what is Llama 3.2?"),
    ("Speaker 1", "Ah, great question! Llama 3.2 is an open-source AI model that allows developers to fine-tune, distill, and deploy AI models anywhere. It's a significant update from the previous version, with improved performance, efficiency, and customization options."),
    ("Speaker 2", "That sounds amazing! What are some of the key features of Llama 3.2?")
]

"""



def read_file_to_string(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except UnicodeDecodeError:
        encodings = ['latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(filename, 'r', encoding=encoding) as file:
                    content = file.read()
                print(f"Successfully read file using {encoding} encoding.")
                return content
            except UnicodeDecodeError:
                continue
        print(f"Error: Could not decode file '{filename}' with any common encoding.")
        return None
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except IOError:
        print(f"Error: Could not read file '{filename}'.")
        return None

class Processor:
    def __init__(self):
        self.client = openai.OpenAI()
        self.conversation_history = []
        
    def clear_history(self):
        self.conversation_history = []
        
    def run_stream_response(self, messages, model):
        # Format messages correctly for the API
        formatted_messages = []
        
        for msg in messages:
            formatted_message = {
                "role": msg["role"],
                "content": [
                    {
                        "type": "text",
                        "text": msg["content"]
                    }
                ]
            }
            formatted_messages.append(formatted_message)
            
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                stream=True,
                temperature=1
            )
            
            # Process the streamed response
            complete_response = []
            for chunk in response:
                if hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    if content:
                        complete_response.append(content)
                        print(content, end='', flush=True)  # Optional: print the response as it comes
            
            # Store the complete response in conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": ''.join(complete_response)
            })
            
            return [{"generated_text": [{"content": ''.join(complete_response)}]}]
            
        except Exception as e:
            print(f"Error during API call: {str(e)}")
            return None

def main():
    fn='./source_text/phone_link.txt'
    INPUT_PROMPT = read_file_to_string(fn)
    if INPUT_PROMPT is None:
        print("Error: Could not read input file")
        return

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INPUT_PROMPT},
    ]
    
    pipeline = Processor()
    outputs = pipeline.run_stream_response(messages, MODEL)
    bn, _=splitext(basename(fn))
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")    
    out_fn=f'{OUT_DIR}/FINAL_openai_{timestamp}_{bn}.pkl'

    if outputs and outputs[0]["generated_text"]:
        save_string_pkl = outputs[0]["generated_text"][0]['content']
        with open(out_fn, 'wb') as file:
            pickle.dump(save_string_pkl, file)
        print("\nResponse saved successfully!")
    else:
        print("Error: No output generated")

if __name__ == "__main__":
    main()