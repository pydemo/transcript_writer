import pickle
import warnings
import datetime
from os.path import basename, splitext
from anthropic import Anthropic
from pubsub import pub
from pprint import pprint as pp

warnings.filterwarnings('ignore')
MODEL = "claude-3-5-sonnet-20241022"  # claude-3-5-sonnet-latest
MODEL= 'claude-3-5-haiku-latest' #'claude-3-5-haiku-20241022' #claude-3-5-haiku-latest
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
        self.client = Anthropic()
        self.conversation_history = []
        
    def clear_history(self):
        self.conversation_history = []
        
    def run_stream_response(self, messages, model):
        # Format the system prompt and user message for Anthropic's API
        system_message = next(msg["content"] for msg in messages if msg["role"] == "system")
        user_message = next(msg["content"] for msg in messages if msg["role"] == "user")
        
        try:
            # Create the message with Anthropic's API
            response = self.client.messages.create(
                model=model,
                max_tokens=8*1024,
                temperature=1,
                system=system_message,
                messages=[
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                stream=True
            )
            
            # Process the streamed response
            complete_response = []
            for chunk in response:
                # Check if it's a content event and has content
                if hasattr(chunk, 'type') and chunk.type == 'content_block_delta':
                    if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                        content = chunk.delta.text
                        if content:
                            complete_response.append(content)
                            print(content, end='', flush=True)  # Print the response as it comes
            
            # Join all the response chunks
            full_response = ''.join(complete_response)
            
            # Store the complete response in conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": full_response
            })
            
            return [{"generated_text": [{"content": full_response}]}]
            
        except Exception as e:
            print(f"Error during API call: {str(e)}")
            raise  # Re-raise the exception to see the full traceback
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
    out_fn=f'{OUT_DIR}/claude_{timestamp}_{bn}.pkl'

    if outputs and outputs[0]["generated_text"]:
        save_string_pkl = outputs[0]["generated_text"][0]['content']
        with open(out_fn, 'wb') as file:
            pickle.dump(save_string_pkl, file)
        print("\nResponse saved successfully!")
    else:
        print("Error: No output generated")

if __name__ == "__main__":
    main()