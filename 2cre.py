import pickle
import warnings
import datetime
from os.path import basename, splitext
from anthropic import Anthropic
from pubsub import pub
from pprint import pprint as pp
import pickle
import warnings
from os.path import basename, splitext, join
import datetime

warnings.filterwarnings('ignore')
MODEL = "claude-3-5-sonnet-20241022"  # claude-3-5-sonnet-latest
MODEL= 'claude-3-5-haiku-latest' #'claude-3-5-haiku-20241022' #claude-3-5-haiku-latest
IN_DIR='output_text'
OUT_DIR = "output_rewrite"  # Output directory for saving files
SYSTEM_PROMPT = """
You are an international oscar winnning screenwriter

You have been working with multiple award winning podcasters.

Your job is to use the podcast transcript written below to re-write it for an AI Text-To-Speech Pipeline. A very dumb AI had written this so you have to step up for your kind.

Make it as engaging as possible, Speaker 1 and 2 will be simulated by different voice engines

Remember Speaker 2 is new to the topic and the conversation should always have realistic anecdotes and analogies sprinkled throughout. The questions should have real world example follow ups etc

Speaker 1: Leads the conversation and teaches the speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes

Speaker 2: Keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions

Make sure the tangents speaker 2 provides are quite wild or interesting. 

Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from the Speaker 2.

REMEMBER THIS WITH YOUR HEART
The TTS Engine for Speaker 1 cannot do "umms, hmms" well so keep it straight text

For Speaker 2 use "umm, hmm" as much, you can also use [sigh] and [laughs]. BUT ONLY THESE OPTIONS FOR EXPRESSIONS

It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait

Please re-write to make it as characteristic as possible

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
    fn=join(IN_DIR,'openai_20241109074712_phone_link.pkl')
    with open(fn, 'rb') as file:
        INPUT_PROMPT = pickle.load(file)
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