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
OUT_DIR = "out_interview_text"  # Output directory for saving files
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

SYSTEM_PROMPT = """
You are a world-class podcast writer, known for crafting highly focused technical interviews. Your task is to create a realistic and informative interview based directly on a structured list of questions and answers provided in the PDF.

### Interview Structure
- **Stick strictly to the provided Q&A format**: Each question from the PDF should be introduced by Speaker 1 in order, with Speaker 2 delivering the provided answer verbatim or in a natural conversational format.
- **No additional clarification requests or options needed**: Stay focused on each question and answer sequence as outlined.
- **Keep the tone technical and informative**: Avoid overly casual language; ensure that Speaker 1’s questions prompt clear, structured answers from Speaker 2.

### Speaker Roles:
- **Speaker 1**: The interviewer, presenting each question as written in the PDF and occasionally adding very brief affirmations or follow-ups to keep the flow conversational.
- **Speaker 2**: The interviewee, a technical expert, providing each answer with clarity and precision. They may add brief examples to illustrate key points if necessary, but should stay aligned with the provided content.

### Formatting Instructions:
- **Begin directly with Speaker 1** introducing the show and setting up the main topic.
- **Format**: Return your response strictly as a list of tuples.
- **Dialogues Only**: Write only the spoken lines, no titles, and no chapters.


Transform this comprehensive list of Snowflake Q&As into a podcast-style interview format.
Create a dialogue with Speaker 1 (interviewer) and Speaker 2 (Snowflake expert) that covers the technical content systematically.

The interview should maintain a conversational tone while ensuring that the technical details are accurately presented.
Interview format:
1. A full-length interview covering all 50 questions

Provide the entire transcript with all 50 questions answered in this podcast interview format, maintaining the technical depth while keeping it conversational

### Response Format:
[
    ("Speaker 1", "Welcome to our show, where we delve into the technical world of data warehousing. Today, we're focusing on Snowflake, a cloud-based data warehousing platform that's been making waves. Our guest is here to answer some important questions about what Snowflake offers."),
    ("Speaker 2", "Thanks for having me! Excited to talk about it."),
    ("Speaker 1", "Alright, let’s dive in. First question: What is Snowflake and how does it differ from traditional data warehouses?"),
    ("Speaker 2", "Great question. Snowflake is a cloud-based data warehousing platform that separates compute and storage resources, offering scalability, concurrency, and pay-per-use pricing. Unlike traditional data warehouses..."),
    ("Speaker 1", "Interesting! Next up, can you explain the architecture of Snowflake?"),
    ("Speaker 2", "Of course. Snowflake’s architecture has three main layers...")
    ("Speaker 1", "Let's take a break and return with more insights."),
]
Do not finish podcast if you did not answer all 50 questions. Do a podcast break and continue with the remaining questions in the next podcast episode.
Do not return free form text, stick to the podcast interview format as described above.
do not strat or end with free form text like this:
"Here's the podcast-style interview transcript for the Snowflake Q&A"
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
        self.system_message = None
        
    def clear_history(self):
        self.conversation_history = []
        
    def run_stream_response(self, messages, model):
        # Format the system prompt and user message for Anthropic's API
        if not self.system_message:
            self.system_message = next(msg["content"] for msg in messages if msg["role"] == "system")
        user_message = next(msg["content"] for msg in messages if msg["role"] == "user")
        self.conversation_history.append({
                        "role": "user",
                        "content": user_message
                    }
        )
        try:
            # Create the message with Anthropic's API
            response = self.client.messages.create(
                model=model,
                max_tokens=8*1024,
                temperature=1,
                system=self.system_message,
                messages=self.conversation_history,
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
    fn='./source_qna/snowflake.txt'
    INPUT_PROMPT = read_file_to_string(fn)
    if INPUT_PROMPT is None:
        print("Error: Could not read input file")
        return

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INPUT_PROMPT},
    ]

    pipeline = Processor()
    bn, _=splitext(basename(fn))
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")    
    
        
    if 1:
        outputs = pipeline.run_stream_response(messages, MODEL)
           
        out_fn=f'{OUT_DIR}/1_QNA_claude_{timestamp}_{bn}.pkl'

        if outputs and outputs[0]["generated_text"]:
            save_string_pkl = outputs[0]["generated_text"][0]['content']
            with open(out_fn, 'wb') as file:
                pickle.dump(save_string_pkl, file)
            print("\nResponse saved successfully!")
        else:
            print("Error: No output generated")
    if 1: #second batch
        #get usir input
        answer=input("Do you want to continue with the next batch of questions and answers? (yes/no): ")
        if answer.lower() not in ['yes', 'y']:
            return
        messages = [
        
        {"role": "user", "content": 'Yes, provide the next batch of questions and answers if there are any'},
        ]

        outputs = pipeline.run_stream_response(messages, MODEL)
           
        out_fn=f'{OUT_DIR}/2_QNA_claude_{timestamp}_{bn}.pkl'

        if outputs and outputs[0]["generated_text"]:
            save_string_pkl = outputs[0]["generated_text"][0]['content']
            with open(out_fn, 'wb') as file:
                pickle.dump(save_string_pkl, file)
            print("\nResponse saved successfully!")
        else:
            print("Error: No output generated")

if __name__ == "__main__":
    main()