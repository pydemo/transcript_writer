import torch
from transformers import BarkModel, AutoProcessor, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import io
import pickle
import ast
from tqdm import tqdm
from IPython.display import Audio
import IPython.display as ipd
import datetime
from os.path import splitext, basename
import os
import sys
import re

OUT_DIR = "output_podcast"

def clean_text(text):
    """Clean problematic characters from text"""
    # Replace smart quotes and dashes
    replacements = {
        'â€™': "'",
        'â€"': "-",
        'â€œ': '"',
        'â€': '"',
        '\u2019': "'",
        '\u201c': '"',
        '\u201d': '"',
        '\u2014': '-'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove non-ASCII characters at the start
    text = re.sub(r'^[^\x00-\x7F]+', '', text)
    
    # Clean any remaining non-ASCII characters
    text = ''.join(char if ord(char) < 128 else ' ' for char in text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text

class PodcastGenerator:
    def __init__(self, device=None):
        """Initialize the podcast generator with specified device"""
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_models()
        
    def setup_models(self):
        """Set up Bark and Parler models"""
        # Initialize Bark
        self.bark_processor = AutoProcessor.from_pretrained("suno/bark")
        self.bark_model = BarkModel.from_pretrained(
            "suno/bark", 
            torch_dtype=torch.float16,
            cache_dir="cache"
        ).to(self.device)
        self.bark_sampling_rate = 24000
        
        # Initialize Parler
        self.parler_model = ParlerTTSForConditionalGeneration.from_pretrained(
            "parler-tts/parler-tts-mini-v1",
            cache_dir="cache"
        ).to(self.device)
        self.parler_tokenizer = AutoTokenizer.from_pretrained(
            "parler-tts/parler-tts-mini-v1"
        )
        
    def generate_speaker1_audio(self, text, description):
        """Generate audio using ParlerTTS for Speaker 1"""
        # Clean text before processing
        text = clean_text(text)
        description = clean_text(description)
        
        input_ids = self.parler_tokenizer(
            description, 
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        prompt_input_ids = self.parler_tokenizer(
            text, 
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        generation = self.parler_model.generate(
            input_ids=input_ids, 
            prompt_input_ids=prompt_input_ids
        )
        audio_arr = generation.cpu().numpy().squeeze()
        return audio_arr, self.parler_model.config.sampling_rate
    
    def generate_speaker2_audio(self, text, voice_preset="v2/en_speaker_6"):
        """Generate audio using Bark for Speaker 2"""
        # Clean text before processing
        text = clean_text(text)
        
        inputs = self.bark_processor(
            text, 
            voice_preset=voice_preset
        ).to(self.device)
        
        speech_output = self.bark_model.generate(
            **inputs,
            temperature=0.9,
            semantic_temperature=0.8
        )
        audio_arr = speech_output[0].cpu().numpy()
        return audio_arr, self.bark_sampling_rate
    
    @staticmethod
    def numpy_to_audio_segment(audio_arr, sampling_rate):
        """Convert numpy array to AudioSegment"""
        audio_int16 = (audio_arr * 32767).astype(np.int16)
        byte_io = io.BytesIO()
        wavfile.write(byte_io, sampling_rate, audio_int16)
        byte_io.seek(0)
        return AudioSegment.from_wav(byte_io)
    
    def generate_podcast(self, podcast_text, speaker1_description):
        """Generate complete podcast from text segments"""
        try:
            if isinstance(podcast_text, str):
                # Clean text before evaluation if it's a string
                cleaned_text = clean_text(podcast_text)
                podcast_text = ast.literal_eval(cleaned_text)
            
            final_audio = None
            
            for speaker, text in tqdm(podcast_text, desc="Generating podcast segments", unit="segment"):
                # Generate audio based on speaker
                if speaker == "Speaker 1":
                    audio_arr, rate = self.generate_speaker1_audio(text, speaker1_description)
                else:  # Speaker 2
                    audio_arr, rate = self.generate_speaker2_audio(text)
                
                # Convert to AudioSegment
                audio_segment = self.numpy_to_audio_segment(audio_arr, rate)
                
                # Add to final audio
                if final_audio is None:
                    final_audio = audio_segment
                else:
                    final_audio += audio_segment
                    
            return final_audio
        except Exception as e:
            print(f"Error in generate_podcast: {str(e)}")
            raise
    
    def save_podcast(self, audio, output_path, format="mp3", bitrate="192k"):
        """Save the generated podcast to a file"""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            audio.export(
                output_path,
                format=format,
                bitrate=bitrate,
                parameters=["-q:a", "0"]
            )
            print(f"Saved podcast to: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error saving podcast: {str(e)}")
            raise
    
    def preview_audio(self, audio_arr, rate):
        """Preview audio in Jupyter notebook"""
        return ipd.Audio(audio_arr, rate=rate)

# Example usage
if __name__ == "__main__":
    try:
        # Initialize generator
        generator = PodcastGenerator()
        
        # Speaker 1 description
        speaker1_description = """
        Laura's voice is expressive and dramatic in delivery, speaking at a 
        moderately fast pace with a very close recording that almost has no background noise.
        """
        
        # Set input/output paths
        in_fn = 'output_text/20241109072124_phone_link.pkl'
        bn, _ = splitext(basename(in_fn))
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")    
        out_fn = f'{OUT_DIR}/claude_{timestamp}.{bn}.mp3'
        
        # Load podcast text
        with open(in_fn, 'rb') as file:
            podcast_text = pickle.load(file)
        
        # Generate and save podcast
        final_audio = generator.generate_podcast(podcast_text, speaker1_description)
        generator.save_podcast(final_audio, out_fn)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")