"""
JARVIS Voice Cloning - WITHOUT PyAudio
Sounddevice se recording
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
import time

class SimpleRecorder:
    """Record without PyAudio - using sounddevice"""
    
    def __init__(self):
        self.sample_rate = 24000
        self.data_dir = Path("./voice_training")
        self.data_dir.mkdir(exist_ok=True)
    
    def record_sample(self, duration=3):
        """Record audio sample"""
        print(f"\n🎤 Recording for {duration} seconds...")
        
        # Record
        recording = sd.rec(int(duration * self.sample_rate), 
                          samplerate=self.sample_rate, 
                          channels=1, 
                          dtype='float32')
        
        # Show progress
        for i in range(duration):
            print(f"  {duration-i} seconds left...")
            time.sleep(1)
        
        sd.wait()
        print("✓ Recording complete")
        
        return recording.flatten()
    
    def record_samples(self, num_samples=10):
        """Record multiple samples"""
        print("\n" + "="*50)
        print("🎤 VOICE RECORDING SESSION")
        print("="*50)
        
        sentences = [
            "Hello, this is my voice sample.",
            "I am training an AI to clone my voice.",
            "The quick brown fox jumps over the lazy dog.",
            "How are you doing today?",
            "This is an important technology.",
            "Artificial intelligence is amazing.",
            "I hope this works well.",
            "Speaking clearly and naturally.",
            "One more sample to go.",
            "Last sample, great job!"
        ]
        
        for i in range(min(num_samples, len(sentences))):
            print(f"\n📝 Sample {i+1}/{num_samples}")
            print(f"Say: \"{sentences[i]}\"")
            input("Press Enter when ready...")
            
            # Record
            audio = self.record_sample(3)
            
            # Save
            filename = self.data_dir / f"sample_{i+1:03d}.wav"
            sf.write(filename, audio, self.sample_rate)
            
            # Save transcript
            with open(self.data_dir / f"sample_{i+1:03d}.txt", 'w') as f:
                f.write(sentences[i])
            
            print(f"✓ Saved: {filename}")
        
        print(f"\n✅ Recorded {num_samples} samples in {self.data_dir}")

# ==================== COMPLETE WORKING CODE ====================
import torch
import torchaudio
from TTS.api import TTS

class VoiceCloner:
    """Complete voice cloning without PyAudio"""
    
    def __init__(self):
        self.recorder = SimpleRecorder()
        self.model = None
        self.speaker_wav = None
        
    def load_model(self):
        """Load TTS model"""
        print("\n🔄 Loading TTS model...")
        try:
            self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", 
                            gpu=torch.cuda.is_available())
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("Trying smaller model...")
            self.model = TTS("tts_models/en/ljspeech/tacotron2-DDC")
    
    def prepare_training_data(self):
        """Prepare recorded samples"""
        audio_files = []
        texts = []
        
        for wav_file in sorted(self.recorder.data_dir.glob("*.wav")):
            txt_file = wav_file.with_suffix('.txt')
            if txt_file.exists():
                audio_files.append(str(wav_file))
                with open(txt_file, 'r') as f:
                    texts.append(f.read().strip())
        
        return audio_files, texts
    
    def train(self):
        """Train on your voice"""
        print("\n🎯 Training on your voice...")
        
        # Get samples
        audio_files, texts = self.prepare_training_data()
        
        if not audio_files:
            print("No samples found. Recording now...")
            self.recorder.record_samples(5)
            audio_files, texts = self.prepare_training_data()
        
        # Use first sample as reference
        self.speaker_wav = audio_files[0]
        print(f"✓ Speaker reference: {self.speaker_wav}")
        
        # For XTTS, we just need the reference
        print("✅ Model ready! You can now generate speech.")
    
    def speak(self, text, output_file=None):
        """Generate speech in your voice"""
        if not self.model:
            self.load_model()
        
        if not self.speaker_wav:
            print("No speaker reference. Training first...")
            self.train()
        
        print(f"\n🎙️ Generating: '{text}'")
        
        try:
            # Generate speech
            audio = self.model.tts(
                text=text,
                speaker_wav=self.speaker_wav,
                language="en"
            )
            
            audio = np.array(audio)
            
            # Play
            print("🔊 Playing...")
            sd.play(audio, 24000)
            sd.wait()
            
            # Save
            if output_file:
                sf.write(output_file, audio, 24000)
                print(f"✓ Saved: {output_file}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    def interactive_mode(self):
        """Interactive console"""
        print("\n" + "="*50)
        print("🎤 JARVIS VOICE CLONING")
        print("="*50)
        
        while True:
            print("\n📋 Commands:")
            print("  record - Record voice samples")
            print("  train  - Train on your voice")
            print("  speak  - Generate speech")
            print("  exit   - Exit")
            
            cmd = input("\n> ").strip().lower()
            
            if cmd == 'exit':
                break
            elif cmd == 'record':
                num = int(input("Number of samples (5-20): ") or "10")
                self.recorder.record_samples(num)
            elif cmd == 'train':
                self.train()
            elif cmd == 'speak':
                text = input("Text to speak: ")
                if text:
                    self.speak(text)
            else:
                print("Unknown command")

# ==================== MAIN ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', type=int, help='Record N samples')
    parser.add_argument('--text', type=str, help='Text to speak')
    parser.add_argument('--output', type=str, help='Output file')
    
    args = parser.parse_args()
    
    cloner = VoiceCloner()
    
    if args.record:
        cloner.recorder.record_samples(args.record)
    elif args.text:
        cloner.load_model()
        # Check if we have samples
        audio_files, _ = cloner.prepare_training_data()
        if audio_files:
            cloner.speaker_wav = audio_files[0]
            cloner.speak(args.text, args.output)
        else:
            print("No voice samples found. Please record first:")
            cloner.recorder.record_samples(5)
            cloner.speak(args.text, args.output)
    else:
        cloner.interactive_mode()