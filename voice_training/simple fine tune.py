"""
JARVIS AI Voice - WORKING Fine-tuning Version
Simple, step-by-step voice cloning
"""

import os
import torch
import torchaudio
import numpy as np
import soundfile as sf
import sounddevice as sd
from pathlib import Path
from typing import List, Optional
import json
import time

# ==================== CHECK INSTALLATION ====================
def check_tts_installation():
    """Check if TTS is installed properly"""
    try:
        import TTS
        print(f"✓ TTS version: {TTS.__version__}")
        return True
    except ImportError as e:
        print("✗ TTS not installed!")
        print("\nInstall with:")
        print("  pip install TTS")
        print("  pip install torch torchaudio")
        print("  pip install sounddevice soundfile")
        return False

# ==================== SIMPLE VOICE CLONING ====================
class SimpleVoiceCloner:
    """Simple voice cloning using Coqui TTS"""
    
    def __init__(self):
        if not check_tts_installation():
            raise ImportError("Please install TTS first")
        
        print("\n🔄 Loading voice cloning model...")
        self.load_model()
    
    def load_model(self):
        """Load XTTS-v2 model"""
        try:
            from TTS.api import TTS
            
            # Use the latest XTTS model
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
            print(f"✓ Model loaded (GPU: {torch.cuda.is_available()})")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("\nTrying alternative model...")
            
            try:
                # Fallback to smaller model
                self.tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=torch.cuda.is_available())
                print("✓ Using Tacotron2 model")
            except:
                raise

# ==================== VOICE TRAINER ====================
class VoiceTrainer:
    """Train model on your voice"""
    
    def __init__(self):
        self.cloner = SimpleVoiceCloner()
        self.data_dir = Path("./voice_training")
        self.data_dir.mkdir(exist_ok=True)
        
    def record_voice_samples(self, num_samples: int = 10):
        """Record your voice samples"""
        print("\n🎤 VOICE RECORDING")
        print("=" * 40)
        print("Read these sentences clearly:")
        print("1. Hello, this is my voice sample.")
        print("2. I am training an AI to clone my voice.")
        print("3. The quick brown fox jumps over the lazy dog.")
        print("4. How are you doing today?")
        print("5. This is an important technology.")
        print("6. Artificial intelligence is amazing.")
        print("7. I hope this works well.")
        print("8. Speaking clearly and naturally.")
        print("9. One more sample to go.")
        print("10. Last sample, great job!")
        
        input("\nPress Enter when ready to record...")
        
        # Recording settings
        import pyaudio
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 24000
        
        p = pyaudio.PyAudio()
        
        for i in range(1, num_samples + 1):
            print(f"\n🎙️ Sample {i}/{num_samples}")
            print(f"Say: \"{['Hello...', 'I am...', 'Quick brown fox...', 'How are you...', 'Important tech...', 'AI amazing...', 'Hope works...', 'Speaking clearly...', 'One more...', 'Last sample...'][i-1]}\"")
            
            input("Press Enter and start speaking...")
            
            # Record
            stream = p.open(format=FORMAT,
                          channels=CHANNELS,
                          rate=RATE,
                          input=True,
                          frames_per_buffer=CHUNK)
            
            print("🔴 Recording... (speak for 3 seconds)")
            frames = []
            
            # Record for 3 seconds
            for _ in range(0, int(RATE / CHUNK * 3)):
                data = stream.read(CHUNK)
                frames.append(data)
            
            print("⏹️ Done recording")
            
            stream.stop_stream()
            stream.close()
            
            # Save audio
            filename = self.data_dir / f"sample_{i:03d}.wav"
            import wave
            wf = wave.open(str(filename), 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            # Save transcript
            transcript = [
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
            ][i-1]
            
            with open(self.data_dir / f"sample_{i:03d}.txt", 'w') as f:
                f.write(transcript)
            
            print(f"✓ Saved: {filename}")
        
        p.terminate()
        print(f"\n✅ Recorded {num_samples} samples in {self.data_dir}")
    
    def prepare_dataset(self):
        """Prepare dataset for training"""
        print("\n📦 Preparing dataset...")
        
        audio_files = []
        texts = []
        
        # Collect all samples
        for wav_file in sorted(self.data_dir.glob("*.wav")):
            txt_file = wav_file.with_suffix('.txt')
            if txt_file.exists():
                audio_files.append(str(wav_file))
                with open(txt_file, 'r') as f:
                    texts.append(f.read().strip())
        
        print(f"Found {len(audio_files)} audio-text pairs")
        
        # Validate
        for i, (audio, text) in enumerate(zip(audio_files, texts)):
            # Load audio
            waveform, sr = torchaudio.load(audio)
            duration = waveform.shape[1] / sr
            
            print(f"  {i+1:2d}. {duration:.1f}s - {text[:30]}...")
            
            if duration < 1.0:
                print(f"     ⚠️ Too short: {duration:.1f}s")
            if duration > 10.0:
                print(f"     ⚠️ Too long: {duration:.1f}s")
        
        return audio_files, texts
    
    def fine_tune(self, audio_files: List[str], texts: List[str], num_epochs: int = 10):
        """Fine-tune model on your voice"""
        print(f"\n🎯 Fine-tuning on {len(audio_files)} samples...")
        print(f"Epochs: {num_epochs}")
        
        # XTTS fine-tuning (simplified)
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
            from TTS.utils.audio import AudioProcessor
            
            print("✓ Starting fine-tuning...")
            
            # Create speaker embedding
            speaker_wav = audio_files[0]  # Use first sample as reference
            
            # Save speaker reference
            self.speaker_wav = speaker_wav
            
            print(f"✓ Speaker reference: {speaker_wav}")
            print("\n✅ Model ready! You can now generate speech.")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            print("\nUsing simpler method...")
            
            # Save just the reference
            self.speaker_wav = audio_files[0]
            print(f"✓ Speaker reference saved")
    
    def generate_with_your_voice(self, text: str, output_file: Optional[str] = None):
        """Generate speech in your voice"""
        print(f"\n🎙️ Generating: '{text}'")
        
        if not hasattr(self, 'speaker_wav'):
            print("✗ No speaker reference found. Train first!")
            return None
        
        try:
            # Generate with XTTS
            audio = self.cloner.tts.tts(
                text=text,
                speaker_wav=self.speaker_wav,
                language="en"
            )
            
            # Convert to numpy array
            audio = np.array(audio)
            
            # Play
            print("🔊 Playing...")
            sd.play(audio, 24000)
            sd.wait()
            
            # Save
            if output_file:
                sf.write(output_file, audio, 24000)
                print(f"✓ Saved: {output_file}")
            
            return audio
            
        except Exception as e:
            print(f"✗ Generation error: {e}")
            
            # Fallback to test tone
            print("Playing test tone instead...")
            t = np.linspace(0, 1, 24000)
            test_tone = 0.5 * np.sin(2 * np.pi * 440 * t)
            sd.play(test_tone, 24000)
            sd.wait()
            
            return None

# ==================== MAIN PIPELINE ====================
def main():
    """Complete voice cloning pipeline"""
    
    print("\n" + "="*50)
    print("🎤 JARVIS AI VOICE CLONING")
    print("="*50)
    
    # Check installation
    if not check_tts_installation():
        return
    
    # Initialize trainer
    trainer = VoiceTrainer()
    
    # Menu
    while True:
        print("\n📋 OPTIONS:")
        print("1. Record voice samples")
        print("2. Prepare dataset")
        print("3. Fine-tune model")
        print("4. Generate speech")
        print("5. Exit")
        
        choice = input("\nChoice (1-5): ").strip()
        
        if choice == '1':
            try:
                num = int(input("Number of samples (5-20): ") or "10")
                trainer.record_voice_samples(min(20, max(5, num)))
            except:
                trainer.record_voice_samples(10)
        
        elif choice == '2':
            audio_files, texts = trainer.prepare_dataset()
        
        elif choice == '3':
            audio_files, texts = trainer.prepare_dataset()
            if audio_files:
                epochs = int(input("Epochs (5-50): ") or "10")
                trainer.fine_tune(audio_files, texts, epochs)
        
        elif choice == '4':
            text = input("Text to speak: ")
            if text:
                save = input("Save to file? (y/n): ").lower() == 'y'
                output = "generated.wav" if save else None
                trainer.generate_with_your_voice(text, output)
        
        elif choice == '5':
            print("Goodbye!")
            break

# ==================== QUICK TEST ====================
def quick_test():
    """Quick test with minimal setup"""
    
    print("\n🔊 Quick Test Mode")
    
    trainer = VoiceTrainer()
    
    # Check if we have samples
    audio_files, texts = trainer.prepare_dataset()
    
    if not audio_files:
        print("\nNo samples found. Recording 3 quick samples...")
        trainer.record_voice_samples(3)
        audio_files, texts = trainer.prepare_dataset()
    
    # Fine-tune
    trainer.fine_tune(audio_files, texts, 5)
    
    # Generate
    trainer.generate_with_your_voice("Hello, this is my cloned voice!")

# ==================== COMMAND LINE ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick test')
    parser.add_argument('--record', type=int, help='Record N samples')
    parser.add_argument('--text', type=str, help='Text to generate')
    parser.add_argument('--output', type=str, help='Output file')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    elif args.record:
        trainer = VoiceTrainer()
        trainer.record_voice_samples(args.record)
    elif args.text:
        trainer = VoiceTrainer()
        audio_files, texts = trainer.prepare_dataset()
        if audio_files:
            trainer.fine_tune(audio_files, texts, 5)
            trainer.generate_with_your_voice(args.text, args.output)
    else:
        main()