"""
JARVIS Voice System - WORKING MALE VOICE
Simple, clean, working version
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal
from dataclasses import dataclass
from typing import List, Dict
import hashlib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==================== SIMPLE CONFIG ====================
@dataclass
class SimpleVoiceConfig:
    """Simple working voice config"""
    f0: float = 120.0  # Pitch
    volume: float = 0.8  # Volume
    name: str = "Male Voice"

# ==================== SIMPLE PHONEMES ====================
class SimplePhonemes:
    """Simple phoneme database"""
    
    # Only essential phonemes
    PHONEMES = {
        # Vowels
        'a': {'type': 'vowel', 'freq': 800, 'duration': 0.2},
        'e': {'type': 'vowel', 'freq': 500, 'duration': 0.2},
        'i': {'type': 'vowel', 'freq': 350, 'duration': 0.2},
        'o': {'type': 'vowel', 'freq': 450, 'duration': 0.2},
        'u': {'type': 'vowel', 'freq': 300, 'duration': 0.2},
        'ə': {'type': 'vowel', 'freq': 500, 'duration': 0.15},
        
        # Consonants
        'h': {'type': 'consonant', 'noise': True, 'duration': 0.1},
        'l': {'type': 'consonant', 'noise': False, 'duration': 0.12},
        'r': {'type': 'consonant', 'noise': False, 'duration': 0.12},
        'w': {'type': 'consonant', 'noise': False, 'duration': 0.1},
        'j': {'type': 'consonant', 'noise': False, 'duration': 0.1},
        'd': {'type': 'consonant', 'noise': True, 'duration': 0.1},
        't': {'type': 'consonant', 'noise': True, 'duration': 0.1},
        's': {'type': 'consonant', 'noise': True, 'duration': 0.15},
        'z': {'type': 'consonant', 'noise': True, 'duration': 0.15},
        'm': {'type': 'consonant', 'noise': False, 'duration': 0.15},
        'n': {'type': 'consonant', 'noise': False, 'duration': 0.15},
        'p': {'type': 'consonant', 'noise': True, 'duration': 0.1},
        'b': {'type': 'consonant', 'noise': True, 'duration': 0.1},
        'k': {'type': 'consonant', 'noise': True, 'duration': 0.12},
        'g': {'type': 'consonant', 'noise': True, 'duration': 0.12},
    }
    
    # Word to phoneme mapping
    WORDS = {
        'hello': ['h', 'ə', 'l', 'o'],
        'world': ['w', 'ə', 'r', 'l', 'd'],
        'jarvis': ['d', 'ʒ', 'ɑ', 'r', 'v', 'ɪ', 's'],
        'i': ['a', 'ɪ'],
        'am': ['æ', 'm'],
        'here': ['h', 'ɪ', 'r'],
        'test': ['t', 'e', 's', 't'],
        'one': ['w', 'ʌ', 'n'],
        'two': ['t', 'u'],
        'three': ['θ', 'r', 'i'],
    }
    
    @classmethod
    def get_phoneme(cls, symbol):
        """Get phoneme data"""
        return cls.PHONEMES.get(symbol)

# ==================== SIMPLE VOICE ENGINE ====================
class SimpleVoiceEngine:
    """Simple working voice engine"""
    
    def __init__(self, sample_rate=22050):
        self.sr = sample_rate
        self.phonemes = SimplePhonemes()
    
    def generate_tone(self, freq, duration, volume=0.5):
        """Generate a simple sine wave"""
        n_samples = int(self.sr * duration)
        if n_samples == 0:
            return np.array([])
        
        t = np.linspace(0, duration, n_samples, False)
        
        # Fundamental + harmonics for richer sound
        sound = np.zeros(n_samples)
        for h in range(1, 4):
            amp = volume / h
            sound += amp * np.sin(2 * np.pi * h * freq * t)
        
        # Apply envelope
        attack = int(0.01 * self.sr)
        release = int(0.02 * self.sr)
        
        if attack < n_samples:
            sound[:attack] *= np.linspace(0, 1, attack)
        if release < n_samples:
            sound[-release:] *= np.linspace(1, 0, release)
        
        return sound
    
    def generate_noise(self, duration, volume=0.3):
        """Generate noise for consonants"""
        n_samples = int(self.sr * duration)
        if n_samples == 0:
            return np.array([])
        
        noise = np.random.randn(n_samples) * volume
        
        # Apply envelope
        attack = int(0.002 * self.sr)
        release = int(0.005 * self.sr)
        
        if attack < n_samples:
            noise[:attack] *= np.linspace(0, 1, attack)
        if release < n_samples:
            noise[-release:] *= np.linspace(1, 0, release)
        
        return noise
    
    def generate_vowel(self, phoneme, duration, config):
        """Generate vowel sound"""
        data = self.phonemes.get_phoneme(phoneme)
        if not data:
            return np.array([])
        
        # Use phoneme frequency or default
        freq = data.get('freq', 500)
        
        # Blend with voice pitch
        sound = self.generate_tone(freq, duration, 0.5)
        
        # Add some pitch variation
        t = np.linspace(0, duration, len(sound))
        vibrato = 1 + 0.01 * np.sin(2 * np.pi * 5 * t)
        sound = sound * vibrato
        
        return sound * config.volume
    
    def generate_consonant(self, phoneme, duration, config):
        """Generate consonant sound"""
        data = self.phonemes.get_phoneme(phoneme)
        if not data:
            return np.array([])
        
        if data.get('noise', False):
            # Noise-based consonant
            sound = self.generate_noise(duration, 0.3)
            
            # Add some tone for voiced consonants
            if phoneme in ['b', 'd', 'g', 'z', 'v']:
                tone = self.generate_tone(150, duration, 0.1)
                if len(tone) == len(sound):
                    sound = sound + tone
        else:
            # Tone-based consonant (l, r, w, y, m, n)
            sound = self.generate_tone(200, duration, 0.3)
        
        return sound * config.volume
    
    def generate_silence(self, duration):
        """Generate silence"""
        n_samples = int(self.sr * duration)
        return np.zeros(n_samples)

# ==================== SIMPLE JARVIS ====================
class SimpleJarvis:
    """Simple working Jarvis voice"""
    
    def __init__(self):
        self.engine = SimpleVoiceEngine()
        self.config = SimpleVoiceConfig()
        self.phonemes = SimplePhonemes()
        
        # Voice presets
        self.presets = {
            'deep': SimpleVoiceConfig(f0=90, volume=0.9, name="Deep Male"),
            'standard': SimpleVoiceConfig(f0=120, volume=0.8, name="Standard Male"),
            'young': SimpleVoiceConfig(f0=150, volume=0.7, name="Young Male"),
        }
        
        self.cache_dir = Path("voice_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def set_voice(self, name):
        """Set voice preset"""
        if name in self.presets:
            self.config = self.presets[name]
            print(f"✓ Voice: {self.config.name} ({self.config.f0} Hz)")
        else:
            print(f"✗ Voice not found")
    
    def set_pitch(self, f0):
        """Set pitch"""
        self.config.f0 = max(60, min(300, f0))
        print(f"✓ Pitch: {self.config.f0} Hz")
    
    def set_volume(self, vol):
        """Set volume"""
        self.config.volume = max(0, min(1, vol))
        print(f"✓ Volume: {self.config.volume}")
    
    def text_to_phonemes(self, text):
        """Convert text to phonemes"""
        text = text.lower().strip()
        words = text.split()
        
        phonemes = []
        for word in words:
            # Clean word
            clean = ''.join(c for c in word if c.isalpha())
            
            # Get phonemes from dictionary
            if clean in self.phonemes.WORDS:
                phonemes.extend(self.phonemes.WORDS[clean])
            else:
                # Fallback - spell it out
                for c in clean:
                    if c in 'aeiou':
                        phonemes.append('ə')
                    else:
                        phonemes.append('t')
            
            # Add pause between words
            phonemes.append('pau')
        
        return phonemes
    
    def generate_speech(self, text, output_file=None):
        """Generate speech"""
        if not text:
            print("No text")
            return None
        
        print(f"\n🎙️ Generating: '{text}'")
        print(f"   Pitch: {self.config.f0} Hz, Volume: {self.config.volume}")
        
        # Get phonemes
        phonemes = self.text_to_phonemes(text)
        print(f"   Phonemes: {phonemes}")
        
        if not phonemes:
            return None
        
        # Check cache
        cache_key = hashlib.md5(f"{text}{self.config.f0}".encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.npy"
        
        if cache_file.exists():
            print("   Loading from cache")
            audio = np.load(cache_file)
        else:
            # Generate each phoneme
            segments = []
            
            for p in phonemes:
                if p == 'pau':
                    seg = self.engine.generate_silence(0.1)
                else:
                    data = self.phonemes.get_phoneme(p)
                    if data:
                        duration = data.get('duration', 0.15)
                        if data['type'] == 'vowel':
                            seg = self.engine.generate_vowel(p, duration, self.config)
                        else:
                            seg = self.engine.generate_consonant(p, duration, self.config)
                    else:
                        continue
                
                if seg is not None and len(seg) > 0:
                    segments.append(seg)
            
            if not segments:
                print("No audio generated")
                return None
            
            # Combine
            audio = np.concatenate(segments)
            
            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * self.config.volume
            
            # Save cache
            np.save(cache_file, audio)
            print(f"   Cached: {cache_key}")
        
        # Play test tone first (to check if sound working)
        test_tone = self.engine.generate_tone(440, 0.1, 0.3)
        sd.play(test_tone, self.engine.sr)
        sd.wait()
        
        # Save output
        if output_file:
            sf.write(output_file, audio, self.engine.sr)
            print(f"   Saved: {output_file}")
        
        return audio
    
    def speak(self, text):
        """Speak text"""
        audio = self.generate_speech(text)
        if audio is not None and len(audio) > 0:
            print(f"   Playing: {len(audio)/self.engine.sr:.2f} seconds")
            sd.play(audio, self.engine.sr)
            sd.wait()
            print("   Done")
    
    def test_sound(self):
        """Test if sound is working"""
        print("\n🔊 Testing sound...")
        
        # Test different frequencies
        for freq in [220, 440, 880]:
            tone = self.engine.generate_tone(freq, 0.2, 0.5)
            sd.play(tone, self.engine.sr)
            sd.wait()
        
        print("✓ Sound test complete")

# ==================== MAIN ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Test sound')
    parser.add_argument('--interactive', '-i', action='store_true')
    parser.add_argument('--voice', '-v', default='standard')
    parser.add_argument('--pitch', '-p', type=float)
    parser.add_argument('--volume', '-vol', type=float)
    parser.add_argument('--text', '-t')
    parser.add_argument('--output', '-o')
    
    args = parser.parse_args()
    
    jarvis = SimpleJarvis()
    
    if args.test:
        jarvis.test_sound()
        exit()
    
    jarvis.set_voice(args.voice)
    
    if args.pitch:
        jarvis.set_pitch(args.pitch)
    
    if args.volume:
        jarvis.set_volume(args.volume)
    
    if args.text:
        jarvis.speak(args.text)
    elif args.interactive:
        print("\n🎤 SIMPLE JARVIS - Working Voice")
        print("Commands: /pitch <hz>, /volume <0-1>, /test, /exit")
        
        while True:
            cmd = input("\n🎬 ").strip()
            
            if cmd == '/exit':
                break
            elif cmd == '/test':
                jarvis.test_sound()
            elif cmd.startswith('/pitch'):
                _, p = cmd.split()
                jarvis.set_pitch(float(p))
            elif cmd.startswith('/volume'):
                _, v = cmd.split()
                jarvis.set_volume(float(v))
            elif cmd:
                jarvis.speak(cmd)
    else:
        # Test with simple phrase
        jarvis.speak("hello world")

