"""
JARVIS Voice System - CLEAN VERSION (No Random Sounds)
Sirf defined words ke liye voice generate karega
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import hashlib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==================== ENUMS ====================
class Emotion(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    CONFIDENT = "confident"
    FRIENDLY = "friendly"

class VoiceQuality(Enum):
    WARM = "warm"
    BRIGHT = "bright"
    SMOOTH = "smooth"

# ==================== VOICE CONFIG ====================
@dataclass
class VoiceConfig:
    name: str = "JARVIS"
    gender: str = "male"
    base_f0: float = 115.0
    formants: List[float] = None
    breathiness: float = 0.06
    jitter: float = 0.008
    vibrato_rate: float = 5.0
    vibrato_depth: float = 0.015
    emotion: Emotion = Emotion.CONFIDENT
    quality: VoiceQuality = VoiceQuality.SMOOTH
    
    def __post_init__(self):
        if self.formants is None:
            self.formants = [520, 1120, 2520, 3220, 3870]

# ==================== PHONEME DATABASE (Only Defined) ====================
class PhonemeDB:
    """Sirf defined phonemes - koi random nahi"""
    
    # Sirf yeh words kaam karenge
    ALLOWED_WORDS = {
        # Greetings
        "hello": ["hh", "ah", "l", "ow"],
        "hi": ["hh", "ay"],
        "hey": ["hh", "ey"],
        "good": ["g", "uh", "d"],
        "morning": ["m", "ao", "r", "n", "ih", "ng"],
        "evening": ["iy", "v", "ih", "n", "ih", "ng"],
        
        # Responses
        "yes": ["y", "eh", "s"],
        "no": ["n", "ow"],
        "okay": ["ow", "k", "ey"],
        "thanks": ["th", "ae", "ng", "k", "s"],
        "thank": ["th", "ae", "ng", "k"],
        "you": ["y", "uw"],
        
        # JARVIS specific
        "jarvis": ["jh", "aa", "r", "v", "ih", "s"],
        "sir": ["s", "er"],
        "boss": ["b", "aa", "s"],
        "stark": ["s", "t", "aa", "r", "k"],
        "tony": ["t", "ow", "n", "iy"],
        
        # Actions
        "ready": ["r", "eh", "d", "iy"],
        "online": ["aa", "n", "l", "ay", "n"],
        "active": ["ae", "k", "t", "ih", "v"],
        "system": ["s", "ih", "s", "t", "ah", "m"],
        "processing": ["p", "r", "aa", "s", "eh", "s", "ih", "ng"],
        
        # Numbers (basic)
        "one": ["w", "ah", "n"],
        "two": ["t", "uw"],
        "three": ["th", "r", "iy"],
        "four": ["f", "ao", "r"],
        "five": ["f", "ay", "v"],
    }
    
    # Phoneme to sound mapping (sirf yeh phonemes defined hain)
    PHONEME_SOUNDS = {
        # Vowels
        'aa': {'type': 'vowel', 'formants': [730, 1090, 2450], 'duration': 0.19},
        'ae': {'type': 'vowel', 'formants': [660, 1700, 2400], 'duration': 0.2},
        'ah': {'type': 'vowel', 'formants': [520, 1190, 2400], 'duration': 0.16},
        'ao': {'type': 'vowel', 'formants': [570, 840, 2410], 'duration': 0.2},
        'ay': {'type': 'vowel', 'formants': [590, 1850, 2500], 'duration': 0.22},
        'eh': {'type': 'vowel', 'formants': [530, 1850, 2500], 'duration': 0.18},
        'er': {'type': 'vowel', 'formants': [490, 1350, 1700], 'duration': 0.2},
        'ey': {'type': 'vowel', 'formants': [400, 2000, 2600], 'duration': 0.2},
        'ih': {'type': 'vowel', 'formants': [390, 2000, 2600], 'duration': 0.18},
        'iy': {'type': 'vowel', 'formants': [270, 2300, 3000], 'duration': 0.22},
        'ow': {'type': 'vowel', 'formants': [450, 1000, 2350], 'duration': 0.2},
        'uh': {'type': 'vowel', 'formants': [440, 1020, 2250], 'duration': 0.16},
        'uw': {'type': 'vowel', 'formants': [300, 870, 2250], 'duration': 0.2},
        
        # Consonants
        'b': {'type': 'plosive', 'voiced': True, 'duration': 0.1},
        'd': {'type': 'plosive', 'voiced': True, 'duration': 0.1},
        'f': {'type': 'fricative', 'voiced': False, 'duration': 0.15},
        'g': {'type': 'plosive', 'voiced': True, 'duration': 0.12},
        'hh': {'type': 'fricative', 'voiced': False, 'duration': 0.1},
        'jh': {'type': 'affricate', 'voiced': True, 'duration': 0.15},
        'k': {'type': 'plosive', 'voiced': False, 'duration': 0.12},
        'l': {'type': 'liquid', 'voiced': True, 'duration': 0.15},
        'm': {'type': 'nasal', 'voiced': True, 'duration': 0.15},
        'n': {'type': 'nasal', 'voiced': True, 'duration': 0.14},
        'ng': {'type': 'nasal', 'voiced': True, 'duration': 0.15},
        'p': {'type': 'plosive', 'voiced': False, 'duration': 0.1},
        'r': {'type': 'liquid', 'voiced': True, 'duration': 0.15},
        's': {'type': 'fricative', 'voiced': False, 'duration': 0.18},
        't': {'type': 'plosive', 'voiced': False, 'duration': 0.1},
        'th': {'type': 'fricative', 'voiced': False, 'duration': 0.15},
        'v': {'type': 'fricative', 'voiced': True, 'duration': 0.15},
        'w': {'type': 'glide', 'voiced': True, 'duration': 0.12},
        'y': {'type': 'glide', 'voiced': True, 'duration': 0.12},
        'z': {'type': 'fricative', 'voiced': True, 'duration': 0.16},
    }
    
    @classmethod
    def get_word_phonemes(cls, word):
        """Sirf defined words ke liye phonemes do"""
        return cls.ALLOWED_WORDS.get(word.lower(), None)
    
    @classmethod
    def get_phoneme_sound(cls, phoneme):
        """Phoneme ka sound data do"""
        return cls.PHONEME_SOUNDS.get(phoneme, None)

# ==================== VOICE ENGINE ====================
class CleanVoiceEngine:
    """Simple voice engine - sirf defined words"""
    
    def __init__(self, sample_rate=22050):
        self.sr = sample_rate
        self.phonemes = PhonemeDB()
    
    def generate_sound(self, phoneme, duration, config):
        """Generate sound for a phoneme"""
        sound_data = self.phonemes.get_phoneme_sound(phoneme)
        if not sound_data:
            return None
        
        n_samples = int(self.sr * duration)
        if n_samples == 0:
            return None
        
        # Simple sine wave for vowels, noise for consonants
        t = np.linspace(0, duration, n_samples)
        
        if sound_data['type'] == 'vowel':
            # Generate vowel with formants
            f0 = config.base_f0
            # Simple sine wave with harmonics
            sound = (0.5 * np.sin(2 * np.pi * f0 * t) +
                    0.3 * np.sin(2 * np.pi * f0 * 2 * t) +
                    0.2 * np.sin(2 * np.pi * f0 * 3 * t))
            
            # Add formants (simplified)
            for i, f in enumerate(sound_data['formants'][:2]):
                sound += 0.2 * np.sin(2 * np.pi * f * t) / (i + 2)
            
        else:
            # Consonants - noise based
            if sound_data['type'] == 'plosive':
                # Short burst
                burst_len = min(int(0.01 * self.sr), n_samples)
                sound = np.random.randn(n_samples) * 0.1
                sound[:burst_len] *= 5  # Burst
            else:
                # Continuous noise
                sound = np.random.randn(n_samples) * 0.15
        
        # Apply envelope
        envelope = np.ones(n_samples)
        attack = int(0.005 * self.sr)
        release = int(0.01 * self.sr)
        
        if attack < n_samples:
            envelope[:attack] = np.linspace(0, 1, attack)
        if release < n_samples:
            envelope[-release:] = np.linspace(1, 0, release)
        
        sound = sound * envelope
        
        # Normalize
        if np.max(np.abs(sound)) > 0:
            sound = sound / np.max(np.abs(sound)) * 0.5
        
        return sound
    
    def generate_silence(self, duration):
        """Generate silence"""
        n_samples = int(self.sr * duration)
        return np.zeros(n_samples)

# ==================== SIMPLE JARVIS VOICE ====================
class SimpleJarvisVoice:
    """Simple voice generator - sirf defined words"""
    
    def __init__(self):
        self.engine = CleanVoiceEngine()
        self.config = VoiceConfig()
        self.phonemes = PhonemeDB()
        
        # Voice presets
        self.presets = {
            'jarvis': VoiceConfig(
                name="JARVIS", gender='male', base_f0=112,
                formants=[520, 1120, 2520, 3220, 3870],
                breathiness=0.04, quality=VoiceQuality.SMOOTH
            ),
            'friday': VoiceConfig(
                name="FRIDAY", gender='female', base_f0=195,
                formants=[720, 1350, 2750, 3550, 4200],
                breathiness=0.08, quality=VoiceQuality.BRIGHT
            ),
            'pepper': VoiceConfig(
                name="Pepper", gender='female', base_f0=185,
                formants=[710, 1330, 2730, 3530, 4180],
                breathiness=0.08, quality=VoiceQuality.WARM
            ),
        }
        
        # Cache directory
        self.cache_dir = Path("voice_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def set_voice(self, name):
        """Set voice preset"""
        if name in self.presets:
            self.config = self.presets[name]
            print(f"✓ Voice set to: {self.config.name}")
        else:
            print(f"✗ Voice '{name}' not found. Using default.")
    
    def text_to_phonemes(self, text):
        """Convert text to phonemes - ONLY if word exists"""
        text = text.lower().strip()
        words = text.split()
        
        all_phonemes = []
        unknown_words = []
        
        for word in words:
            # Clean word
            clean_word = ''.join(c for c in word if c.isalpha())
            if not clean_word:
                continue
            
            # Get phonemes for word
            phonemes = self.phonemes.get_word_phonemes(clean_word)
            if phonemes:
                all_phonemes.extend(phonemes)
                all_phonemes.append('pau')  # Add pause
            else:
                unknown_words.append(clean_word)
        
        return all_phonemes, unknown_words
    
    def generate_speech(self, text, output_file=None):
        """Generate speech - ONLY for known words"""
        if not text:
            print("No text provided")
            return None
        
        print(f"\n🎙️ Text: '{text}'")
        
        # Convert to phonemes
        phonemes, unknown = self.text_to_phonemes(text)
        
        if unknown:
            print(f"❌ Unknown words: {', '.join(unknown)}")
            print("✅ Sirf yeh words kaam karte hain:")
            # Show first 10 allowed words
            allowed = list(PhonemeDB.ALLOWED_WORDS.keys())[:10]
            print(f"   {', '.join(allowed)}...")
            return None
        
        if not phonemes:
            print("❌ No phonemes generated")
            return None
        
        print(f"📝 Phonemes: {phonemes}")
        
        # Check cache
        cache_key = hashlib.md5(f"{text}{self.config.name}".encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.npy"
        
        if cache_file.exists():
            print("📦 Loading from cache")
            audio = np.load(cache_file)
        else:
            # Generate audio
            audio_segments = []
            
            for phoneme in phonemes:
                if phoneme == 'pau':
                    segment = self.engine.generate_silence(0.1)
                else:
                    sound_data = self.phonemes.get_phoneme_sound(phoneme)
                    if sound_data:
                        duration = sound_data.get('duration', 0.15)
                        segment = self.engine.generate_sound(phoneme, duration, self.config)
                    else:
                        continue
                
                if segment is not None and len(segment) > 0:
                    audio_segments.append(segment)
            
            if not audio_segments:
                print("❌ No audio generated")
                return None
            
            audio = np.concatenate(audio_segments)
            np.save(cache_file, audio)
        
        # Save if requested
        if output_file:
            sf.write(output_file, audio, self.engine.sr)
            print(f"✓ Saved to: {output_file}")
        
        return audio
    
    def speak(self, text):
        """Speak text"""
        audio = self.generate_speech(text)
        if audio is not None:
            sd.play(audio, self.engine.sr)
            sd.wait()
    
    def list_allowed_words(self):
        """List all words that work"""
        print("\n📖 Allowed Words:")
        print("-" * 40)
        words = sorted(PhonemeDB.ALLOWED_WORDS.keys())
        for i, word in enumerate(words, 1):
            print(f"{word:15}", end="\n" if i % 5 == 0 else "")
        print("\n" + "-" * 40)
        print(f"Total: {len(words)} words")

# ==================== INTERACTIVE CONSOLE ====================
class SimpleConsole:
    """Simple console - sirf allowed words"""
    
    def __init__(self):
        self.jarvis = SimpleJarvisVoice()
        self.running = True
    
    def run(self):
        """Run console"""
        print("\n" + "="*50)
        print("🎙️ SIMPLE JARVIS VOICE")
        print("="*50)
        print("\nCommands:")
        print("  /voices    - List available voices")
        print("  /voice     - Change voice (jarvis/friday/pepper)")
        print("  /words     - Show allowed words")
        print("  /exit      - Exit")
        print("-" * 40)
        print("Sirf allowed words bol sakte hain!")
        
        while self.running:
            try:
                cmd = input("\nYou: ").strip()
                
                if cmd.startswith('/'):
                    if cmd == '/exit':
                        self.running = False
                    elif cmd == '/voices':
                        for name in self.jarvis.presets:
                            print(f"  • {name}")
                    elif cmd == '/words':
                        self.jarvis.list_allowed_words()
                    elif cmd.startswith('/voice'):
                        parts = cmd.split()
                        if len(parts) > 1:
                            self.jarvis.set_voice(parts[1])
                        else:
                            print("Usage: /voice [name]")
                    else:
                        print("Unknown command")
                elif cmd:
                    self.jarvis.speak(cmd)
                    
            except KeyboardInterrupt:
                self.running = False
                print("\nBye!")

# ==================== MAIN ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', '-i', action='store_true')
    parser.add_argument('--voice', '-v', default='jarvis')
    parser.add_argument('--text', '-t', help='Text to speak')
    parser.add_argument('--output', '-o', help='Output file')
    
    args = parser.parse_args()
    
    if args.interactive:
        console = SimpleConsole()
        console.run()
    elif args.text:
        jarvis = SimpleJarvisVoice()
        jarvis.set_voice(args.voice)
        
        # Check if all words are allowed
        _, unknown = jarvis.text_to_phonemes(args.text)
        if unknown:
            print(f"❌ Unknown words: {unknown}")
            print("Use /words to see allowed words")
        else:
            if args.output:
                jarvis.generate_speech(args.text, args.output)
            else:
                jarvis.speak(args.text)
    else:
        console = SimpleConsole()
        console.run()