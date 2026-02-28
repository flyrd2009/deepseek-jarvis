"""
JARVIS Custom Voice Synthesis Engine - DEBUGGED VERSION
Complete system for generating ANY voice type from scratch
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal
from dataclasses import dataclass
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
@dataclass
class VoiceConfig:
    """Complete voice configuration parameters"""
    base_f0: float = 120.0
    f0_variation: float = 0.3
    vibrato_rate: float = 5.0
    vibrato_depth: float = 0.02
    formants: List[float] = None
    formant_bandwidths: List[float] = None
    breathiness: float = 0.1
    jitter: float = 0.01
    shimmer: float = 0.1
    gender: str = "male"
    age: str = "adult"
    effect_type: str = "natural"
    modulation_freq: float = 0
    
    def __post_init__(self):
        if self.formants is None:
            self.set_formants_for_gender()
        if self.formant_bandwidths is None:
            self.formant_bandwidths = [80, 90, 120, 150]
    
    def set_formants_for_gender(self):
        """Set formant frequencies based on gender and age"""
        if self.gender == "male":
            if self.age == "child":
                self.formants = [800, 1400, 2800, 3500]
                self.base_f0 = 250
            elif self.age == "young":
                self.formants = [600, 1200, 2600, 3300]
                self.base_f0 = 130
            elif self.age == "adult":
                self.formants = [500, 1100, 2500, 3200]
                self.base_f0 = 120
            elif self.age == "old":
                self.formants = [450, 1000, 2400, 3100]
                self.base_f0 = 100
        elif self.gender == "female":
            if self.age == "child":
                self.formants = [1000, 1800, 3200, 4000]
                self.base_f0 = 300
            elif self.age == "young":
                self.formants = [750, 1400, 2800, 3600]
                self.base_f0 = 210
            elif self.age == "adult":
                self.formants = [700, 1300, 2700, 3500]
                self.base_f0 = 200
            elif self.age == "old":
                self.formants = [650, 1200, 2600, 3400]
                self.base_f0 = 180
        else:  # child
            self.formants = [900, 1600, 3000, 3800]
            self.base_f0 = 280

# ==================== SIMPLIFIED PHONEME DATABASE ====================
class PhonemeDatabase:
    """Simplified phoneme definitions"""
    
    # Simple phoneme to sound type mapping
    PHONEME_MAP = {
        # Vowels
        'aa': 'vowel', 'ae': 'vowel', 'ah': 'vowel', 'ao': 'vowel',
        'aw': 'vowel', 'ay': 'vowel', 'eh': 'vowel', 'er': 'vowel',
        'ey': 'vowel', 'ih': 'vowel', 'iy': 'vowel', 'ow': 'vowel',
        'oy': 'vowel', 'uh': 'vowel', 'uw': 'vowel',
        
        # Consonants
        'b': 'plosive', 'ch': 'affricate', 'd': 'plosive', 'dh': 'fricative',
        'f': 'fricative', 'g': 'plosive', 'hh': 'fricative', 'jh': 'affricate',
        'k': 'plosive', 'l': 'liquid', 'm': 'nasal', 'n': 'nasal',
        'ng': 'nasal', 'p': 'plosive', 'r': 'liquid', 's': 'fricative',
        'sh': 'fricative', 't': 'plosive', 'th': 'fricative', 'v': 'fricative',
        'w': 'glide', 'y': 'glide', 'z': 'fricative', 'zh': 'fricative',
    }
    
    # Vowel formants (simplified)
    VOWEL_FORMANTS = {
        'iy': [270, 2300, 3000],  # /i/ as in "heed"
        'ih': [390, 2000, 2600],  # /ɪ/ as in "hid"
        'eh': [530, 1850, 2500],  # /ɛ/ as in "head"
        'ae': [660, 1700, 2400],  # /æ/ as in "had"
        'ah': [520, 1190, 2400],  # /ʌ/ as in "hut"
        'aa': [730, 1090, 2450],  # /ɑ/ as in "hot"
        'ao': [570, 840, 2410],   # /ɔ/ as in "hawed"
        'uh': [440, 1020, 2250],  # /ʊ/ as in "hood"
        'uw': [300, 870, 2250],   # /u/ as in "who'd"
        'er': [490, 1350, 1700],  # /ɝ/ as in "herd"
        'ay': [590, 1850, 2500],  # /aɪ/ as in "hide"
        'aw': [590, 1300, 2500],  # /aʊ/ as in "how"
        'oy': [430, 1300, 2250],  # /ɔɪ/ as in "boy"
        'ey': [400, 2000, 2600],  # /eɪ/ as in "hayed"
        'ow': [450, 1000, 2350],  # /oʊ/ as in "hoe"
    }
    
    @classmethod
    def get_phoneme_type(cls, symbol):
        """Get phoneme type"""
        return cls.PHONEME_MAP.get(symbol, 'unknown')
    
    @classmethod
    def get_vowel_formants(cls, symbol):
        """Get formants for vowel"""
        return cls.VOWEL_FORMANTS.get(symbol, [500, 1500, 2500])

# ==================== TEXT TO PHONEME (SIMPLIFIED) ====================
class TextToPhoneme:
    """Simple text to phoneme converter"""
    
    # Word to phoneme mapping
    WORD_TO_PHONEMES = {
        'hello': ['hh', 'ah', 'l', 'ow'],
        'hi': ['hh', 'ay'],
        'goodbye': ['g', 'uh', 'd', 'b', 'ay'],
        'bye': ['b', 'ay'],
        'yes': ['y', 'eh', 's'],
        'no': ['n', 'ow'],
        'thanks': ['th', 'ae', 'ng', 'k', 's'],
        'thank': ['th', 'ae', 'ng', 'k'],
        'you': ['y', 'uw'],
        'please': ['p', 'l', 'iy', 'z'],
        'sorry': ['s', 'aa', 'r', 'iy'],
        'help': ['hh', 'eh', 'l', 'p'],
        'jarvis': ['jh', 'aa', 'r', 'v', 'ih', 's'],
        'iron': ['ay', 'er', 'n'],
        'man': ['m', 'ae', 'n'],
        'tony': ['t', 'ow', 'n', 'iy'],
        'stark': ['s', 't', 'aa', 'r', 'k'],
        'computer': ['k', 'ah', 'm', 'p', 'y', 'uw', 't', 'er'],
        'system': ['s', 'ih', 's', 't', 'ah', 'm'],
        'activate': ['ae', 'k', 't', 'ih', 'v', 'ey', 't'],
        'deactivate': ['d', 'iy', 'ae', 'k', 't', 'ih', 'v', 'ey', 't'],
        'status': ['s', 't', 'ey', 't', 'ah', 's'],
        'report': ['r', 'ih', 'p', 'ao', 'r', 't'],
        'ready': ['r', 'eh', 'd', 'iy'],
        'online': ['aa', 'n', 'l', 'ay', 'n'],
        'offline': ['aa', 'f', 'l', 'ay', 'n'],
        'processing': ['p', 'r', 'aa', 's', 'eh', 's', 'ih', 'ng'],
        'complete': ['k', 'ah', 'm', 'p', 'l', 'iy', 't'],
        'error': ['eh', 'r', 'er'],
        'warning': ['w', 'ao', 'r', 'n', 'ih', 'ng'],
    }
    
    # Character to phoneme mapping (fallback)
    CHAR_TO_PHONEME = {
        'a': ['ae'], 'b': ['b'], 'c': ['k'], 'd': ['d'], 'e': ['eh'],
        'f': ['f'], 'g': ['g'], 'h': ['hh'], 'i': ['ih'], 'j': ['jh'],
        'k': ['k'], 'l': ['l'], 'm': ['m'], 'n': ['n'], 'o': ['aa'],
        'p': ['p'], 'q': ['k'], 'r': ['r'], 's': ['s'], 't': ['t'],
        'u': ['ah'], 'v': ['v'], 'w': ['w'], 'x': ['k', 's'], 'y': ['y'],
        'z': ['z'],
    }
    
    @classmethod
    def text_to_phonemes(cls, text):
        """Convert text to list of phonemes"""
        if not text:
            return []
            
        text = text.lower().strip()
        words = text.split()
        phonemes = []
        
        for word in words:
            # Remove punctuation
            word = ''.join(c for c in word if c.isalpha())
            if not word:
                continue
                
            # Check if word is in dictionary
            if word in cls.WORD_TO_PHONEMES:
                phonemes.extend(cls.WORD_TO_PHONEMES[word])
            else:
                # Fallback: convert each character
                for char in word:
                    if char in cls.CHAR_TO_PHONEME:
                        phonemes.extend(cls.CHAR_TO_PHONEME[char])
                    else:
                        phonemes.append('ah')  # Default schwa sound
            
            # Add short pause between words
            phonemes.append('pau')
        
        return phonemes

# ==================== VOICE SYNTHESIS ENGINE ====================
class VoiceSynthesisEngine:
    """Core engine for generating voice"""
    
    def __init__(self, sample_rate=22050):  # Lower sample rate for simplicity
        self.sr = sample_rate
        self.phonemes = PhonemeDatabase()
        
    def generate_glottal_pulse(self, duration, f0, config):
        """Generate glottal pulse"""
        n_samples = int(self.sr * duration)
        if n_samples == 0:
            return np.array([])
            
        t = np.linspace(0, duration, n_samples, False)
        
        # Add jitter
        jitter_noise = 1 + config.jitter * np.random.randn(n_samples)
        f0_instant = f0 * jitter_noise
        
        # Add vibrato
        vibrato = 1 + config.vibrato_depth * np.sin(2 * np.pi * config.vibrato_rate * t)
        f0_instant *= vibrato
        
        # Generate pulse
        phase = np.cumsum(2 * np.pi * f0_instant / self.sr)
        pulse = 0.5 * (1 + np.sin(phase))
        
        # Add breathiness
        if config.breathiness > 0:
            noise = config.breathiness * np.random.randn(n_samples)
            pulse = pulse + noise
            pulse = pulse / (1 + config.breathiness)
        
        return pulse
    
    def apply_formant_filter(self, signal_data, formants, bandwidths):
        """Apply formant filter"""
        if len(signal_data) == 0:
            return signal_data
            
        result = signal_data.copy()
        
        for f, bw in zip(formants[:3], bandwidths[:3]):  # Use first 3 formants
            # Design resonator
            r = np.exp(-np.pi * bw / self.sr)
            theta = 2 * np.pi * f / self.sr
            
            # Filter coefficients
            b = [1, -2 * r * np.cos(theta), r**2]
            a = [1, -2 * r * np.cos(theta), r**2]
            
            # Apply filter
            try:
                result = signal.lfilter(b, a, result)
            except:
                pass
        
        return result
    
    def generate_vowel(self, vowel_symbol, duration, config):
        """Generate a vowel sound"""
        # Get formants for this vowel
        target_formants = self.phonemes.get_vowel_formants(vowel_symbol)
        
        # Scale formants based on voice config
        scale_factor = np.mean(config.formants) / 500
        scaled_formants = [f * scale_factor for f in target_formants]
        
        # Generate source
        source = self.generate_glottal_pulse(duration, config.base_f0, config)
        if len(source) == 0:
            return np.array([])
        
        # Apply formants
        voiced = self.apply_formant_filter(source, scaled_formants, config.formant_bandwidths)
        
        # Apply envelope
        envelope = np.ones_like(voiced)
        attack = int(0.01 * self.sr)
        release = int(0.01 * self.sr)
        
        if len(voiced) > attack + release:
            envelope[:attack] = np.linspace(0, 1, attack)
            envelope[-release:] = np.linspace(1, 0, release)
            voiced = voiced * envelope
        
        return voiced
    
    def generate_consonant(self, cons_symbol, duration, config):
        """Generate a consonant sound"""
        n_samples = int(self.sr * duration)
        if n_samples == 0:
            return np.array([])
            
        cons_type = self.phonemes.get_phoneme_type(cons_symbol)
        
        if cons_type in ['plosive', 'affricate']:
            # Burst sound
            burst_len = int(0.01 * self.sr)
            burst = np.random.randn(burst_len) * 0.3
            silence = np.zeros(max(0, n_samples - burst_len))
            sound = np.concatenate([burst, silence])
            
        elif cons_type == 'fricative':
            # Noise-based
            t = np.linspace(0, duration, n_samples, False)
            noise = np.random.randn(n_samples) * 0.2
            
            # Simple filtering based on consonant type
            if cons_symbol in ['s', 'z']:
                # High frequency
                b, a = signal.butter(2, 0.4, 'high')
            elif cons_symbol in ['f', 'v']:
                # Mid frequency
                b, a = signal.butter(2, [0.1, 0.3], 'band')
            else:
                b, a = signal.butter(2, 0.2, 'low')
                
            try:
                sound = signal.filtfilt(b, a, noise)
            except:
                sound = noise
                
        elif cons_type == 'nasal':
            # Nasal sounds
            source = self.generate_glottal_pulse(duration, config.base_f0 * 0.8, config)
            if len(source) > 0:
                sound = self.apply_formant_filter(source, [250, 1200], [60, 100])
            else:
                sound = np.random.randn(n_samples) * 0.1
                
        elif cons_type in ['liquid', 'glide']:
            # Liquid consonants
            source = self.generate_glottal_pulse(duration, config.base_f0, config)
            if len(source) > 0:
                if cons_symbol in ['l', 'r']:
                    sound = self.apply_formant_filter(source, [400, 1200], [80, 100])
                else:
                    sound = self.apply_formant_filter(source, [300, 800], [80, 100])
            else:
                sound = np.random.randn(n_samples) * 0.1
                
        else:
            # Default
            sound = np.random.randn(n_samples) * 0.1
        
        # Ensure correct length
        if len(sound) > n_samples:
            sound = sound[:n_samples]
        elif len(sound) < n_samples:
            sound = np.pad(sound, (0, n_samples - len(sound)))
        
        return sound
    
    def generate_silence(self, duration):
        """Generate silence"""
        n_samples = int(self.sr * duration)
        return np.zeros(n_samples)
    
    def apply_special_effects(self, audio, config):
        """Apply special effects"""
        if len(audio) == 0:
            return audio
            
        if config.effect_type == "robotic":
            t = np.linspace(0, len(audio)/self.sr, len(audio))
            mod_freq = config.modulation_freq if config.modulation_freq > 0 else 50
            modulator = np.sin(2 * np.pi * mod_freq * t)
            audio = audio * (0.7 + 0.3 * modulator)
            
        elif config.effect_type == "alien":
            # Add distortion
            audio = audio + 0.2 * audio**3
            # Pitch shift effect
            audio = np.interp(np.arange(0, len(audio), 0.8), np.arange(len(audio)), audio)
            audio = np.nan_to_num(audio)
            
        elif config.effect_type == "monster":
            audio = np.tanh(audio * 2)
            
        elif config.effect_type == "ghost":
            noise = np.random.randn(len(audio)) * 0.2
            audio = audio * 0.5 + noise
            
        # Normalize
        if len(audio) > 0 and np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9
            
        return audio

# ==================== JARVIS VOICE SYSTEM ====================
class JarvisVoiceSystem:
    """Main JARVIS voice system"""
    
    def __init__(self):
        self.engine = VoiceSynthesisEngine()
        self.tts = TextToPhoneme()
        self.current_config = VoiceConfig()
        
        # Voice presets
        self.voice_presets = {
            'jarvis_default': VoiceConfig(
                gender='male', age='adult', base_f0=110,
                formants=[550, 1150, 2550, 3250],
                breathiness=0.05, jitter=0.008, shimmer=0.08
            ),
            'jarvis_professional': VoiceConfig(
                gender='male', age='adult', base_f0=120,
                formants=[500, 1100, 2500, 3200],
                breathiness=0.02, jitter=0.005, shimmer=0.05
            ),
            'jarvis_friendly': VoiceConfig(
                gender='male', age='young', base_f0=130,
                formants=[550, 1200, 2600, 3300],
                breathiness=0.1, jitter=0.015, shimmer=0.12
            ),
            'female_assistant': VoiceConfig(
                gender='female', age='young', base_f0=200,
                formants=[700, 1300, 2700, 3500],
                breathiness=0.08, jitter=0.01, shimmer=0.1
            ),
            'child_voice': VoiceConfig(
                gender='child', age='child', base_f0=280,
                formants=[900, 1600, 3000, 3800],
                breathiness=0.15, jitter=0.02, shimmer=0.15
            ),
            'robotic_jarvis': VoiceConfig(
                gender='male', age='adult', base_f0=120,
                effect_type='robotic', modulation_freq=50
            ),
            'alien_jarvis': VoiceConfig(
                gender='male', age='adult', base_f0=150,
                effect_type='alien'
            ),
        }
    
    def set_voice(self, voice_name):
        """Set voice by preset name"""
        if voice_name in self.voice_presets:
            self.current_config = self.voice_presets[voice_name]
            print(f"✓ Voice set to: {voice_name}")
        else:
            print(f"✗ Voice '{voice_name}' not found. Using default.")
    
    def generate_speech(self, text, output_file=None):
        """Generate speech from text"""
        if not text:
            print("No text provided")
            return None
            
        print(f"Generating: '{text}'")
        
        # Convert text to phonemes
        phonemes = self.tts.text_to_phonemes(text)
        print(f"Phonemes: {phonemes}")
        
        if not phonemes:
            print("No phonemes generated")
            return None
        
        # Generate audio for each phoneme
        audio_segments = []
        
        for phoneme in phonemes:
            if phoneme == 'pau':  # Pause
                segment = self.engine.generate_silence(0.1)
                if len(segment) > 0:
                    audio_segments.append(segment)
                continue
            
            # Determine phoneme type and generate appropriate sound
            phoneme_type = PhonemeDatabase.get_phoneme_type(phoneme)
            
            if phoneme_type == 'vowel':
                segment = self.engine.generate_vowel(phoneme, 0.15, self.current_config)
            elif phoneme_type != 'unknown':
                segment = self.engine.generate_consonant(phoneme, 0.1, self.current_config)
            else:
                # Unknown phoneme, use default vowel
                segment = self.engine.generate_vowel('ah', 0.1, self.current_config)
            
            if segment is not None and len(segment) > 0:
                audio_segments.append(segment)
        
        if not audio_segments:
            print("No audio generated!")
            return None
        
        # Combine all segments
        audio = np.concatenate(audio_segments)
        
        # Apply effects
        audio = self.engine.apply_special_effects(audio, self.current_config)
        
        # Save if requested
        if output_file:
            sf.write(output_file, audio, self.engine.sr)
            print(f"✓ Saved to: {output_file}")
        
        return audio
    
    def speak(self, text):
        """Speak text immediately"""
        audio = self.generate_speech(text)
        if audio is not None and len(audio) > 0:
            sd.play(audio, self.engine.sr)
            sd.wait()
    
    def list_voices(self):
        """List available voices"""
        print("\nAvailable Voices:")
        for i, name in enumerate(self.voice_presets.keys(), 1):
            config = self.voice_presets[name]
            print(f"{i}. {name} - {config.gender}, {config.effect_type}")

# ==================== TEST FUNCTION ====================
def test_jarvis():
    """Test the JARVIS voice system"""
    
    print("=" * 50)
    print("JARVIS VOICE SYNTHESIS TEST")
    print("=" * 50)
    
    # Create instance
    jarvis = JarvisVoiceSystem()
    
    # List voices
    jarvis.list_voices()
    
    # Test phrases
    test_phrases = [
        "hello",
        "hi jarvis",
        "thank you",
        "goodbye"
    ]
    
    # Test each voice with first phrase
    print("\n" + "=" * 50)
    print("TESTING VOICES")
    print("=" * 50)
    
    for voice_name in ['jarvis_default', 'female_assistant', 'robotic_jarvis']:
        print(f"\nTesting: {voice_name}")
        jarvis.set_voice(voice_name)
        
        for phrase in test_phrases[:2]:  # Test first 2 phrases
            print(f"Saying: '{phrase}'")
            audio = jarvis.generate_speech(phrase, f"test_{voice_name}_{phrase}.wav")
            
            if audio is not None:
                # Uncomment to play
                # sd.play(audio, jarvis.engine.sr)
                # sd.wait()
                pass
    
    print("\n✓ Test complete! Check generated WAV files.")
    
    # Interactive test
    print("\n" + "=" * 50)
    print("INTERACTIVE TEST")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    jarvis.set_voice('jarvis_default')
    
    while True:
        try:
            text = input("\nYou: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
            if text:
                jarvis.speak(text)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

# ==================== MAIN ====================
if __name__ == "__main__":
    # Run the test
    test_jarvis()