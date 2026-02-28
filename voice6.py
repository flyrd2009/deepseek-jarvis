"""
JARVIS Voice System - ULTRA CLEAR EDITION
Hollywood quality + Crystal clear speech + Maximum intelligibility
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
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
    ANGRY = "angry"
    EXCITED = "excited"
    CALM = "calm"
    CONFIDENT = "confident"
    FRIENDLY = "friendly"

class VoiceQuality(Enum):
    STANDARD = "standard"
    WARM = "warm"
    BRIGHT = "bright"
    DARK = "dark"
    BREATHY = "breathy"
    SMOOTH = "smooth"
    VELVETY = "velvety"
    CRISP = "crisp"
    CLEAR = "clear"  # NEW: Ultra clear quality

# ==================== VOICE CONFIG ====================
@dataclass
class VoiceConfig:
    """Hollywood grade voice configuration with CLARITY focus"""
    name: str = "JARVIS"
    gender: str = "male"
    age: str = "adult"
    base_f0: float = 115.0
    formants: List[float] = None
    formant_bandwidths: List[float] = None
    formant_strength: float = 1.2  # INCREASED for clarity
    breathiness: float = 0.02  # REDUCED for clarity (less noise)
    jitter: float = 0.004  # REDUCED for stability
    shimmer: float = 0.03  # REDUCED for stability
    vibrato_rate: float = 4.5  # SLOWER for clarity
    vibrato_depth: float = 0.008  # REDUCED for clarity
    tremolo_rate: float = 3.0
    tremolo_depth: float = 0.003
    spectral_tilt: float = -2.0  # REDUCED for more highs
    harmonic_richness: float = 1.2  # INCREASED for richer sound
    effect_type: str = "natural"
    modulation_freq: float = 0.0
    reverb_amount: float = 0.01  # MINIMAL reverb for clarity
    reverb_decay: float = 0.2
    echo_delay: float = 0.0
    echo_decay: float = 0.0
    emotion: Emotion = Emotion.CONFIDENT
    emotion_intensity: float = 0.5
    quality: VoiceQuality = VoiceQuality.CLEAR
    
    # CLARITY BOOST parameters
    master_volume: float = 2.0
    pre_amp: float = 1.5
    output_gain: float = 1.2
    clarity_boost: float = 1.3  # NEW: High frequency boost
    consonant_emphasis: float = 1.4  # NEW: Make consonants clearer
    vowel_purity: float = 1.2  # NEW: Cleaner vowels
    
    def __post_init__(self):
        if self.formants is None:
            self.set_formants_for_gender()
        if self.formant_bandwidths is None:
            # NARROWER bandwidths for clearer formants
            self.formant_bandwidths = [40, 50, 70, 100, 120]
        self.apply_emotion_to_params()
        self.apply_quality_to_params()
    
    def set_formants_for_gender(self):
        """Professional formant settings - EMPHASIZED for clarity"""
        if self.gender == "male":
            if self.age == "adult":
                # WIDER spaced formants for clarity
                self.formants = [550, 1150, 2600, 3300, 4000]
            elif self.age == "young":
                self.formants = [650, 1300, 2700, 3400, 4100]
            elif self.age == "old":
                self.formants = [500, 1100, 2500, 3200, 3900]
        elif self.gender == "female":
            if self.age == "adult":
                self.formants = [750, 1400, 2800, 3600, 4300]
            elif self.age == "young":
                self.formants = [800, 1500, 2900, 3700, 4400]
            else:
                self.formants = [700, 1350, 2750, 3550, 4200]
        else:
            self.formants = [650, 1300, 2700, 3400, 4100]
    
    def apply_emotion_to_params(self):
        """Apply emotion modifications - GENTLE for clarity"""
        intensity = self.emotion_intensity * 0.7  # Reduced impact
        
        emotion_map = {
            Emotion.HAPPY: {'base_f0': 1 + 0.1 * intensity, 'vibrato_rate': 1 * intensity},
            Emotion.SAD: {'base_f0': 1 - 0.08 * intensity},
            Emotion.ANGRY: {'base_f0': 1 + 0.12 * intensity, 'formant_strength': 1 + 0.1 * intensity},
            Emotion.EXCITED: {'base_f0': 1 + 0.15 * intensity, 'vibrato_rate': 1.5 * intensity},
            Emotion.CALM: {'base_f0': 1 - 0.03 * intensity, 'vibrato_depth': -0.002 * intensity},
            Emotion.FRIENDLY: {'base_f0': 1 + 0.05 * intensity, 'breathiness': 0.02 * intensity}
        }
        
        if self.emotion in emotion_map:
            for param, factor in emotion_map[self.emotion].items():
                if hasattr(self, param):
                    current = getattr(self, param)
                    if isinstance(current, (int, float)):
                        setattr(self, param, current * (1 + factor))
    
    def apply_quality_to_params(self):
        """Apply quality modifications - CLARITY focused"""
        quality_map = {
            VoiceQuality.CLEAR: {  # NEW: Ultra clear preset
                'formants': [f * 1.05 for f in self.formants],
                'formant_bandwidths': [bw * 0.8 for bw in self.formant_bandwidths],
                'breathiness': self.breathiness * 0.5,
                'jitter': self.jitter * 0.5,
                'shimmer': self.shimmer * 0.5,
                'spectral_tilt': self.spectral_tilt + 3,
                'harmonic_richness': self.harmonic_richness * 1.2,
                'clarity_boost': 1.5,
                'consonant_emphasis': 1.5
            },
            VoiceQuality.VELVETY: {
                'formants': [f * 0.95 for f in self.formants],
                'breathiness': self.breathiness + 0.08,
                'spectral_tilt': self.spectral_tilt - 2
            },
            VoiceQuality.WARM: {
                'formants': [f * 0.97 for f in self.formants],
                'breathiness': self.breathiness + 0.05
            },
            VoiceQuality.BRIGHT: {
                'formants': [f * 1.08 for f in self.formants],
                'spectral_tilt': self.spectral_tilt + 4
            },
            VoiceQuality.CRISP: {
                'formants': [f * 1.03 for f in self.formants],
                'formant_bandwidths': [bw * 0.9 for bw in self.formant_bandwidths],
                'jitter': self.jitter * 0.7
            },
        }
        
        if self.quality in quality_map:
            for param, value in quality_map[self.quality].items():
                if param == 'formants':
                    self.formants = value
                elif param == 'formant_bandwidths':
                    self.formant_bandwidths = value
                elif hasattr(self, param):
                    setattr(self, param, value)

# ==================== PHONEME DATABASE ====================
class PhonemeEngine:
    """Complete phoneme database with ENHANCED formants for clarity"""
    
    # Core phoneme sounds - with OPTIMIZED formants for clarity
    PHONEMES = {
        # Vowels - WIDER formant spacing for clarity
        'iy': {'type': 'vowel', 'formants': [300, 2400, 3100], 'duration': 0.22},
        'ih': {'type': 'vowel', 'formants': [420, 2100, 2700], 'duration': 0.18},
        'eh': {'type': 'vowel', 'formants': [560, 1950, 2600], 'duration': 0.18},
        'ae': {'type': 'vowel', 'formants': [700, 1800, 2500], 'duration': 0.2},
        'ah': {'type': 'vowel', 'formants': [550, 1250, 2500], 'duration': 0.16},
        'aa': {'type': 'vowel', 'formants': [770, 1150, 2550], 'duration': 0.19},
        'ao': {'type': 'vowel', 'formants': [600, 900, 2500], 'duration': 0.2},
        'uh': {'type': 'vowel', 'formants': [470, 1100, 2350], 'duration': 0.16},
        'uw': {'type': 'vowel', 'formants': [330, 950, 2350], 'duration': 0.21},
        'er': {'type': 'vowel', 'formants': [520, 1400, 1800], 'duration': 0.2},
        'ay': {'type': 'vowel', 'formants': [620, 1950, 2600], 'duration': 0.22},
        'aw': {'type': 'vowel', 'formants': [620, 1400, 2600], 'duration': 0.22},
        'oy': {'type': 'vowel', 'formants': [460, 1400, 2350], 'duration': 0.22},
        'ey': {'type': 'vowel', 'formants': [430, 2100, 2700], 'duration': 0.2},
        'ow': {'type': 'vowel', 'formants': [480, 1100, 2450], 'duration': 0.2},
        
        # Consonants - ENHANCED for clarity
        'p': {'type': 'plosive', 'voiced': False, 'duration': 0.12, 'energy': 1.3},
        'b': {'type': 'plosive', 'voiced': True, 'duration': 0.12, 'energy': 1.2},
        't': {'type': 'plosive', 'voiced': False, 'duration': 0.12, 'energy': 1.4},
        'd': {'type': 'plosive', 'voiced': True, 'duration': 0.12, 'energy': 1.3},
        'k': {'type': 'plosive', 'voiced': False, 'duration': 0.14, 'energy': 1.4},
        'g': {'type': 'plosive', 'voiced': True, 'duration': 0.14, 'energy': 1.3},
        
        'f': {'type': 'fricative', 'voiced': False, 'duration': 0.2, 'energy': 1.2},
        'v': {'type': 'fricative', 'voiced': True, 'duration': 0.18, 'energy': 1.1},
        'th': {'type': 'fricative', 'voiced': False, 'duration': 0.18, 'energy': 1.2},
        'dh': {'type': 'fricative', 'voiced': True, 'duration': 0.16, 'energy': 1.1},
        's': {'type': 'fricative', 'voiced': False, 'duration': 0.22, 'energy': 1.5},  # EMPHASIZED
        'z': {'type': 'fricative', 'voiced': True, 'duration': 0.2, 'energy': 1.3},
        'sh': {'type': 'fricative', 'voiced': False, 'duration': 0.22, 'energy': 1.3},
        'zh': {'type': 'fricative', 'voiced': True, 'duration': 0.2, 'energy': 1.2},
        'hh': {'type': 'fricative', 'voiced': False, 'duration': 0.14, 'energy': 1.0},
        
        'ch': {'type': 'affricate', 'voiced': False, 'duration': 0.17, 'energy': 1.3},
        'jh': {'type': 'affricate', 'voiced': True, 'duration': 0.17, 'energy': 1.2},
        
        'm': {'type': 'nasal', 'voiced': True, 'duration': 0.17, 'formants': [280, 1300]},
        'n': {'type': 'nasal', 'voiced': True, 'duration': 0.16, 'formants': [280, 1400]},
        'ng': {'type': 'nasal', 'voiced': True, 'duration': 0.17, 'formants': [280, 1200]},
        
        'l': {'type': 'liquid', 'voiced': True, 'duration': 0.17, 'formants': [450, 1300]},
        'r': {'type': 'liquid', 'voiced': True, 'duration': 0.17, 'formants': [400, 1200]},
        
        'w': {'type': 'glide', 'voiced': True, 'duration': 0.14, 'formants': [350, 900]},
        'y': {'type': 'glide', 'voiced': True, 'duration': 0.14, 'formants': [350, 2300]},
    }
    
    # Letter to phoneme mapping
    LETTER_TO_PHONEME = {
        'a': ['ae'], 'b': ['b'], 'c': ['k'], 'd': ['d'], 'e': ['eh'],
        'f': ['f'], 'g': ['g'], 'h': ['hh'], 'i': ['ih'], 'j': ['jh'],
        'k': ['k'], 'l': ['l'], 'm': ['m'], 'n': ['n'], 'o': ['aa'],
        'p': ['p'], 'q': ['k'], 'r': ['r'], 's': ['s'], 't': ['t'],
        'u': ['ah'], 'v': ['v'], 'w': ['w'], 'x': ['k', 's'], 'y': ['y'],
        'z': ['z'],
    }
    
    # Common words (same as before, but using enhanced phonemes)
    COMMON_WORDS = {
        'the': ['dh', 'ah'],
        'and': ['ae', 'n', 'd'],
        'for': ['f', 'ao', 'r'],
        'you': ['y', 'uw'],
        'are': ['aa', 'r'],
        'is': ['ih', 'z'],
        'it': ['ih', 't'],
        'this': ['dh', 'ih', 's'],
        'that': ['dh', 'ae', 't'],
        'with': ['w', 'ih', 'th'],
        'from': ['f', 'r', 'ah', 'm'],
        'have': ['hh', 'ae', 'v'],
        'will': ['w', 'ih', 'l'],
        'can': ['k', 'ae', 'n'],
        'say': ['s', 'ey'],
        'hello': ['hh', 'ah', 'l', 'ow'],
        'hi': ['hh', 'ay'],
        'good': ['g', 'uh', 'd'],
        'morning': ['m', 'ao', 'r', 'n', 'ih', 'ng'],
        'jarvis': ['jh', 'aa', 'r', 'v', 'ih', 's'],
    }
    
    @classmethod
    def get_phoneme(cls, symbol):
        """Get phoneme data with enhanced energy"""
        return cls.PHONEMES.get(symbol, cls.PHONEMES.get('ah'))
    
    @classmethod
    def word_to_phonemes(cls, word):
        """Convert any word to phonemes"""
        word = word.lower().strip()
        
        if word in cls.COMMON_WORDS:
            return cls.COMMON_WORDS[word]
        
        # Simple prediction
        phonemes = []
        for char in word:
            if char in cls.LETTER_TO_PHONEME:
                phonemes.extend(cls.LETTER_TO_PHONEME[char])
            elif char in 'aeiou':
                phonemes.append('ah')
        
        return phonemes if phonemes else ['ah']

# ==================== ULTRA CLEAR VOICE ENGINE ====================
class UltraClearVoiceEngine:
    """Voice engine OPTIMIZED for maximum clarity and intelligibility"""
    
    def __init__(self, sample_rate=48000):  # HIGHER sample rate for clarity
        self.sr = sample_rate
        self.phonemes = PhonemeEngine()
    
    def generate_glottal_pulse(self, duration, f0, config):
        """Generate glottal pulse with CLARITY optimization"""
        n_samples = int(self.sr * duration)
        if n_samples == 0:
            return np.array([])
        
        t = np.linspace(0, duration, n_samples, False)
        
        # Generate pulse with STRONG harmonics for clarity
        pulse = np.zeros(n_samples)
        harmonic_weights = [1.0, 0.7, 0.5, 0.3, 0.2]  # Stronger higher harmonics
        
        for h, weight in enumerate(harmonic_weights, 1):
            pulse += weight * np.sin(2 * np.pi * h * f0 * t)
        
        # BOOST high frequencies for clarity
        pulse = pulse * 1.5
        
        # Minimal jitter/shimmer for stability
        jitter_mod = 1 + config.jitter * 0.5 * np.random.randn(n_samples)
        pulse = np.interp(np.arange(n_samples) * np.mean(jitter_mod), 
                         np.arange(n_samples), pulse, left=0, right=0)
        pulse = np.nan_to_num(pulse)
        
        # Very subtle vibrato
        vibrato = 1 + config.vibrato_depth * 0.5 * np.sin(2 * np.pi * config.vibrato_rate * t)
        pulse = pulse * vibrato
        
        return pulse * config.pre_amp
    
    def apply_vocal_tract(self, source, formants, bandwidths, config):
        """Apply vocal tract with CLARITY emphasis"""
        if len(source) == 0:
            return source
        
        result = source.copy()
        
        # Apply formants with NARROW bandwidths for clarity
        for f, bw in zip(formants[:4], bandwidths[:4]):
            r = np.exp(-np.pi * bw / self.sr)
            theta = 2 * np.pi * f / self.sr
            
            b = [1 - r**2]
            a = [1, -2 * r * np.cos(theta), r**2]
            
            try:
                result = signal.lfilter(b, a, result)
            except:
                pass
        
        # Apply HIGH FREQUENCY boost for clarity
        if config.clarity_boost > 1:
            # High-shelf filter
            b, a = signal.butter(2, 2000/(self.sr/2), 'high')
            try:
                highs = signal.filtfilt(b, a, result)
                result = result + highs * (config.clarity_boost - 1)
            except:
                pass
        
        # Minimal spectral tilt (more highs)
        if config.spectral_tilt > -5:
            b = [1, -0.9]
            try:
                result = signal.filtfilt(b, [1], result)
            except:
                pass
        
        return result * config.formant_strength
    
    def generate_phoneme(self, phoneme, duration, config):
        """Generate phoneme with MAXIMUM clarity"""
        phoneme_data = self.phonemes.get_phoneme(phoneme)
        if not phoneme_data:
            return None
        
        n_samples = int(self.sr * duration)
        if n_samples == 0:
            return None
        
        # Get energy level for this phoneme
        energy = phoneme_data.get('energy', 1.0)
        
        if phoneme_data['type'] == 'vowel':
            # Vowels - PURE and CLEAR
            voice_formants = config.formants[:3]
            target_formants = phoneme_data['formants']
            
            blended = []
            for i in range(3):
                blended.append(target_formants[i] * 0.8 + voice_formants[i] * 0.2)
            
            source = self.generate_glottal_pulse(duration, config.base_f0, config)
            sound = self.apply_vocal_tract(source, blended, config.formant_bandwidths, config)
            
            # Apply vowel purity
            sound = sound * config.vowel_purity * energy
            
        else:
            # Consonants - EMPHASIZED for clarity
            if phoneme_data['type'] == 'plosive':
                burst_len = min(int(0.012 * self.sr), n_samples)
                sound = np.random.randn(n_samples) * 0.2
                sound[:burst_len] *= 10 * energy * config.consonant_emphasis
                
            elif phoneme_data['type'] == 'fricative':
                sound = np.random.randn(n_samples) * 0.3 * energy * config.consonant_emphasis
                # High-pass filter for sibilants
                if phoneme in ['s', 'z', 'sh', 'zh']:
                    b, a = signal.butter(2, 3000/(self.sr/2), 'high')
                    try:
                        sound = signal.filtfilt(b, a, sound)
                    except:
                        pass
                        
            elif phoneme_data['type'] == 'nasal':
                source = self.generate_glottal_pulse(duration, config.base_f0 * 0.8, config)
                sound = source * 0.6 * energy
                
            elif phoneme_data['type'] == 'liquid':
                source = self.generate_glottal_pulse(duration, config.base_f0, config)
                sound = source * 0.5 * energy
                
            elif phoneme_data['type'] == 'glide':
                source = self.generate_glottal_pulse(duration, config.base_f0, config)
                sound = source * 0.4 * energy
                
            else:  # affricates
                burst_len = min(int(0.008 * self.sr), n_samples)
                sound = np.random.randn(n_samples) * 0.25 * energy * config.consonant_emphasis
                sound[:burst_len] *= 8
        
        # Crisp envelope with faster attack for consonants
        envelope = np.ones(n_samples)
        attack = int(0.002 * self.sr)  # Faster attack
        release = int(0.008 * self.sr)  # Faster release
        
        if attack < n_samples:
            envelope[:attack] = np.linspace(0, 1, attack)
        if release < n_samples:
            envelope[-release:] = np.linspace(1, 0, release)
        
        if len(sound) > len(envelope):
            sound = sound[:len(envelope)]
        elif len(sound) < len(envelope):
            sound = np.pad(sound, (0, len(envelope) - len(sound)))
        
        sound = sound * envelope
        
        # Apply master volume
        sound = sound * config.master_volume * config.output_gain
        
        return sound
    
    def generate_silence(self, duration):
        """Generate silence"""
        n_samples = int(self.sr * duration)
        return np.zeros(n_samples)
    
    def apply_clarity_effects(self, audio, config):
        """Apply effects that ENHANCE clarity"""
        if len(audio) == 0:
            return audio
        
        # MINIMAL reverb (just a touch for warmth)
        if config.reverb_amount > 0:
            reverb_len = int(0.03 * self.sr)  # Shorter reverb
            impulse = np.zeros(reverb_len)
            impulse[0] = 1
            impulse[1:] = np.exp(-np.arange(1, reverb_len) / (config.reverb_decay * self.sr))
            reverb = np.convolve(audio, impulse * config.reverb_amount * 0.3, mode='same')
            audio = audio + reverb
        
        # Final HIGH-FREQUENCY boost for clarity
        b, a = signal.butter(2, 2000/(self.sr/2), 'high')
        try:
            highs = signal.filtfilt(b, a, audio)
            audio = audio + highs * 0.2
        except:
            pass
        
        # COMPRESSION for consistent volume
        threshold = 0.3
        ratio = 3.0
        compressed = np.where(np.abs(audio) > threshold,
                             threshold + (np.abs(audio) - threshold) / ratio,
                             np.abs(audio))
        audio = np.sign(audio) * compressed
        
        # Final normalize to HOT but clean
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.98
        
        return audio

# ==================== ULTRA CLEAR JARVIS ====================
class UltraClearJarvis:
    """Complete voice system - MAXIMUM CLARITY and INTELLIGIBILITY"""
    
    def __init__(self):
        self.engine = UltraClearVoiceEngine()
        self.config = VoiceConfig()
        self.phonemes = PhonemeEngine()
        
        # Voice presets with CLARITY focus
        self.presets = {
            'jarvis_clear': VoiceConfig(
                name="JARVIS Clear", gender='male', base_f0=118,
                formants=[570, 1180, 2650, 3350, 4050],
                formant_bandwidths=[35, 45, 60, 85, 100],
                breathiness=0.01, jitter=0.002, shimmer=0.02,
                vibrato_rate=4.0, vibrato_depth=0.005,
                harmonic_richness=1.3,
                master_volume=2.2, pre_amp=1.6, output_gain=1.3,
                clarity_boost=1.5, consonant_emphasis=1.5, vowel_purity=1.3,
                quality=VoiceQuality.CLEAR
            ),
            'friday_clear': VoiceConfig(
                name="FRIDAY Clear", gender='female', base_f0=200,
                formants=[780, 1450, 2850, 3650, 4350],
                formant_bandwidths=[40, 50, 65, 90, 105],
                breathiness=0.02, jitter=0.003, shimmer=0.03,
                master_volume=2.0, clarity_boost=1.4, consonant_emphasis=1.4,
                quality=VoiceQuality.CLEAR
            ),
            'pepper_clear': VoiceConfig(
                name="Pepper Clear", gender='female', base_f0=190,
                formants=[740, 1380, 2780, 3580, 4250],
                formant_bandwidths=[38, 48, 62, 88, 102],
                breathiness=0.02, quality=VoiceQuality.WARM,
                clarity_boost=1.3, consonant_emphasis=1.3
            ),
            'robotic_clear': VoiceConfig(
                name="Robotic Clear", gender='male', base_f0=135,
                effect_type='robotic', modulation_freq=35,
                formant_bandwidths=[30, 40, 50, 70, 90],
                master_volume=2.5, clarity_boost=1.6, consonant_emphasis=1.6
            ),
        }
        
        self.cache_dir = Path("voice_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def set_voice(self, name):
        """Set voice preset"""
        if name in self.presets:
            self.config = self.presets[name]
            print(f"✓ Voice set to: {self.config.name}")
            print(f"  Clarity: {self.config.clarity_boost}x, Consonants: {self.config.consonant_emphasis}x")
        else:
            print(f"✗ Voice '{name}' not found")
    
    def set_clarity(self, level: float):
        """Manually adjust clarity (1.0 to 2.0)"""
        self.config.clarity_boost = max(1.0, min(2.0, level))
        self.config.consonant_emphasis = self.config.clarity_boost * 1.1
        print(f"✓ Clarity set to: {self.config.clarity_boost}x")
    
    def text_to_phonemes(self, text):
        """Convert text to phonemes"""
        text = text.lower().strip()
        words = text.split()
        
        all_phonemes = []
        for word in words:
            clean_word = ''.join(c for c in word if c.isalpha())
            if clean_word:
                word_phonemes = self.phonemes.word_to_phonemes(clean_word)
                all_phonemes.extend(word_phonemes)
                all_phonemes.append('pau')
        
        return all_phonemes
    
    def generate_speech(self, text, output_file=None):
        """Generate ULTRA CLEAR speech"""
        if not text:
            print("No text provided")
            return None
        
        print(f"\n🎙️ Generating CLEAR voice: '{text}'")
        print(f"🔊 Clarity Boost: {self.config.clarity_boost}x")
        print(f"🔊 Consonant Emphasis: {self.config.consonant_emphasis}x")
        
        phonemes = self.text_to_phonemes(text)
        print(f"📝 Phonemes: {len(phonemes)} sounds")
        
        if not phonemes:
            return None
        
        # Check cache
        cache_key = hashlib.md5(f"{text}{self.config.name}{self.config.clarity_boost}".encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.npy"
        
        if cache_file.exists():
            print("📦 Loading from cache")
            audio = np.load(cache_file)
        else:
            audio_segments = []
            
            for phoneme in phonemes:
                if phoneme == 'pau':
                    segment = self.engine.generate_silence(0.08)  # Slightly shorter pauses
                else:
                    phoneme_data = self.phonemes.get_phoneme(phoneme)
                    if phoneme_data:
                        duration = phoneme_data.get('duration', 0.15)
                        segment = self.engine.generate_phoneme(phoneme, duration, self.config)
                    else:
                        continue
                
                if segment is not None and len(segment) > 0:
                    audio_segments.append(segment)
            
            if not audio_segments:
                return None
            
            audio = np.concatenate(audio_segments)
            audio = self.engine.apply_clarity_effects(audio, self.config)
            np.save(cache_file, audio)
        
        if output_file:
            sf.write(output_file, audio, self.engine.sr)
            print(f"✓ Saved to: {output_file}")
        
        return audio
    
    def speak(self, text):
        """Speak with MAXIMUM clarity"""
        audio = self.generate_speech(text)
        if audio is not None:
            sd.play(audio, self.engine.sr)
            sd.wait()
    
    def list_voices(self):
        """List all voices"""
        print("\n🎭 ULTRA CLEAR VOICES:")
        for name, config in self.presets.items():
            print(f"  • {name:15} (Clarity: {config.clarity_boost}x)")

# ==================== INTERACTIVE CONSOLE ====================
class ClarityConsole:
    """Interactive console with clarity control"""
    
    def __init__(self):
        self.jarvis = UltraClearJarvis()
        self.running = True
    
    def run(self):
        """Run console"""
        print("\n" + "="*70)
        print("🎙️ JARVIS ULTRA CLEAR VOICE - MAXIMUM INTELLIGIBILITY")
        print("   Crystal clear speech - Every word perfectly audible")
        print("="*70)
        print("\nCommands:")
        print("  /voices        - List voices")
        print("  /voice         - Change voice (jarvis_clear/friday_clear/pepper_clear/robotic_clear)")
        print("  /clarity       - Adjust clarity (1.0 to 2.0)")
        print("  /exit          - Exit")
        print("-" * 50)
        print("Type anything - EVERY WORD will be crystal clear!")
        
        while self.running:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith('/'):
                    cmd = user_input[1:].split()
                    
                    if cmd[0] == 'exit':
                        self.running = False
                    elif cmd[0] == 'voices':
                        self.jarvis.list_voices()
                    elif cmd[0] == 'voice' and len(cmd) > 1:
                        self.jarvis.set_voice(cmd[1])
                    elif cmd[0] == 'clarity' and len(cmd) > 1:
                        try:
                            clarity = float(cmd[1])
                            self.jarvis.set_clarity(clarity)
                        except:
                            print("Use: /clarity 1.5")
                    else:
                        print("Unknown command")
                else:
                    self.jarvis.speak(user_input)
                    
            except KeyboardInterrupt:
                self.running = False
                print("\nGoodbye!")

# ==================== MAIN ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', '-i', action='store_true')
    parser.add_argument('--voice', '-v', default='jarvis_clear')
    parser.add_argument('--clarity', '-c', type=float, default=1.5)
    parser.add_argument('--text', '-t')
    parser.add_argument('--output', '-o')
    
    args = parser.parse_args()
    
    if args.interactive:
        console = ClarityConsole()
        console.run()
    elif args.text:
        jarvis = UltraClearJarvis()
        jarvis.set_voice(args.voice)
        jarvis.set_clarity(args.clarity)
        
        print(f"🔊 Clarity: {args.clarity}x")
        if args.output:
            jarvis.generate_speech(args.text, args.output)
        else:
            jarvis.speak(args.text)
    else:
        console = ClarityConsole()
        console.run()