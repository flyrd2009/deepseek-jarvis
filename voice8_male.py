"""
JARVIS Voice System - PERFECT HUMAN MALE VOICE (FIXED)
Complete control over every acoustic parameter
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal, interpolate, fft
from scipy.signal import butter, filtfilt, lfilter  # <-- CORRECT: filtfilt
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum
import hashlib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==================== CONSTANTS ====================
SAMPLE_RATE = 48000
NYQUIST = SAMPLE_RATE / 2

# ==================== ENUMS ====================
class GlottalMode(Enum):
    MODAL = "modal"
    BREATHY = "breathy"
    CREAKY = "creaky"
    FALSETTO = "falsetto"
    PRESSED = "pressed"

class VocalTractShape(Enum):
    NEUTRAL = "neutral"
    PHARYNGEAL = "pharyngeal"
    FAUCAL = "faucal"
    RETROFLEX = "retroflex"
    PALATAL = "palatal"
    VELAR = "velar"

# ==================== VOICE CONFIG ====================
@dataclass
class PerfectMaleVoiceConfig:
    """Complete male voice configuration"""
    f0: float = 120.0
    f1: float = 550.0
    f2: float = 1200.0
    f3: float = 2500.0
    f4: float = 3500.0
    f5: float = 4500.0
    
    bw1: float = 50.0
    bw2: float = 70.0
    bw3: float = 110.0
    bw4: float = 150.0
    bw5: float = 200.0
    
    gain1: float = 0.0
    gain2: float = 0.0
    gain3: float = -3.0
    gain4: float = -6.0
    gain5: float = -9.0
    
    open_quotient: float = 0.6
    aspiration_level: float = 0.02
    vibrato_rate: float = 5.2
    vibrato_depth: float = 0.3
    jitter_amount: float = 0.2
    shimmer_amount: float = 0.1
    
    volume_db: float = -3.0
    name: str = "Male Voice"
    
    def get_formants(self) -> np.ndarray:
        return np.array([self.f1, self.f2, self.f3, self.f4, self.f5])
    
    def get_bandwidths(self) -> np.ndarray:
        return np.array([self.bw1, self.bw2, self.bw3, self.bw4, self.bw5])
    
    def get_gains(self) -> np.ndarray:
        gains_db = np.array([self.gain1, self.gain2, self.gain3, self.gain4, self.gain5])
        return 10 ** (gains_db / 20)

# ==================== PHONEME DATABASE ====================
class PhonemeDatabase:
    """Complete phoneme database"""
    
    VOWELS = {
        'i': {'f1': 270, 'f2': 2300, 'f3': 3000},
        'ɪ': {'f1': 400, 'f2': 2000, 'f3': 2600},
        'e': {'f1': 350, 'f2': 2100, 'f3': 2700},
        'ɛ': {'f1': 550, 'f2': 1900, 'f3': 2500},
        'æ': {'f1': 700, 'f2': 1800, 'f3': 2500},
        'ə': {'f1': 500, 'f2': 1500, 'f3': 2500},
        'ɜ': {'f1': 550, 'f2': 1400, 'f3': 2400},
        'a': {'f1': 800, 'f2': 1400, 'f3': 2400},
        'u': {'f1': 300, 'f2': 800, 'f3': 2300},
        'ʊ': {'f1': 450, 'f2': 1000, 'f3': 2300},
        'o': {'f1': 400, 'f2': 800, 'f3': 2200},
        'ɔ': {'f1': 600, 'f2': 900, 'f3': 2400},
        'ɑ': {'f1': 750, 'f2': 1100, 'f3': 2500},
        'aɪ': {'type': 'diphthong', 'f1_start': 750, 'f1_end': 400, 
               'f2_start': 1200, 'f2_end': 2000},
        'aʊ': {'type': 'diphthong', 'f1_start': 750, 'f1_end': 450,
               'f2_start': 1200, 'f2_end': 1000},
        'ɔɪ': {'type': 'diphthong', 'f1_start': 600, 'f1_end': 400,
               'f2_start': 900, 'f2_end': 2000},
        'eɪ': {'type': 'diphthong', 'f1_start': 500, 'f1_end': 400,
               'f2_start': 1900, 'f2_end': 2100},
        'oʊ': {'type': 'diphthong', 'f1_start': 450, 'f1_end': 400,
               'f2_start': 900, 'f2_end': 850},
    }
    
    CONSONANTS = {
        'p': {'type': 'plosive', 'voiced': False, 'burst_freq': [500, 1500]},
        'b': {'type': 'plosive', 'voiced': True, 'burst_freq': [400, 1400], 'voicing_bar': 100},
        't': {'type': 'plosive', 'voiced': False, 'burst_freq': [3500, 5000]},
        'd': {'type': 'plosive', 'voiced': True, 'burst_freq': [3000, 4500], 'voicing_bar': 120},
        'k': {'type': 'plosive', 'voiced': False, 'burst_freq': [1500, 2500]},
        'g': {'type': 'plosive', 'voiced': True, 'burst_freq': [1400, 2400], 'voicing_bar': 110},
        
        'f': {'type': 'fricative', 'voiced': False, 'noise_center': 7000, 'noise_bandwidth': 2000},
        'v': {'type': 'fricative', 'voiced': True, 'noise_center': 6500, 'noise_bandwidth': 1800},
        'θ': {'type': 'fricative', 'voiced': False, 'noise_center': 6000, 'noise_bandwidth': 1500},
        'ð': {'type': 'fricative', 'voiced': True, 'noise_center': 5500, 'noise_bandwidth': 1500},
        's': {'type': 'fricative', 'voiced': False, 'noise_center': 7000, 'noise_bandwidth': 3000},
        'z': {'type': 'fricative', 'voiced': True, 'noise_center': 6500, 'noise_bandwidth': 2800},
        'ʃ': {'type': 'fricative', 'voiced': False, 'noise_center': 4000, 'noise_bandwidth': 2000},
        'ʒ': {'type': 'fricative', 'voiced': True, 'noise_center': 3800, 'noise_bandwidth': 1800},
        'h': {'type': 'fricative', 'voiced': False, 'noise_center': 1500, 'noise_bandwidth': 1000},
        
        'm': {'type': 'nasal', 'voiced': True, 'f1': 250, 'f2': 1200, 'f3': 2200},
        'n': {'type': 'nasal', 'voiced': True, 'f1': 250, 'f2': 1300, 'f3': 2300},
        'ŋ': {'type': 'nasal', 'voiced': True, 'f1': 250, 'f2': 1100, 'f3': 2100},
        
        'l': {'type': 'liquid', 'voiced': True, 'f1': 350, 'f2': 1200, 'f3': 2600},
        'r': {'type': 'liquid', 'voiced': True, 'f1': 350, 'f2': 1100, 'f3': 2200},
        'w': {'type': 'glide', 'voiced': True, 'f1': 300, 'f2': 700, 'f3': 2100},
        'j': {'type': 'glide', 'voiced': True, 'f1': 300, 'f2': 2200, 'f3': 2900},
    }
    
    @classmethod
    def get_vowel(cls, symbol):
        return cls.VOWELS.get(symbol)
    
    @classmethod
    def get_consonant(cls, symbol):
        return cls.CONSONANTS.get(symbol)

# ==================== GLOTTAL SOURCE ====================
class GlottalSource:
    """LF glottal model"""
    
    def __init__(self, sr=SAMPLE_RATE):
        self.sr = sr
    
    def generate_pulse(self, duration, f0, config):
        """Generate glottal pulse"""
        n_samples = int(self.sr * duration)
        if n_samples == 0:
            return np.array([])
        
        t = np.linspace(0, duration, n_samples, False)
        T0 = 1.0 / max(f0, 50)
        
        # LF model
        pulse = np.zeros(n_samples)
        for i, time_point in enumerate(t):
            phase = (time_point % T0) / T0
            if phase < 0.8:
                pulse[i] = np.sin(np.pi * phase / 0.8)**2
            else:
                pulse[i] = -np.exp(-(phase - 0.8) / 0.05)
        
        # Add variations
        jitter = 1 + (config.jitter_amount / 100) * np.random.randn(n_samples)
        pulse = np.interp(np.arange(n_samples) * np.mean(jitter), 
                         np.arange(n_samples), pulse, left=0, right=0)
        pulse = np.nan_to_num(pulse)
        
        shimmer = 1 + (config.shimmer_amount / 100) * np.random.randn(n_samples)
        pulse = pulse * shimmer
        
        # Vibrato
        vibrato = 1 + (config.vibrato_depth / 100) * np.sin(2 * np.pi * config.vibrato_rate * t)
        pulse = pulse * vibrato
        
        return pulse

# ==================== VOCAL TRACT ====================
class VocalTract:
    """Vocal tract model"""
    
    def __init__(self, sr=SAMPLE_RATE):
        self.sr = sr
    
    def apply_formants(self, signal, formants, bandwidths, gains):
        """Apply formant filter"""
        if len(signal) == 0:
            return signal
        
        result = signal.copy()
        
        for f, bw, gain in zip(formants, bandwidths, gains):
            if f <= 0 or bw <= 0:
                continue
            
            r = np.exp(-np.pi * bw / self.sr)
            theta = 2 * np.pi * f / self.sr
            
            b = [gain * (1 - r**2)]
            a = [1, -2 * r * np.cos(theta), r**2]
            
            try:
                result = signal.lfilter(b, a, result)
            except:
                pass
        
        return result

# ==================== PERFECT MALE VOICE ENGINE (FIXED) ====================
class PerfectMaleVoiceEngine:
    """Complete voice synthesis engine - FIXED version"""
    
    def __init__(self, sr=SAMPLE_RATE):
        self.sr = sr
        self.glottis = GlottalSource(sr)
        self.tract = VocalTract(sr)
        self.phonemes = PhonemeDatabase()
    
    def generate_vowel(self, vowel_symbol, duration, config):
        """Generate vowel"""
        vowel = self.phonemes.get_vowel(vowel_symbol)
        if not vowel:
            return np.array([])
        
        # Handle diphthongs
        if vowel.get('type') == 'diphthong':
            return self.generate_diphthong(vowel, duration, config)
        
        # Blend with voice
        f1 = 0.8 * vowel['f1'] + 0.2 * config.f1
        f2 = 0.8 * vowel['f2'] + 0.2 * config.f2
        f3 = 0.8 * vowel['f3'] + 0.2 * config.f3
        
        formants = np.array([f1, f2, f3, config.f4, config.f5])
        bandwidths = config.get_bandwidths()
        gains = config.get_gains()
        
        # Generate source
        source = self.glottis.generate_pulse(duration, config.f0, config)
        if len(source) == 0:
            return np.array([])
        
        # Apply formants
        sound = self.tract.apply_formants(source, formants, bandwidths, gains)
        
        return sound
    
    def generate_diphthong(self, diphthong, duration, config):
        """Generate diphthong"""
        n_samples = int(self.sr * duration)
        if n_samples == 0:
            return np.array([])
        
        sound = np.zeros(n_samples)
        
        for i in range(n_samples):
            t = i / n_samples
            
            # Interpolate formants
            f1 = diphthong['f1_start'] * (1 - t) + diphthong['f1_end'] * t
            f2 = diphthong['f2_start'] * (1 - t) + diphthong['f2_end'] * t
            
            formants = np.array([f1, f2, config.f3, config.f4, config.f5])
            bandwidths = config.get_bandwidths()
            gains = config.get_gains()
            
            # Generate short segment
            source = self.glottis.generate_pulse(1/self.sr, config.f0, config)
            if len(source) > 0:
                seg = self.tract.apply_formants(source, formants, bandwidths, gains)
                if len(seg) > 0:
                    sound[i] = seg[0]
        
        return sound
    
    def generate_consonant(self, cons_symbol, duration, config):
        """Generate consonant - FIXED: filtfilt typo corrected"""
        cons = self.phonemes.get_consonant(cons_symbol)
        if not cons:
            return np.array([])
        
        n_samples = int(self.sr * duration)
        if n_samples == 0:
            return np.array([])
        
        cons_type = cons['type']
        voiced = cons.get('voiced', False)
        
        if cons_type == 'plosive':
            # Generate burst
            burst_len = min(int(0.01 * self.sr), n_samples)
            burst_freqs = cons.get('burst_freq', [1000, 2000])
            
            # Filtered noise burst
            burst_noise = np.random.randn(burst_len) * 0.5
            b, a = butter(4, [burst_freqs[0]/NYQUIST, burst_freqs[1]/NYQUIST], 'band')
            burst = filtfilt(b, a, burst_noise)  # <-- FIXED: was 'filtfilter'
            
            # Add voicing bar if voiced
            if voiced and 'voicing_bar' in cons:
                voicing = self.glottis.generate_pulse(burst_len/self.sr, cons['voicing_bar'], config)
                if len(voicing) == burst_len:
                    burst = burst * 0.4 + voicing * 0.6
            
            # Aspiration
            aspiration_len = n_samples - burst_len
            if aspiration_len > 0:
                aspiration = self.generate_aspiration(aspiration_len/self.sr, config)
                if len(aspiration) > aspiration_len:
                    aspiration = aspiration[:aspiration_len]
                sound = np.concatenate([burst, aspiration])
            else:
                sound = burst[:n_samples]
        
        elif cons_type == 'fricative':
            # Fricative noise
            noise = np.random.randn(n_samples)
            center = cons.get('noise_center', 4000)
            bw = cons.get('noise_bandwidth', 1000)
            
            b, a = butter(4, [(center - bw/2)/NYQUIST, (center + bw/2)/NYQUIST], 'band')
            sound = filtfilt(b, a, noise) * 0.3
            
            # Add voicing if needed
            if voiced:
                voicing = self.glottis.generate_pulse(duration, config.f0 * 0.7, config)
                if len(voicing) == len(sound):
                    sound = sound * 0.3 + voicing * 0.7
        
        elif cons_type == 'nasal':
            # Nasal
            source = self.glottis.generate_pulse(duration, config.f0 * 0.8, config)
            formants = np.array([cons.get('f1', 250), 
                                 cons.get('f2', 1200),
                                 cons.get('f3', 2200),
                                 config.f4, config.f5])
            bandwidths = config.get_bandwidths()
            gains = config.get_gains()
            sound = self.tract.apply_formants(source, formants, bandwidths, gains)
        
        else:  # liquids and glides
            source = self.glottis.generate_pulse(duration, config.f0, config)
            formants = np.array([cons.get('f1', 350),
                                 cons.get('f2', 1200),
                                 cons.get('f3', 2500),
                                 config.f4, config.f5])
            bandwidths = config.get_bandwidths()
            gains = config.get_gains()
            sound = self.tract.apply_formants(source, formants, bandwidths, gains)
        
        # Ensure correct length
        if len(sound) > n_samples:
            sound = sound[:n_samples]
        elif len(sound) < n_samples:
            sound = np.pad(sound, (0, n_samples - len(sound)))
        
        return sound
    
    def generate_aspiration(self, duration, config):
        """Generate aspiration noise"""
        n_samples = int(self.sr * duration)
        if n_samples <= 0:
            return np.array([])
        
        noise = np.random.randn(n_samples)
        b, a = butter(2, 1000/NYQUIST, 'low')
        aspiration = filtfilt(b, a, noise)
        
        # Apply decay
        envelope = np.exp(-np.linspace(0, 4, n_samples))
        aspiration = aspiration * envelope
        
        return aspiration * 0.2
    
    def generate_silence(self, duration):
        """Generate silence"""
        n_samples = int(self.sr * duration)
        return np.zeros(n_samples)

# ==================== PERFECT MALE JARVIS ====================
class PerfectMaleJarvis:
    """Complete male voice system"""
    
    def __init__(self):
        self.engine = PerfectMaleVoiceEngine()
        self.config = PerfectMaleVoiceConfig()
        self.phonemes = PhonemeDatabase()
        
        # Voice presets
        self.presets = {
            'male_deep': PerfectMaleVoiceConfig(
                f0=95, f1=500, f2=1100, f3=2400,
                name="Deep Male"
            ),
            'male_standard': PerfectMaleVoiceConfig(
                f0=120, f1=550, f2=1200, f3=2500,
                name="Standard Male"
            ),
            'male_young': PerfectMaleVoiceConfig(
                f0=140, f1=600, f2=1300, f3=2600,
                name="Young Male"
            ),
        }
        
        self.cache_dir = Path("male_voice_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def set_voice(self, name):
        """Set voice preset"""
        if name in self.presets:
            self.config = self.presets[name]
            print(f"✓ Voice: {self.config.name}")
        else:
            print(f"✗ Voice '{name}' not found")
    
    def set_pitch(self, f0):
        """Set pitch"""
        self.config.f0 = max(80, min(300, f0))
        print(f"✓ Pitch: {self.config.f0} Hz")
    
    def text_to_phonemes(self, text):
        """Simple text to phonemes"""
        text = text.lower()
        words = text.split()
        
        phonemes = []
        for word in words:
            if word == 'hello':
                phonemes.extend(['h', 'ə', 'l', 'oʊ'])
            elif word == 'world':
                phonemes.extend(['w', 'ɜ', 'r', 'l', 'd'])
            elif word == 'jarvis':
                phonemes.extend(['dʒ', 'ɑ', 'r', 'v', 'ɪ', 's'])
            elif word == 'i':
                phonemes.append('aɪ')
            elif word == 'am':
                phonemes.extend(['æ', 'm'])
            else:
                # Fallback
                for c in word:
                    if c in 'aeiou':
                        phonemes.append('ə')
                    else:
                        phonemes.append('t')
            phonemes.append('pau')
        
        return phonemes
    
    def generate_speech(self, text, output_file=None):
        """Generate speech"""
        if not text:
            return None
        
        print(f"\n🎙️ Generating: '{text}'")
        print(f"   Pitch: {self.config.f0} Hz")
        
        phonemes = self.text_to_phonemes(text)
        print(f"   Phonemes: {len(phonemes)}")
        
        if not phonemes:
            return None
        
        # Check cache
        cache_key = hashlib.md5(f"{text}{self.config.f0}".encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.npy"
        
        if cache_file.exists():
            print("   Loading from cache")
            audio = np.load(cache_file)
        else:
            # Generate
            segments = []
            
            for phoneme in phonemes:
                if phoneme == 'pau':
                    segment = self.engine.generate_silence(0.1)
                else:
                    if phoneme in self.phonemes.VOWELS:
                        segment = self.engine.generate_vowel(phoneme, 0.2, self.config)
                    elif phoneme in self.phonemes.CONSONANTS:
                        segment = self.engine.generate_consonant(phoneme, 0.15, self.config)
                    else:
                        continue
                
                if segment is not None and len(segment) > 0:
                    segments.append(segment)
            
            if not segments:
                return None
            
            audio = np.concatenate(segments)
            
            # Normalize
            volume = 10 ** (self.config.volume_db / 20)
            audio = audio * volume
            
            np.save(cache_file, audio)
        
        # Save output
        if output_file:
            sf.write(output_file, audio, self.engine.sr)
            print(f"   Saved: {output_file}")
        
        return audio
    
    def speak(self, text):
        """Speak text"""
        audio = self.generate_speech(text)
        if audio is not None:
            sd.play(audio, self.engine.sr)
            sd.wait()
    
    def list_voices(self):
        """List voices"""
        print("\n🎭 Male Voices:")
        for name, config in self.presets.items():
            print(f"  • {name}: {config.name} - {config.f0}Hz")

# ==================== MAIN ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', '-i', action='store_true')
    parser.add_argument('--voice', '-v', default='male_standard')
    parser.add_argument('--pitch', '-p', type=float)
    parser.add_argument('--text', '-t')
    parser.add_argument('--output', '-o')
    
    args = parser.parse_args()
    
    jarvis = PerfectMaleJarvis()
    jarvis.set_voice(args.voice)
    
    if args.pitch:
        jarvis.set_pitch(args.pitch)
    
    if args.text:
        if args.output:
            jarvis.generate_speech(args.text, args.output)
        else:
            jarvis.speak(args.text)
    elif args.interactive:
        print("\n🎤 PERFECT MALE VOICE")
        print("Commands: /pitch <hz>, /exit")
        while True:
            cmd = input("\n🎬 ").strip()
            if cmd == '/exit':
                break
            elif cmd.startswith('/pitch'):
                _, p = cmd.split()
                jarvis.set_pitch(float(p))
            elif cmd:
                jarvis.speak(cmd)
    else:
        jarvis.speak("Hello, I am Jarvis")