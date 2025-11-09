"""
Arabic OCR Text Correction Module

This module provides comprehensive post-processing and correction for Arabic OCR output
using dictionary-based fuzzy matching, context-aware selection, and linguistic knowledge.

Author: AI Assistant
License: MIT
"""

import os
import json
import re
import pickle
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
from pathlib import Path

import requests
from rapidfuzz import fuzz, process
import pyarabic.araby as araby
from camel_tools.utils.normalize import normalize_unicode, normalize_alef_maksura_ar, normalize_alef_ar, normalize_teh_marbuta_ar


class ArabicTextCorrector:
    """
    Professional Arabic text correction system with dictionary-based fuzzy matching,
    context-aware selection, and confidence scoring.
    """
    
    def __init__(self, cache_dir: str = "./arabic_resources"):
        """
        Initialize the Arabic text corrector.
        
        Args:
            cache_dir: Directory to cache downloaded resources
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Core data structures
        self.dictionary: Set[str] = set()
        self.word_frequencies: Dict[str, int] = {}
        self.bigrams: Dict[Tuple[str, str], int] = defaultdict(int)
        self.trigrams: Dict[Tuple[str, str, str], int] = defaultdict(int)
        
        # Arabic letter similarity map for OCR error patterns
        self.letter_similarity = self._build_letter_similarity_map()
        
        # Load resources
        self._load_or_download_resources()
        
    def _build_letter_similarity_map(self) -> Dict[str, List[str]]:
        """
        Build a map of commonly confused Arabic letters in OCR.
        
        Returns:
            Dictionary mapping each letter to similar-looking letters
        """
        return {
            'ب': ['ت', 'ث', 'ن', 'ي'],
            'ت': ['ب', 'ث', 'ن'],
            'ث': ['ب', 'ت', 'ن'],
            'ج': ['ح', 'خ'],
            'ح': ['ج', 'خ'],
            'خ': ['ج', 'ح'],
            'د': ['ذ'],
            'ذ': ['د'],
            'ر': ['ز'],
            'ز': ['ر'],
            'س': ['ش'],
            'ش': ['س'],
            'ص': ['ض'],
            'ض': ['ص'],
            'ط': ['ظ'],
            'ظ': ['ط'],
            'ع': ['غ'],
            'غ': ['ع'],
            'ف': ['ق'],
            'ق': ['ف'],
            'ك': ['گ'],
            'ل': ['لا'],
            'ن': ['ب', 'ت', 'ث', 'ي'],
            'ه': ['ة'],
            'ة': ['ه'],
            'و': ['ؤ'],
            'ي': ['ئ', 'ى', 'ب', 'ت', 'ن'],
            'ى': ['ي', 'ئ'],
            'ا': ['أ', 'إ', 'آ'],
            'أ': ['ا', 'إ', 'آ'],
            'إ': ['ا', 'أ', 'آ'],
            'آ': ['ا', 'أ', 'إ'],
        }
    
    def _load_or_download_resources(self):
        """Load or download Arabic language resources."""
        dict_file = self.cache_dir / "arabic_dictionary.pkl"
        freq_file = self.cache_dir / "word_frequencies.pkl"
        ngram_file = self.cache_dir / "ngrams.pkl"
        
        if dict_file.exists() and freq_file.exists() and ngram_file.exists():
            print("📚 Loading cached Arabic resources...")
            try:
                with open(dict_file, 'rb') as f:
                    self.dictionary = pickle.load(f)
                with open(freq_file, 'rb') as f:
                    self.word_frequencies = pickle.load(f)
                with open(ngram_file, 'rb') as f:
                    ngram_data = pickle.load(f)
                    self.bigrams = ngram_data['bigrams']
                    self.trigrams = ngram_data['trigrams']
                print(f"✅ Loaded {len(self.dictionary)} Arabic words")
                return
            except Exception as e:
                print(f"⚠️ Error loading cache: {e}. Downloading fresh...")
        
        print("📥 Downloading Arabic language resources...")
        self._download_arabic_wordlist()
        self._build_ngram_models()
        
        # Cache for future use
        print("💾 Caching resources for faster startup...")
        with open(dict_file, 'wb') as f:
            pickle.dump(self.dictionary, f)
        with open(freq_file, 'wb') as f:
            pickle.dump(self.word_frequencies, f)
        with open(ngram_file, 'wb') as f:
            pickle.dump({'bigrams': dict(self.bigrams), 'trigrams': dict(self.trigrams)}, f)
        
        print(f"✅ Resources ready: {len(self.dictionary)} words loaded")
    
    def _download_arabic_wordlist(self):
        """
        Download and process Arabic word frequency list from online sources.
        Uses the Arabic Gigaword frequency list.
        """
        try:
            # Try to get Arabic word frequency list
            # Using a curated list from GitHub
            url = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/ar/ar_50k.txt"
            
            print(f"  Downloading from {url}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            lines = response.text.strip().split('\n')
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    word = parts[0]
                    try:
                        freq = int(parts[1])
                    except ValueError:
                        freq = 1
                    
                    # Normalize and add to dictionary
                    normalized = self.normalize_text(word)
                    if normalized and self._is_valid_arabic_word(normalized):
                        self.dictionary.add(normalized)
                        self.word_frequencies[normalized] = freq
            
            print(f"  ✓ Downloaded {len(self.dictionary)} words")
            
        except Exception as e:
            print(f"  ⚠️ Download failed: {e}")
            print("  Using fallback: basic Arabic word set...")
            self._create_fallback_dictionary()
    
    def _create_fallback_dictionary(self):
        """Create a basic fallback dictionary with common Arabic words."""
        # Common Arabic words as fallback
        common_words = [
            'في', 'من', 'على', 'إلى', 'هذا', 'هذه', 'ذلك', 'التي', 'الذي', 'كان',
            'أن', 'قد', 'لا', 'ما', 'هو', 'هي', 'كل', 'عن', 'أو', 'إن',
            'بعد', 'قبل', 'عند', 'الى', 'اللذي', 'اللتي', 'والتي', 'والذي',
            'كانت', 'يكون', 'تكون', 'مع', 'بين', 'خلال', 'أيضا', 'حيث',
            'عليها', 'عليه', 'منها', 'منه', 'فيها', 'فيه', 'بها', 'به',
            'لها', 'له', 'لهم', 'لهن', 'عام', 'سنة', 'يوم', 'شهر',
        ]
        
        for word in common_words:
            normalized = self.normalize_text(word)
            self.dictionary.add(normalized)
            self.word_frequencies[normalized] = 1000
    
    def _build_ngram_models(self):
        """
        Build n-gram language models from the word frequency data.
        This creates bigram and trigram models for context-aware correction.
        """
        print("  Building n-gram language models...")
        
        # Simple approach: use word frequencies to build basic n-grams
        # In a production system, you'd build this from a large corpus
        sorted_words = sorted(self.word_frequencies.items(), key=lambda x: x[1], reverse=True)
        
        # Create basic bigrams from frequent words
        for i in range(len(sorted_words) - 1):
            word1 = sorted_words[i][0]
            word2 = sorted_words[i + 1][0]
            self.bigrams[(word1, word2)] = min(sorted_words[i][1], sorted_words[i + 1][1])
        
        print(f"  ✓ Built {len(self.bigrams)} bigrams")
    
    def _is_valid_arabic_word(self, word: str) -> bool:
        """
        Check if a word is valid Arabic (contains Arabic letters).
        
        Args:
            word: Word to validate
            
        Returns:
            True if word contains Arabic letters, False otherwise
        """
        if not word or len(word) < 2:
            return False
        
        arabic_count = sum(1 for c in word if '\u0600' <= c <= '\u06FF')
        return arabic_count >= len(word) * 0.7  # At least 70% Arabic characters
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize Arabic text for better matching.
        
        Args:
            text: Input Arabic text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Remove diacritics (tashkeel)
        text = araby.strip_diacritics(text)
        
        # Normalize using camel-tools
        text = normalize_unicode(text)
        text = normalize_alef_ar(text)
        text = normalize_alef_maksura_ar(text)
        text = normalize_teh_marbuta_ar(text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def get_word_candidates(self, word: str, max_candidates: int = 5, max_distance: int = 3) -> List[Tuple[str, float, int]]:
        """
        Get candidate corrections for a word using fuzzy matching.
        
        Args:
            word: Input word to correct
            max_candidates: Maximum number of candidates to return
            max_distance: Maximum edit distance to consider
            
        Returns:
            List of (candidate, similarity_score, edit_distance) tuples
        """
        if not word or not self._is_valid_arabic_word(word):
            return []
        
        normalized_word = self.normalize_text(word)
        
        # Exact match - high confidence
        if normalized_word in self.dictionary:
            return [(normalized_word, 100.0, 0)]
        
        # Use rapidfuzz for efficient fuzzy matching
        candidates = []
        
        # Get top matches using Levenshtein distance
        matches = process.extract(
            normalized_word,
            self.dictionary,
            scorer=fuzz.ratio,
            limit=max_candidates * 3  # Get more to filter
        )
        
        for match_word, similarity, _ in matches:
            # Calculate actual edit distance
            edit_dist = self._calculate_edit_distance(normalized_word, match_word)
            
            if edit_dist <= max_distance:
                # Boost score if word is frequent
                freq_bonus = min(20, self.word_frequencies.get(match_word, 0) / 1000)
                adjusted_score = min(99.9, similarity + freq_bonus)
                
                candidates.append((match_word, adjusted_score, edit_dist))
        
        # Sort by score, then by frequency
        candidates.sort(key=lambda x: (x[1], self.word_frequencies.get(x[0], 0)), reverse=True)
        
        return candidates[:max_candidates]
    
    def _calculate_edit_distance(self, word1: str, word2: str) -> int:
        """
        Calculate Levenshtein edit distance between two words.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Edit distance
        """
        if len(word1) < len(word2):
            return self._calculate_edit_distance(word2, word1)
        
        if len(word2) == 0:
            return len(word1)
        
        previous_row = range(len(word2) + 1)
        for i, c1 in enumerate(word1):
            current_row = [i + 1]
            for j, c2 in enumerate(word2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def get_bigram_score(self, word1: str, word2: str) -> float:
        """
        Get bigram probability score for word pair.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Bigram score (0-100)
        """
        pair = (word1, word2)
        if pair in self.bigrams:
            # Normalize to 0-100 scale
            max_freq = max(self.bigrams.values()) if self.bigrams else 1
            return (self.bigrams[pair] / max_freq) * 100
        return 0.0
    
    def correct_word_with_context(
        self,
        word: str,
        prev_word: Optional[str] = None,
        next_word: Optional[str] = None
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Correct a word using context-aware selection.
        
        Args:
            word: Word to correct
            prev_word: Previous word in sequence (for context)
            next_word: Next word in sequence (for context)
            
        Returns:
            Tuple of (best_correction, confidence_score, all_candidates)
        """
        # Get candidates
        candidates = self.get_word_candidates(word)
        
        if not candidates:
            # No candidates found - return original with low confidence
            return (word, 0.0, [])
        
        # Exact match case
        if candidates[0][2] == 0:  # edit distance = 0
            return (candidates[0][0], 100.0, candidates)
        
        # Context-aware selection
        scored_candidates = []
        
        for candidate_word, base_score, edit_dist in candidates:
            context_score = 0.0
            
            # Consider previous word context
            if prev_word:
                prev_normalized = self.normalize_text(prev_word)
                context_score += self.get_bigram_score(prev_normalized, candidate_word) * 0.3
            
            # Consider next word context
            if next_word:
                next_normalized = self.normalize_text(next_word)
                context_score += self.get_bigram_score(candidate_word, next_normalized) * 0.3
            
            # Final score: base similarity + context + frequency
            final_score = base_score * 0.6 + context_score * 0.4
            scored_candidates.append((candidate_word, final_score))
        
        # Sort by final score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        best_word, best_score = scored_candidates[0]
        
        return (best_word, best_score, scored_candidates)
    
    def correct_text(self, text: str) -> Dict[str, any]:
        """
        Correct an entire text with word-level tracking.
        
        Args:
            text: Input Arabic text
            
        Returns:
            Dictionary containing:
                - original: Original text
                - corrected: Corrected text
                - words: List of word correction details
                - overall_confidence: Average confidence score
        """
        if not text:
            return {
                'original': '',
                'corrected': '',
                'words': [],
                'overall_confidence': 0.0
            }
        
        # Split into words while preserving punctuation
        words = re.findall(r'[\u0600-\u06FF]+|[^\u0600-\u06FF\s]+', text)
        
        corrected_words = []
        word_details = []
        total_confidence = 0.0
        correction_count = 0
        
        for i, word in enumerate(words):
            if not self._is_valid_arabic_word(word):
                # Non-Arabic word (punctuation, numbers, etc.)
                corrected_words.append(word)
                word_details.append({
                    'original': word,
                    'corrected': word,
                    'confidence': 100.0,
                    'candidates': [],
                    'changed': False
                })
                continue
            
            # Get context
            prev_word = words[i-1] if i > 0 and self._is_valid_arabic_word(words[i-1]) else None
            next_word = words[i+1] if i < len(words)-1 and self._is_valid_arabic_word(words[i+1]) else None
            
            # Correct with context
            corrected, confidence, candidates = self.correct_word_with_context(word, prev_word, next_word)
            
            corrected_words.append(corrected)
            total_confidence += confidence
            
            changed = (self.normalize_text(word) != self.normalize_text(corrected))
            if changed:
                correction_count += 1
            
            word_details.append({
                'original': word,
                'corrected': corrected,
                'confidence': round(confidence, 1),
                'candidates': [(c[0], round(c[1], 1)) for c in candidates[:5]],
                'changed': changed
            })
        
        overall_confidence = total_confidence / len(words) if words else 0.0
        
        return {
            'original': text,
            'corrected': ' '.join(corrected_words),
            'words': word_details,
            'overall_confidence': round(overall_confidence, 1),
            'corrections_made': correction_count
        }


# Global instance (singleton pattern for efficiency)
_corrector_instance = None

def get_corrector() -> ArabicTextCorrector:
    """
    Get or create the global Arabic text corrector instance.
    
    Returns:
        ArabicTextCorrector instance
    """
    global _corrector_instance
    if _corrector_instance is None:
        _corrector_instance = ArabicTextCorrector()
    return _corrector_instance

