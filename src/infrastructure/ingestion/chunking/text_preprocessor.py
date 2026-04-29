"""Text preprocessing utilities for document preparation.
cleans and normalizes text before chunking and embedding."""

import re
import unicodedata
from typing import Dict, Any


class TextPreprocessor:
    """Preprocessor for cleaning and normalizing text before chunking and embedding."""
    
    @staticmethod
    def remove_layout_noise(text: str) -> str:
        """
        Remove common layout / navigation artefacts from extracted documents
        (e.g. page headers, URLs, navigation links).
        """
        cleaned_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append(line)
                continue

            # Heuristics for noise:
            # - pure URLs
            if re.match(r'https?://', stripped):
                continue
            # - "Page X of Y"
            if re.search(r'\bPage\s+\d+\s+of\s+\d+\b', stripped, flags=re.IGNORECASE):
                continue
            # - navigation labels like "zurück weiter"
            if 'zurück weiter' in stripped.lower():
                continue
            # - "Nichtamtliches Inhaltsverzeichnis"
            if 'nichtamtliches inhaltsverzeichnis' in stripped.lower():
                continue
           

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Normalize whitespace characters.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        # Replace all whitespace characters (tabs, newlines, etc.) with spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Normalize Unicode characters (NFKC normalization).
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized Unicode
        """
        # NFKC: Compatibility decomposition, followed by canonical composition
        # This normalizes similar characters (e.g., different types of quotes)
        text = unicodedata.normalize('NFKC', text)
        return text
    
    @staticmethod
    def remove_control_characters(text: str) -> str:
        """
        Remove control characters except newlines and tabs.
        
        Args:
            text: Input text
            
        Returns:
            Text without control characters
        """
        # Keep newlines, tabs, and carriage returns for structure
        # Remove other control characters
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t\r')
        return text
    
    @staticmethod
    def normalize_quotes(text: str) -> str:
        """
        Normalize different types of quotes to standard quotes.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized quotes
        """
        # Replace various quote types with standard quotes
        replacements = [
            ("\u201c", '"'),  # Left double quotation mark
            ("\u201d", '"'),  # Right double quotation mark
            ("\u2018", "'"),  # Left single quotation mark
            ("\u2019", "'"),  # Right single quotation mark
            ("\u201e", '"'),  # Double low-9 quotation mark
            ("\u201a", "'"),  # Single low-9 quotation mark
            ("\u00ab", '"'),  # Left-pointing double angle quotation mark
            ("\u00bb", '"'),  # Right-pointing double angle quotation mark
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text
    
    @staticmethod
    def normalize_dashes(text: str) -> str:
        """
        Normalize different types of dashes to standard dashes.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized dashes
        """
        # Replace em dashes and en dashes with standard hyphens
        text = text.replace('—', '-')  # Em dash
        text = text.replace('–', '-')    # En dash
        return text
    
    @staticmethod
    def fix_pdf_encoding_errors(text: str) -> str:
        """
        Fix common PDF encoding errors, especially from pypdf extraction.
        
        Handles:
        - CID references like (cid:79) -> ü
        - Common encoding mistakes: fßrdert -> fördert, Ma°nahmen -> Maßnahmen
        - Windows-1252/ISO-8859-1 misinterpretations
        
        Args:
            text: Input text with potential encoding errors
            
        Returns:
            Text with corrected encoding errors
        """
        # Step 1: Fix CID references (Character ID mappings)
        # Common CID mappings for German characters
        cid_mappings = {
            r'\(cid:79\)': 'ü',  # ü
            r'\(cid:246\)': 'ö',  # ö
            r'\(cid:228\)': 'ä',  # ä
            r'\(cid:223\)': 'ß',  # ß
            r'\(cid:220\)': 'Ü',  # Ü
            r'\(cid:214\)': 'Ö',  # Ö
            r'\(cid:196\)': 'Ä',  # Ä
        }
        
        for pattern, replacement in cid_mappings.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Step 2: Fix common encoding mistakes from PDF extraction
        # These are patterns where characters were misinterpreted during PDF parsing
        encoding_fixes = {
            # Common mistakes: ß instead of ö, ° instead of ß
            'fßrdert': 'fördert',
            'fßr': 'für',
            'Ma°nahmen': 'Maßnahmen',
            'Ma°nahme': 'Maßnahme',
            'Behßrden': 'Behörden',
            'Behßrde': 'Behörde',
            'Vorkehrungen fßr': 'Vorkehrungen für',
            'fßr die': 'für die',
            'fßr den': 'für den',
            'fßr das': 'für das',
            'fßr eine': 'für eine',
            'fßr einen': 'für einen',
            'fßr ein': 'für ein',
            'Gefßrdert': 'Gefördert',
            'gefßrdert': 'gefördert',
            'Fßrderung': 'Förderung',
            'Fßrderung': 'Förderung',
            'Fßrdermßglichkeit': 'Fördermöglichkeit',
        }
        
        # Apply fixes (case-insensitive where appropriate)
        for wrong, correct in encoding_fixes.items():
            # Case-insensitive replacement
            text = re.sub(re.escape(wrong), correct, text, flags=re.IGNORECASE)
        
        # Step 3: Fix common character substitutions
        # Only replace ° where context strongly suggests it should be ß.
        text = re.sub(r'Ma°', 'Maß', text)
        text = re.sub(r'ma°', 'maß', text)
        text = re.sub(r'Maß°', 'Maß', text)  # Fix double issues
        
        return text
    
    @staticmethod
    def clean_text(text: str, preserve_structure: bool = True) -> str:
        """
        Comprehensive text cleaning.
        
        Args:
            text: Input text
            preserve_structure: If True, preserve paragraph breaks (double newlines)
            
        Returns:
            Cleaned text
        """
        # Step 1: Remove control characters
        text = TextPreprocessor.remove_control_characters(text)

        # Step 2: Remove obvious layout / navigation noise (PDF/HTML artifacts)
        text = TextPreprocessor.remove_layout_noise(text)
        
        # Step 3: Normalize Unicode
        text = TextPreprocessor.normalize_unicode(text)
        
        # Step 4: Normalize quotes
        text = TextPreprocessor.normalize_quotes(text)
        
        # Step 5: Normalize dashes
        text = TextPreprocessor.normalize_dashes(text)
        
        # Step 6: Fix PDF encoding errors (CID references, encoding mistakes)
        text = TextPreprocessor.fix_pdf_encoding_errors(text)
        
        if preserve_structure:
            # Preserve paragraph breaks (double newlines)
            # First, normalize multiple newlines to double newlines
            text = re.sub(r'\n{3,}', '\n\n', text)
            # Normalize whitespace within paragraphs (but keep paragraph breaks)
            paragraphs = text.split('\n\n')
            cleaned_paragraphs = []
            for para in paragraphs:
                # Normalize whitespace within paragraph
                para = re.sub(r'[ \t]+', ' ', para)
                para = para.strip()
                if para:  # Only add non-empty paragraphs
                    cleaned_paragraphs.append(para)
            text = '\n\n'.join(cleaned_paragraphs)
        else:
            # Normalize all whitespace
            text = TextPreprocessor.normalize_whitespace(text)
        
        return text.strip()
    
    @staticmethod
    def preprocess(
        text: str,
        clean: bool = True,
        preserve_structure: bool = True,
        normalize_unicode: bool = True,
        normalize_quotes: bool = True,
        normalize_dashes: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive preprocessing pipeline.
        
        Args:
            text: Input text
            clean: Whether to clean the text
            preserve_structure: Whether to preserve paragraph breaks
            normalize_unicode: Whether to normalize Unicode
            normalize_quotes: Whether to normalize quotes
            normalize_dashes: Whether to normalize dashes
            
        Returns:
            Dictionary with processed text and metadata
        """
        original_length = len(text)
        
        if clean:
            text = TextPreprocessor.clean_text(text, preserve_structure=preserve_structure)
        elif normalize_unicode:
            text = TextPreprocessor.normalize_unicode(text)
        
        if normalize_quotes:
            text = TextPreprocessor.normalize_quotes(text)
        
        if normalize_dashes:
            text = TextPreprocessor.normalize_dashes(text)
        
        # Calculate statistics
        word_count = len(text.split())
        char_count = len(text)
        
        return {
            'text': text,
            'metadata': {
                'original_length': original_length,
                'processed_length': char_count,
                'word_count': word_count,
                'compression_ratio': char_count / original_length if original_length > 0 else 1.0
            }
        }


__all__ = ["TextPreprocessor"]

