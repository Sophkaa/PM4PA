"""Document processing pipeline for ingestion.
processes a document by parsing it via DocumentParser, preprocessing it via TextPreprocessor, and chunking it
before embedding and storing it in the vector store."""

import re
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from ..api.openwebui_client import OpenWebUIClient
from ..vector_store.chroma_store import ChromaVectorStore
from .parsers.document_parser import DocumentParser
from .chunking.text_preprocessor import TextPreprocessor


class DocumentProcessor:
    """Process and ingest documents into the vector store."""
    
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        api_client: OpenWebUIClient,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        min_chunk_size: int = 100,
        max_chunk_size: int = 700,
        enable_preprocessing: bool = True
    ):
        self.vector_store = vector_store
        self.api_client = api_client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.enable_preprocessing = enable_preprocessing
        self.preprocessor = TextPreprocessor() if enable_preprocessing else None
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks (character-based, for backward compatibility).
        Ensures chunks are split at word boundaries.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            if end >= len(text):
                # Last chunk - take remaining text
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break
            
            # Find word boundary near end position
            chunk_text = text[start:end]
            last_space = chunk_text.rfind(' ')
            last_newline = chunk_text.rfind('\n')
            boundary = max(last_space, last_newline)
            
            # Use boundary if it's not too far back (at least 50% of chunk size)
            if boundary > self.chunk_size * 0.5:
                end = start + boundary
                chunk = text[start:end].strip()
            else:
                # No good boundary found, use original end but try to find next space
                next_space = text.find(' ', end)
                if next_space != -1 and next_space < end + 50:  # Only if space is close
                    end = next_space
                chunk = text[start:end].strip()
            
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
        
        return chunks
    
    def _truncate_at_word_boundary(self, text: str, max_length: int) -> str:
        """
        Truncate text at word boundary to avoid splitting words.
        
        Args:
            text: Text to truncate
            max_length: Maximum length in characters
            
        Returns:
            Truncated text ending at word boundary
        """
        if len(text) <= max_length:
            return text
        
        # Find the last whitespace character before max_length
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        last_newline = truncated.rfind('\n')
        last_tab = truncated.rfind('\t')
        boundary = max(last_space, last_newline, last_tab)
        
        # If we found a reasonable boundary (at least 50% of max_length), use it
        if boundary > max_length * 0.5:
            return text[:boundary].rstrip()
        
        # Fallback: search backwards for any whitespace
        for i in range(max_length - 1, max(0, max_length // 2), -1):
            if text[i].isspace():
                return text[:i].rstrip()
        
        # Last resort: return truncated text (should be very rare)
        return truncated.rstrip()
    
    def _ensure_starts_at_word_boundary(self, text: str) -> str:
        """
        Ensure text starts at word boundary by removing leading partial word if any.
        
        Args:
            text: Text to check
            
        Returns:
            Text starting at word boundary
        """
        if not text:
            return text
        
        # If starts with whitespace, it's already at word boundary
        if text[0].isspace():
            return text.lstrip()
        
        # If starts with alphanumeric, find first whitespace and start from there
        # This removes any leading partial word
        for i in range(len(text)):
            if text[i].isspace():
                return text[i:].lstrip()
        
        # If no whitespace found, return as is (should be rare - single word)
        return text
    
    def _get_overlap_text(self, text: str, target_size: int) -> str:
        """
        Extract overlap text from the end of a chunk.
        Collects sentences from the end until target_size is reached.
        
        Args:
            text: Text to extract overlap from
            target_size: Target size for overlap in characters
            
        Returns:
            Overlap text (up to target_size characters)
        """
        if target_size <= 0 or not text:
            return ""
        
        # Split into sentences (preserving sentence endings)
        sentences = re.split(r'([.!?]+\s+)', text)
        # Recombine sentences with their endings
        combined_sentences = []
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                combined_sentences.append(sentences[i] + sentences[i + 1])
            elif i < len(sentences):
                combined_sentences.append(sentences[i])
        
        if not combined_sentences:
            # If no sentences found, truncate at word boundary
            if len(text) > target_size:
                # Take from the end, but truncate at word boundary
                remaining_text = text[-target_size * 2:]  # Take more to find word boundary
                return self._truncate_at_word_boundary(remaining_text, target_size)
            return text
        
        # Collect sentences from the end until we reach target_size
        overlap_parts = []
        current_size = 0
        
        for sentence in reversed(combined_sentences):
            if current_size + len(sentence) <= target_size:
                overlap_parts.insert(0, sentence)
                current_size += len(sentence)
            else:
                # If adding the full sentence would exceed target, take a portion at word boundary
                remaining = target_size - current_size
                if remaining > 0:
                    # Take from the end of the sentence, but truncate at word boundary
                    sentence_portion = sentence[-remaining * 2:]  # Take more to find word boundary
                    truncated_portion = self._truncate_at_word_boundary(sentence_portion, remaining)
                    if truncated_portion:
                        overlap_parts.insert(0, truncated_portion)
                break
        
        overlap_text = "".join(overlap_parts)
        
        # Ensure overlap_text is within target_size and ends at word boundary
        if len(overlap_text) > target_size:
            overlap_text = self._truncate_at_word_boundary(overlap_text, target_size)
        
        # Ensure overlap_text starts at word boundary
        overlap_text = self._ensure_starts_at_word_boundary(overlap_text)
        
        # Fallback: if we still don't have enough, take from the end at word boundary
        if len(overlap_text) < target_size and len(text) > target_size:
            remaining_text = text[-target_size * 2:]  # Take more to find word boundary
            overlap_text = self._truncate_at_word_boundary(remaining_text, target_size)
            overlap_text = self._ensure_starts_at_word_boundary(overlap_text)
        
        return overlap_text
    
    def _create_chunk_dict(self, text: str, page_number: int, heading: Optional[str], chapter: Optional[str]) -> Dict[str, Any]:
        """
        Create a chunk dictionary with metadata.
        
        Args:
            text: Chunk text
            page_number: Page number
            heading: Optional heading
            chapter: Optional chapter
            
        Returns:
            Chunk dictionary
        """
        return {
            'text': text,
            'page_number': page_number,
            'heading': heading,
            'chapter': chapter
        }
    
    def _add_text_with_spacing(self, current: str, new_text: str, separator: str = " ") -> str:
        """
        Add text to current text with proper spacing.
        
        Args:
            current: Current text
            new_text: Text to add
            separator: Separator to use (default: space)
            
        Returns:
            Combined text with proper spacing
        """
        if not current:
            return new_text
        if not current.endswith((' ', '\n', '\t')):
            return current + separator + new_text
        return current + new_text
    
    def _split_large_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a chunk that exceeds max_chunk_size into smaller chunks.
        Handles tables specially by splitting at table row boundaries.
        
        Args:
            chunk: Chunk dictionary with 'text' and metadata
            
        Returns:
            List of smaller chunk dictionaries
        """
        text = chunk['text']
        if len(text) <= self.max_chunk_size:
            return [chunk]
        
        # Check if chunk contains a table
        table_pattern = r'(\[TABELLE auf Seite \d+\]\n)(.*?)(?=\n\n|\n\[|$)'
        table_match = re.search(table_pattern, text, re.DOTALL)
        
        if table_match:
            # Special handling for chunks with tables
            return self._split_chunk_with_table(chunk, table_match)
        
        # Split by sentences (normal text)
        sentences = re.split(r'([.!?]+\s+)', text)
        combined_sentences = []
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                combined_sentences.append(sentences[i] + sentences[i + 1])
            elif i < len(sentences):
                combined_sentences.append(sentences[i])
        
        sub_chunks = []
        current_sub_chunk = ""
        
        for sentence in combined_sentences:
            # If sentence itself is too long, split it at word boundaries
            if len(sentence) > self.max_chunk_size:
                # Split the long sentence into words and create chunks
                words = sentence.split()
                temp_chunk = ""
                
                for word in words:
                    if len(temp_chunk) + len(word) + 1 > self.max_chunk_size and temp_chunk:
                        # Ensure chunk ends at word boundary
                        chunk_text = self._truncate_at_word_boundary(temp_chunk, self.max_chunk_size)
                        # Save current sub-chunk
                        sub_chunks.append(self._create_chunk_dict(
                            chunk_text.strip(),
                            chunk.get('page_number', 1),
                            chunk.get('heading'),
                            chunk.get('chapter')
                        ))
                        # Start new sub-chunk with overlap
                        if self.chunk_overlap > 0:
                            overlap_text = self._get_overlap_text(temp_chunk, self.chunk_overlap)
                            temp_chunk = self._add_text_with_spacing(overlap_text, word) if overlap_text else word
                        else:
                            temp_chunk = word
                        # Ensure new chunk starts at word boundary
                        temp_chunk = self._ensure_starts_at_word_boundary(temp_chunk)
                    else:
                        temp_chunk = self._add_text_with_spacing(temp_chunk, word)
                
                # Add remaining part of sentence to current_sub_chunk
                if temp_chunk:
                    if len(current_sub_chunk) + len(temp_chunk) > self.max_chunk_size and current_sub_chunk:
                        # Ensure chunk ends at word boundary
                        chunk_text = self._truncate_at_word_boundary(current_sub_chunk, self.max_chunk_size)
                        # Save current sub-chunk
                        sub_chunks.append(self._create_chunk_dict(
                            chunk_text.strip(),
                            chunk.get('page_number', 1),
                            chunk.get('heading'),
                            chunk.get('chapter')
                        ))
                        # Start new sub-chunk with overlap
                        if self.chunk_overlap > 0:
                            overlap_text = self._get_overlap_text(current_sub_chunk, self.chunk_overlap)
                            current_sub_chunk = self._add_text_with_spacing(overlap_text, temp_chunk) if overlap_text else temp_chunk
                        else:
                            current_sub_chunk = temp_chunk
                        # Ensure new chunk starts at word boundary
                        current_sub_chunk = self._ensure_starts_at_word_boundary(current_sub_chunk)
                    else:
                        current_sub_chunk = self._add_text_with_spacing(current_sub_chunk, temp_chunk)
            elif len(current_sub_chunk) + len(sentence) > self.max_chunk_size and current_sub_chunk:
                # Ensure chunk ends at word boundary
                chunk_text = self._truncate_at_word_boundary(current_sub_chunk, self.max_chunk_size)
                # Save current sub-chunk
                sub_chunks.append(self._create_chunk_dict(
                    chunk_text.strip(),
                    chunk.get('page_number', 1),
                    chunk.get('heading'),
                    chunk.get('chapter')
                ))
                # Start new sub-chunk with overlap
                if self.chunk_overlap > 0:
                    overlap_text = self._get_overlap_text(current_sub_chunk, self.chunk_overlap)
                    current_sub_chunk = self._add_text_with_spacing(overlap_text, sentence) if overlap_text else sentence
                else:
                    current_sub_chunk = sentence
                # Ensure new chunk starts at word boundary
                current_sub_chunk = self._ensure_starts_at_word_boundary(current_sub_chunk)
            else:
                current_sub_chunk = self._add_text_with_spacing(current_sub_chunk, sentence)
        
        # Add remaining sub-chunk (ensure it ends at word boundary if too long)
        if current_sub_chunk.strip():
            chunk_text = current_sub_chunk.strip()
            # If chunk is too long, truncate at word boundary
            if len(chunk_text) > self.max_chunk_size:
                chunk_text = self._truncate_at_word_boundary(chunk_text, self.max_chunk_size)
            sub_chunks.append(self._create_chunk_dict(
                chunk_text,
                chunk.get('page_number', 1),
                chunk.get('heading'),
                chunk.get('chapter')
            ))
        
        return sub_chunks if sub_chunks else [chunk]
    
    def _split_chunk_with_table(self, chunk: Dict[str, Any], table_match: re.Match) -> List[Dict[str, Any]]:
        """
        Split a chunk containing a table by splitting at table row boundaries.
        This ensures tables are split logically (by rows) rather than blindly (by characters).
        
        Args:
            chunk: Chunk dictionary with 'text' and metadata
            table_match: Regex match object for the table pattern
            
        Returns:
            List of smaller chunk dictionaries with tables split at row boundaries
        """
        text = chunk['text']
        table_start = table_match.start()
        table_end = table_match.end()
        
        # Split text into: before_table, table, after_table
        before_table = text[:table_start]
        table_marker = table_match.group(1)  # e.g. "[TABELLE auf Seite X]\n" (German marker text)
        table_content = table_match.group(2)  # Markdown table content
        
        # Split table by rows (markdown table rows are separated by newlines)
        table_rows = [row for row in table_content.split('\n') if row.strip()]
        
        if not table_rows:
            # Empty table, fall back to normal splitting
            return self._split_large_chunk(chunk)
        
        sub_chunks = []
        
        # Add text before table as separate chunk if substantial
        if before_table.strip() and len(before_table.strip()) > 50:
            # If before_table is too long, split it normally
            if len(before_table) > self.max_chunk_size:
                before_chunk = chunk.copy()
                before_chunk['text'] = before_table
                sub_chunks.extend(self._split_large_chunk(before_chunk))
            else:
                sub_chunks.append(self._create_chunk_dict(
                    before_table.strip(),
                    chunk.get('page_number', 1),
                    chunk.get('heading'),
                    chunk.get('chapter')
                ))
        
        # Split table into logical sub-tables
        # First row is typically the header (or separator line like |---|---|)
        header_row = table_rows[0] if table_rows else ""
        current_table_rows = [header_row] if header_row else []
        current_table_text = table_marker + header_row
        
        for row in table_rows[1:]:
            test_table = current_table_text + '\n' + row
            if len(test_table) > self.max_chunk_size and len(current_table_rows) > 1:
                # Save current sub-table (must have at least header + 1 data row)
                sub_chunks.append(self._create_chunk_dict(
                    current_table_text,
                    chunk.get('page_number', 1),
                    chunk.get('heading'),
                    chunk.get('chapter')
                ))
                # Start new sub-table with header
                current_table_rows = [header_row, row]
                current_table_text = table_marker + header_row + '\n' + row
            else:
                current_table_rows.append(row)
                current_table_text = test_table
        
        # Add remaining table
        if current_table_text.strip() and len(current_table_rows) > 0:
            sub_chunks.append(self._create_chunk_dict(
                current_table_text,
                chunk.get('page_number', 1),
                chunk.get('heading'),
                chunk.get('chapter')
            ))
        
        # Handle text after table
        after_table = text[table_end:]
        if after_table.strip() and len(after_table.strip()) > 50:
            # If after_table is too long, split it normally
            if len(after_table) > self.max_chunk_size:
                after_chunk = chunk.copy()
                after_chunk['text'] = after_table
                sub_chunks.extend(self._split_large_chunk(after_chunk))
            else:
                sub_chunks.append(self._create_chunk_dict(
                    after_table.strip(),
                    chunk.get('page_number', 1),
                    chunk.get('heading'),
                    chunk.get('chapter')
                ))
        
        return sub_chunks if sub_chunks else [chunk]
    
    def _identify_protected_units(self, text: str) -> List[tuple[int, int, str]]:
        """
        Identify protected units (tables and images) that should never be split.
        
        Returns list of (start_pos, end_pos, unit_type) tuples.
        Unit types: 'table' or 'image'
        """
        protected_units = []
        
        # Table blocks use markers like [TABELLE auf Seite X] followed by a markdown table
        table_pattern = r'\[TABELLE auf Seite \d+\]\n(.*?)(?=\n\n|\n\[|$)'
        for match in re.finditer(table_pattern, text, re.DOTALL):
            start = match.start()
            end = match.end()
            protected_units.append((start, end, 'table'))
        
        # Image blocks use markers like [BILD auf Seite X - OCR Text]: followed by OCR text
        image_pattern = r'\[BILD auf Seite \d+ - OCR Text\]:\n(.*?)(?=\n\n|\n\[|$)'
        for match in re.finditer(image_pattern, text, re.DOTALL):
            start = match.start()
            end = match.end()
            protected_units.append((start, end, 'image'))
        
        # Sort by start position
        protected_units.sort(key=lambda x: x[0])
        return protected_units
    
    def _is_in_protected_unit(self, position: int, protected_units: List[tuple[int, int, str]]) -> Optional[str]:
        """Check if a position is within a protected unit. Returns unit type or None."""
        for start, end, unit_type in protected_units:
            if start <= position < end:
                return unit_type
        return None
    
    def chunk_text_semantic(
        self,
        text: str,
        page_mapping: Optional[Dict[int, str]] = None,
        structure: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into semantic chunks (paragraph/sentence-based).
        - Never split a single sentence across two chunks.
        - Tables and images are atomic units (never split).
        - Overlap at sentence level (last N sentences of the previous chunk).

        Args:
            text: Text to chunk
            page_mapping: Optional mapping of page numbers to text
            structure: Optional document structure (headings, chapters)
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Identify protected units (tables and images) that must not be split
        protected_units = self._identify_protected_units(text)
        
        # 1) Split by paragraph (double newlines delimit paragraphs)
        paragraphs = re.split(r'\n\s*\n+', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Heading hierarchy from structure metadata
        heading_hierarchy = []
        if structure and 'headings' in structure:
            heading_hierarchy = structure['headings']
        
        # 2) Split paragraphs into sentences and collect metadata per sentence
        #    Each entry: (sentence_text, page, heading, chapter)
        all_sentences: List[tuple[str, int, Optional[str], Optional[str]]] = []

        current_page = 1
        current_heading: Optional[str] = None
        current_chapter: Optional[str] = None
        search_start = 0  

        for para in paragraphs:
            # Detect heading paragraphs
            for heading in heading_hierarchy:
                if para == heading.get('text', ''):
                    current_heading = para
                    if heading.get('level', 1) == 1:
                        current_chapter = para
                    break
            
            # Map paragraph to page using page_mapping
            if page_mapping:
                para_start_pos = text.find(para, search_start)
                if para_start_pos == -1:
                    para_start_pos = text.find(para)
                if para_start_pos != -1:
                    search_start = para_start_pos + len(para)
                
                cumulative_length = 0
                for page_num in sorted(page_mapping.keys()):
                    page_text = page_mapping[page_num]
                    page_start = cumulative_length
                    page_end = cumulative_length + len(page_text)
                    if page_start <= para_start_pos < page_end:
                        current_page = page_num
                        break
                    cumulative_length = page_end
            
            # Split paragraph into sentences (regex also handles newlines)
            raw_parts = re.split(r'([.!?]+[\s\n]+)', para)
            combined_sentences: List[str] = []
            for i in range(0, len(raw_parts), 2):
                if i + 1 < len(raw_parts):
                    combined = raw_parts[i] + raw_parts[i + 1]
                else:
                    combined = raw_parts[i]
                combined = combined.strip()
                if combined:
                    combined_sentences.append(combined)

            # If sentence split found nothing, treat whole paragraph as one "sentence"
            if not combined_sentences and para:
                combined_sentences = [para.strip()]

            # Append sentences with metadata (full sentences, no truncation)
            for sent in combined_sentences:
                sent_clean = sent.strip()
                if sent_clean:
                    all_sentences.append((sent_clean, current_page, current_heading, current_chapter))

        if not all_sentences:
            return []

        # 3) Build chunks from sentences with sentence-level overlap
        #    Tables and images remain atomic protected units
        chunks: List[Dict[str, Any]] = []
        i = 0

        # How many sentences to carry into the next chunk as overlap
        sentence_overlap = 0
        if self.chunk_overlap > 0:
            # Rough heuristic: 1–2 sentences overlap depending on chunk_overlap
            sentence_overlap = 2 if self.chunk_overlap >= 200 else 1
        else:
            sentence_overlap = 0

        n_sent = len(all_sentences)
        while i < n_sent:
            current_text = ""
            chunk_page = all_sentences[i][1]
            chunk_heading = all_sentences[i][2]
            chunk_chapter = all_sentences[i][3]
            start_idx = i
            contains_protected_unit = False

            while i < n_sent:
                sent_text, page, heading, chapter = all_sentences[i]
                
                # Find position of this sentence in original text to check if it's in a protected unit
                if current_text:
                    # Approximate position: find sentence in text after current_text
                    search_start = text.find(current_text) + len(current_text) if current_text in text else 0
                else:
                    search_start = text.find(sent_text) if sent_text in text else 0
                
                sent_pos_in_text = text.find(sent_text, search_start) if sent_text in text else search_start
                unit_type = self._is_in_protected_unit(sent_pos_in_text, protected_units)
                
                # If this sentence is part of a protected unit, we need to include the entire unit
                if unit_type:
                    contains_protected_unit = True
                    # Find the full protected unit text
                    for start, end, ut in protected_units:
                        if start <= sent_pos_in_text < end:
                            # Get the full protected unit text
                            protected_text = text[start:end].strip()
                            
                            # Add context BEFORE the protected unit (if available)
                            # Look for sentences/paragraphs before the protected unit
                            context_before = ""
                            if i > 0:
                                # Get previous sentences for context (up to 2 sentences or 1 paragraph)
                                context_sentences = []
                                context_start_idx = max(0, i - 2)  # Up to 2 sentences before
                                
                                # Check if we should include more context (previous paragraph)
                                for j in range(context_start_idx, i):
                                    if j < len(all_sentences):
                                        prev_sent_text = all_sentences[j][0]
                                        # Check if this sentence is not already in current_text
                                        if not current_text or prev_sent_text not in current_text:
                                            context_sentences.append(prev_sent_text)
                                
                                if context_sentences:
                                    context_before = " ".join(context_sentences)
                            
                            # Build the chunk: context before + protected unit
                            if context_before and context_before not in current_text:
                                if current_text:
                                    current_text = self._add_text_with_spacing(current_text, context_before, " ")
                                else:
                                    current_text = context_before
                            
                            # Add the protected unit
                            if current_text:
                                current_text = self._add_text_with_spacing(current_text, protected_text, "\n\n")
                            else:
                                current_text = protected_text
                            
                            # Skip all sentences that are part of this protected unit
                            protected_unit_end_idx = i
                            while i < n_sent:
                                next_sent_text = all_sentences[i][0]
                                next_sent_pos = text.find(next_sent_text, sent_pos_in_text) if next_sent_text in text else sent_pos_in_text
                                if next_sent_pos >= end:
                                    protected_unit_end_idx = i
                                    break
                                i += 1
                            
                            # Add context AFTER the protected unit (if available and space allows)
                            # Look for sentences after the protected unit
                            if i < n_sent and len(current_text) < self.chunk_size * 1.5:  # Allow some extra space for context
                                context_after_sentences = []
                                context_end_idx = min(n_sent, i + 2)  # Up to 2 sentences after
                                
                                for j in range(i, context_end_idx):
                                    if j < len(all_sentences):
                                        next_sent_text = all_sentences[j][0]
                                        context_after_sentences.append(next_sent_text)
                                
                                if context_after_sentences:
                                    context_after = " ".join(context_after_sentences)
                                    # Only add if it doesn't make the chunk too large
                                    if len(current_text) + len(context_after) < self.max_chunk_size * 1.2:
                                        current_text = self._add_text_with_spacing(current_text, context_after, " ")
                                        # Update i to skip the context sentences we just added
                                        i = context_end_idx
                            
                            # Update metadata from the last sentence we processed
                            if i > 0 and i <= n_sent:
                                chunk_page = all_sentences[min(i-1, n_sent-1)][1]
                                if heading and not chunk_heading:
                                    chunk_heading = heading
                                if chapter and not chunk_chapter:
                                    chunk_chapter = chapter
                            
                            break
                    
                    # After processing protected unit with context, check if we should continue or break
                    # Protected unit with context is complete, break to create chunk
                    # This ensures the protected unit stays together with its context
                    break
                
                # Normal sentence processing (not in protected unit)
                # If adding the next sentence exceeds chunk_size and we already have text → end chunk
                # Exception: if we already contain a protected unit, keep it intact
                if current_text and len(current_text) + len(sent_text) > self.chunk_size:
                    # If we have a protected unit, we must keep it even if it exceeds chunk_size
                    if contains_protected_unit:
                        # Protected unit is complete, break to create chunk
                        break
                    else:
                        # Normal case: break before adding this sentence
                        break

                if not current_text:
                    current_text = sent_text.strip() 
                else:
                    current_text = self._add_text_with_spacing(current_text, sent_text, " ")

                # Refresh metadata from the latest sentence
                chunk_page = page
                if heading and not chunk_heading:
                    chunk_heading = heading
                if chapter and not chunk_chapter:
                    chunk_chapter = chapter

                i += 1

            if not current_text:
                # Fallback — should be rare
                i += 1
                continue

            # Ensure chunk does not exceed max_chunk_size
            # Exception: chunks that contain a protected unit may be larger
            chunk_text = current_text.strip()
            if len(chunk_text) > self.max_chunk_size and not contains_protected_unit:
                chunk_text = self._truncate_at_word_boundary(chunk_text, self.max_chunk_size)

            chunks.append(self._create_chunk_dict(
                chunk_text,
                chunk_page,
                chunk_heading,
                chunk_chapter
            ))

            # Sentence-based overlap: step back a few sentences
            # Protected units: overlap is already covered by integrated context
            # Normal chunks: apply standard overlap
            if sentence_overlap > 0 and i < n_sent and not contains_protected_unit:
                new_start = max(start_idx, i - sentence_overlap)
                if new_start <= start_idx:
                    new_start = i  # Guard against infinite loops
                i = new_start
            # Note: no extra overlap for protected units; context is already in the chunk

        # 4) Post-process very small/large chunks as before
        # Protected-unit chunks are usually not split unless extremely large (> 2x max_chunk_size)
        final_chunks: List[Dict[str, Any]] = []
        for chunk in chunks:
            chunk_text = chunk['text']
            # Check if chunk contains protected units
            has_protected_unit = bool(self._identify_protected_units(chunk_text))
            
            should_split_chunk = len(chunk_text) > self.max_chunk_size and (
                not has_protected_unit or len(chunk_text) > self.max_chunk_size * 2
            )
            if should_split_chunk:
                # Split normal oversized chunks; protected-unit chunks only when extremely large (> 2x max).
                final_chunks.extend(self._split_large_chunk(chunk))
            else:
                # Keep chunk as-is (especially if it contains protected units and is reasonable size)
                final_chunks.append(chunk)
        
        # Merge chunks that are below min_chunk_size
        merged_chunks: List[Dict[str, Any]] = []
        j = 0
        while j < len(final_chunks):
            current = final_chunks[j]
            
            if len(current['text']) < self.min_chunk_size and j < len(final_chunks) - 1:
                next_chunk = final_chunks[j + 1]
                combined_text = current['text'] + "\n\n" + next_chunk['text']
                
                if len(combined_text) <= self.max_chunk_size:
                    merged_chunk = {
                        'text': combined_text,
                        'page_number': current.get('page_number', 1),
                        'heading': current.get('heading') or next_chunk.get('heading'),
                        'chapter': current.get('chapter') or next_chunk.get('chapter')
                    }
                    merged_chunks.append(merged_chunk)
                    j += 2
                    continue
            
            merged_chunks.append(current)
            j += 1
        
        return merged_chunks
    
    async def process_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        use_semantic_chunking: bool = True
    ) -> List[str]:
        """
        Process a document: chunk it, generate embeddings, and store.
        
        Args:
            text: Document text
            metadata: Optional metadata for the document
            use_semantic_chunking: Whether to use semantic chunking (default: True)
            
        Returns:
            List of chunk IDs
        """
        if metadata is None:
            metadata = {}
        
        # Preprocess the text
        if self.enable_preprocessing and self.preprocessor:
            preprocessed = self.preprocessor.preprocess(
                text,
                clean=True,
                preserve_structure=True
            )
            text = preprocessed['text']
            # Add preprocessing metadata
            preprocessing_metadata = preprocessed.get('metadata', {})
            metadata = {**metadata, **preprocessing_metadata}
        
        # Chunk the document
        if use_semantic_chunking:
            chunks_data = self.chunk_text_semantic(text)
            chunks = [chunk['text'] for chunk in chunks_data]
        else:
            chunks = self.chunk_text(text)
            chunks_data = [{'text': chunk} for chunk in chunks]
        
        if not chunks:
            return []
        
        # Generate embeddings for all chunks
        embeddings = await self.api_client.get_embeddings(chunks)
        
        # Prepare metadata for each chunk
        chunk_metadatas = []
        chunk_ids = []
        
        for i, (chunk, chunk_data) in enumerate(zip(chunks, chunks_data)):
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            
            # Add semantic chunking metadata if available
            if use_semantic_chunking and isinstance(chunk_data, dict):
                if 'page_number' in chunk_data:
                    chunk_metadata['page_number'] = chunk_data['page_number']
                if 'heading' in chunk_data and chunk_data['heading']:
                    chunk_metadata['heading'] = chunk_data['heading']
                if 'chapter' in chunk_data and chunk_data['chapter']:
                    chunk_metadata['chapter'] = chunk_data['chapter']
            
            chunk_metadatas.append(chunk_metadata)
            chunk_ids.append(f"{metadata.get('doc_id', 'doc')}_chunk_{i}")
        
        # Add to vector store
        self.vector_store.add_documents(
            documents=chunks,
            embeddings=embeddings,
            metadatas=chunk_metadatas,
            ids=chunk_ids
        )
        
        return chunk_ids
    
    async def process_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        use_semantic_chunking: bool = True,
        return_chunks: bool = False
    ) -> Union[List[str], Dict[str, Any]]:
        """
        Process a file: parse it, chunk it, generate embeddings, and store.
        
        Args:
            file_path: Path to the file to process
            metadata: Optional metadata for the document (will be merged with extracted metadata)
            use_semantic_chunking: Whether to use semantic chunking (default: True)
            return_chunks: If True, return chunks and metadata (for testing). Default: False (returns only chunk IDs)
            
        Returns:
            If return_chunks=False: List of chunk IDs
            If return_chunks=True: Dictionary with 'chunk_ids', 'chunks', 'chunks_data', 'chunk_metadatas'
        """
        file_path = Path(file_path)
        
        # Parse the file
        doc_content = DocumentParser.parse_file(str(file_path))
        
        # Preprocess the text
        if self.enable_preprocessing and self.preprocessor:
            preprocessed = self.preprocessor.preprocess(
                doc_content.text,
                clean=True,
                preserve_structure=True
            )
            doc_content.text = preprocessed['text']
            # Add preprocessing metadata
            preprocessing_metadata = preprocessed.get('metadata', {})
        else:
            preprocessing_metadata = {}
        
        # Merge extracted metadata with provided metadata
        if metadata is None:
            metadata = {}
        
        # Add document-level metadata
        # Use file_path from metadata if provided (e.g., relative path), otherwise use absolute path
        # This ensures the file_path used in the filter matches what's stored
        file_path_metadata = metadata.get('file_path', str(file_path))
        
        merged_metadata = {
            **doc_content.metadata,
            **preprocessing_metadata,
            **metadata,  # metadata comes last to preserve user-provided values (like relative file_path)
            'file_path': file_path_metadata,  # Explicitly set to ensure consistency
            'file_type': doc_content.file_type,
            'document_title': doc_content.metadata.get('title', Path(file_path).stem),
        }
        
        # Chunk the document with semantic chunking
        if use_semantic_chunking:
            chunks_data = self.chunk_text_semantic(
                doc_content.text,
                page_mapping=doc_content.page_mapping,
                structure=doc_content.structure
            )
            chunks = [chunk['text'] for chunk in chunks_data]
        else:
            chunks = self.chunk_text(doc_content.text)
            chunks_data = [{'text': chunk} for chunk in chunks]
        
        if not chunks:
            if return_chunks:
                return {
                    'chunk_ids': [],
                    'chunks': [],
                    'chunks_data': [],
                    'chunk_metadatas': []
                }
            return []
        
        # Generate embeddings for all chunks
        embeddings = await self.api_client.get_embeddings(chunks)
        
        # Prepare metadata for each chunk
        chunk_metadatas = []
        chunk_ids = []
        
        for i, (chunk, chunk_data) in enumerate(zip(chunks, chunks_data)):
            chunk_metadata = {
                **merged_metadata,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            
            # Add semantic chunking metadata
            if isinstance(chunk_data, dict):
                if 'page_number' in chunk_data:
                    chunk_metadata['page_number'] = chunk_data['page_number']
                if 'heading' in chunk_data and chunk_data['heading']:
                    chunk_metadata['heading'] = chunk_data['heading']
                if 'chapter' in chunk_data and chunk_data['chapter']:
                    chunk_metadata['chapter'] = chunk_data['chapter']
            
            chunk_metadatas.append(chunk_metadata)
            doc_id = merged_metadata.get('doc_id', Path(file_path).stem)
            chunk_ids.append(f"{doc_id}_chunk_{i}")
        
        # Add to vector store
        self.vector_store.add_documents(
            documents=chunks,
            embeddings=embeddings,
            metadatas=chunk_metadatas,
            ids=chunk_ids
        )
        
        # Return chunks if requested (for testing)
        if return_chunks:
            return {
                'chunk_ids': chunk_ids,
                'chunks': chunks,
                'chunks_data': chunks_data,
                'chunk_metadatas': chunk_metadatas
            }
        
        return chunk_ids

