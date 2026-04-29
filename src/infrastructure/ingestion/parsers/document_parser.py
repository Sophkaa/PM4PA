"""Document parser for multiple file formats.
extracts text, metadata, and structure from a document."""

import logging

# Reduce pypdf warnings for malformed PDFs (e.g. "Ignoring wrong pointing object")
logging.getLogger("pypdf._reader").setLevel(logging.ERROR)
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class TableContent(BaseModel):
    """Extracted table content."""
    page_number: int
    table_index: int  # Index of table on the page
    data: List[List[str]]  # Table data as rows of cells
    markdown: str  # Table formatted as Markdown
    text: str  # Table formatted as plain text


class ImageContent(BaseModel):
    """Extracted image content."""
    page_number: int
    image_index: int  # Index of image on the page
    ocr_text: Optional[str] = None  # Text extracted via OCR
    description: Optional[str] = None  # Optional description of the image


class PageContent(BaseModel):
    """Content of a single page."""
    page_number: int
    text: str
    metadata: Dict[str, Any] = {}
    tables: List[TableContent] = []
    images: List[ImageContent] = []


class DocumentContent(BaseModel):
    """Extracted document content with metadata."""
    text: str
    metadata: Dict[str, Any] = {}
    pages: List[PageContent] = []
    structure: Dict[str, Any] = {}
    page_mapping: Dict[int, str] = {}
    file_path: str
    file_type: str
    tables: List[TableContent] = []  # All tables from the document
    images: List[ImageContent] = []  # All images from the document


class DocumentParser:
    """Parser for multiple document formats."""
    
    @staticmethod
    def detect_file_type(file_path: str) -> str:
        """
        Detect file type from extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File type (pdf, docx, txt, md, html, etc.)
        """
        ext = Path(file_path).suffix.lower()
        type_mapping = {
            '.pdf': 'pdf',
            '.docx': 'docx',
            '.doc': 'docx',  # Treat .doc as docx
            '.txt': 'txt',
            '.md': 'md',
            '.markdown': 'md',
            '.html': 'html',
            '.htm': 'html',
        }
        return type_mapping.get(ext, 'txt')
    
    @staticmethod
    def _table_to_markdown(table_data: List[List[str]]) -> str:
        """
        Convert table data to Markdown format.
        
        Args:
            table_data: List of rows, each row is a list of cell strings
            
        Returns:
            Markdown-formatted table string
        """
        if not table_data or not table_data[0]:
            return ""
        
        # Use first row as header
        header = table_data[0]
        rows = table_data[1:] if len(table_data) > 1 else []
        
        # Create markdown table
        lines = []
        lines.append("| " + " | ".join(str(cell) for cell in header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        
        for row in rows:
            # Pad row if necessary
            padded_row = row + [""] * (len(header) - len(row))
            lines.append("| " + " | ".join(str(cell) for cell in padded_row[:len(header)]) + " |")
        
        return "\n".join(lines)
    
    @staticmethod
    def _table_to_text(table_data: List[List[str]]) -> str:
        """
        Convert table data to plain text format.
        
        Args:
            table_data: List of rows, each row is a list of cell strings
            
        Returns:
            Plain text representation of the table
        """
        if not table_data:
            return ""
        
        lines = []
        for row in table_data:
            lines.append(" | ".join(str(cell) for cell in row))
        
        return "\n".join(lines)

    @staticmethod
    def _remove_duplicate_table_text(text: str, tables: List[TableContent]) -> str:
        """
        Remove table text from normal text if it's already present as [TABELLE] blocks.
        Uses higher similarity threshold (0.9) and length checks to avoid false positives.

        Args:
            text: Full text that may contain duplicate table content
            tables: List of extracted TableContent objects

        Returns:
            Text with duplicate table content removed
        """
        if not tables:
            return text

        # Split text into paragraphs
        paragraphs = text.split('\n\n')
        cleaned_paragraphs = []

        for para in paragraphs:
            # Skip if this paragraph is a table marker
            if para.strip().startswith('[TABELLE auf Seite'):
                cleaned_paragraphs.append(para)
                continue

            # Check if paragraph is similar to any table text
            is_duplicate = False
            para_clean = para.strip()

            # Only check substantial paragraphs (min 100 chars to avoid false positives)
            if len(para_clean) < 100:
                cleaned_paragraphs.append(para)
                continue

            for table in tables:
                # Compare with table text representation
                table_text = table.text.strip()
                table_markdown = table.markdown.strip()

                # Only check if paragraph and table are similar in length (within 50%)
                if len(para_clean) < len(table_text) * 0.5 or len(para_clean) > len(table_text) * 1.5:
                    continue

                # Check similarity with both text and markdown versions
                similarity_text = SequenceMatcher(None, para_clean, table_text).ratio()
                similarity_markdown = SequenceMatcher(None, para_clean, table_markdown).ratio()

                # Higher threshold (0.9) to avoid false positives with normal text
                if similarity_text > 0.9 or similarity_markdown > 0.9:
                    is_duplicate = True
                    break

            if not is_duplicate:
                cleaned_paragraphs.append(para)

        return '\n\n'.join(cleaned_paragraphs)

    @staticmethod
    def _remove_duplicate_paragraphs(text: str, min_block_size: int = 200, similarity_threshold: float = 0.85) -> str:
        """
        Remove duplicate paragraphs from text based on similarity.

        Args:
            text: Text to deduplicate
            min_block_size: Minimum size of paragraph to consider for deduplication (chars)
            similarity_threshold: Similarity threshold (0.0-1.0) above which paragraphs are considered duplicates

        Returns:
            Text with duplicate paragraphs removed
        """
        # Split into paragraphs (by double newlines)
        paragraphs = text.split('\n\n')

        if len(paragraphs) <= 1:
            return text

        # Filter out very short paragraphs and table/image markers
        substantial_paragraphs = []
        for idx, para in enumerate(paragraphs):
            para_clean = para.strip()
            # Skip very short paragraphs and markers
            if (len(para_clean) >= min_block_size and
                not para_clean.startswith('[TABELLE auf Seite') and
                not para_clean.startswith('[BILD auf Seite')):
                substantial_paragraphs.append((idx, para))

        # Find duplicates by comparing each paragraph with all previous ones
        seen_indices = set()
        unique_paragraphs = []

        for idx, para in substantial_paragraphs:
            if idx in seen_indices:
                continue

            is_duplicate = False
            para_clean = para.strip()

            # Compare with all previously seen unique paragraphs
            for prev_idx, prev_para in unique_paragraphs:
                prev_clean = prev_para.strip()

                # Length check: only compare if similar length (within 30%)
                if (len(para_clean) < len(prev_clean) * 0.7 or
                    len(para_clean) > len(prev_clean) * 1.3):
                    continue

                similarity = SequenceMatcher(None, para_clean, prev_clean).ratio()

                if similarity >= similarity_threshold:
                    is_duplicate = True
                    seen_indices.add(idx)
                    break

            if not is_duplicate:
                unique_paragraphs.append((idx, para))

        # Reconstruct text: keep all paragraphs, but skip duplicates
        result_paragraphs = []
        for i, para in enumerate(paragraphs):
            # Check if this paragraph was marked as duplicate
            para_clean = para.strip()
            if len(para_clean) >= min_block_size and not para_clean.startswith('[TABELLE auf Seite') and not para_clean.startswith('[BILD auf Seite'):
                # Find if this paragraph index was in substantial_paragraphs and marked as duplicate
                found_idx = None
                for idx, _ in substantial_paragraphs:
                    if paragraphs[idx] == para:
                        found_idx = idx
                        break

                if found_idx is not None and found_idx in seen_indices:
                    continue  # Skip duplicate

            result_paragraphs.append(para)

        return '\n\n'.join(result_paragraphs)

    @staticmethod
    def _extract_tables_with_pdfplumber(file_path: str) -> Dict[int, List[TableContent]]:
        """
        Extract tables from PDF using pdfplumber.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary mapping page numbers to list of TableContent
        """
        tables_by_page = {}
        
        try:
            import pdfplumber
        except ImportError:
            # pdfplumber not available, return empty dict
            return tables_by_page
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_tables = page.extract_tables()
                    
                    if page_tables:
                        table_contents = []
                        for table_idx, table in enumerate(page_tables):
                            if table and len(table) > 0:
                                # Sanitize: pdfplumber can return None cells; TableContent requires strings
                                sanitized = [[(str(cell) if cell is not None else "") for cell in row] for row in table]
                                table_content = TableContent(
                                    page_number=page_num,
                                    table_index=table_idx,
                                    data=sanitized,
                                    markdown=DocumentParser._table_to_markdown(sanitized),
                                    text=DocumentParser._table_to_text(sanitized)
                                )
                                table_contents.append(table_content)
                        
                        if table_contents:
                            tables_by_page[page_num] = table_contents
        except Exception as e:
            logger.warning("Failed to extract tables with pdfplumber: %s", e)
        
        return tables_by_page
    
    @staticmethod
    def _extract_images_with_ocr(file_path: str) -> Dict[int, List[ImageContent]]:
        """
        Extract individual images from PDF and perform OCR if available.
        
        This method extracts actual image objects embedded in the PDF (not just
        converting pages to images). It uses pypdf to find image XObjects and
        pdfplumber as a fallback for better image detection.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary mapping page numbers to list of ImageContent
        """
        images_by_page = {}
        
        # Check if OCR is available
        ocr_available = False
        try:
            import pytesseract
            from PIL import Image
            import io
            ocr_available = True
        except ImportError:
            pass
        
        try:
            from pypdf import PdfReader
        except ImportError:
            # pypdf not available, try pdfplumber as fallback
            return DocumentParser._extract_images_with_pdfplumber(file_path, ocr_available)
        
        try:
            reader = PdfReader(file_path)
            
            for page_num, page in enumerate(reader.pages, start=1):
                image_contents = []
                
                # Method 1: Extract images using pypdf (XObject extraction)
                if '/Resources' in page and '/XObject' in page['/Resources']:
                    try:
                        xobjects = page['/Resources']['/XObject'].get_object()
                        
                        for obj_name, obj in xobjects.items():
                            if obj.get('/Subtype') == '/Image':
                                # This is an image object
                                try:
                                    # Extract image data
                                    image_data = obj.get_data()
                                    
                                    # Try to create PIL Image from the data
                                    if ocr_available:
                                        try:
                                            pil_image = Image.open(io.BytesIO(image_data))

                                            # Perform OCR on the extracted image
                                            ocr_text = pytesseract.image_to_string(pil_image, lang='deu+eng')

                                            if ocr_text.strip():
                                                image_content = ImageContent(
                                                    page_number=page_num,
                                                    image_index=len(image_contents),
                                                    ocr_text=ocr_text.strip()
                                                )
                                                image_contents.append(image_content)
                                        except Exception as e:
                                            # Image format not supported or OCR failed
                                            # Still record the image without OCR text
                                            image_content = ImageContent(
                                                page_number=page_num,
                                                image_index=len(image_contents),
                                                ocr_text=None
                                            )
                                            image_contents.append(image_content)
                                    else:
                                        # OCR not available, but still record the image
                                        image_content = ImageContent(
                                            page_number=page_num,
                                            image_index=len(image_contents),
                                            ocr_text=None
                                        )
                                        image_contents.append(image_content)
                                except Exception as e:
                                    logger.debug(
                                        "Failed to extract image %s from page %d: %s",
                                        obj_name,
                                        page_num,
                                        e,
                                    )
                                    continue
                    except Exception as e:
                        logger.debug(
                            "Failed to extract XObjects from page %d, trying pdfplumber: %s",
                            page_num,
                            e,
                        )
                
                # Method 2: Fallback to pdfplumber if pypdf didn't find images
                if not image_contents:
                    pdfplumber_images = DocumentParser._extract_images_with_pdfplumber(
                        file_path, ocr_available, page_num
                    )
                    if page_num in pdfplumber_images:
                        image_contents = pdfplumber_images[page_num]
                
                if image_contents:
                    images_by_page[page_num] = image_contents
                    
        except Exception as e:
            logger.warning("Failed to extract images from PDF: %s", e)
            # Try pdfplumber as last resort
            try:
                pdfplumber_images = DocumentParser._extract_images_with_pdfplumber(file_path, ocr_available)
                images_by_page.update(pdfplumber_images)
            except Exception as fallback_error:
                logger.debug("Fallback image extraction with pdfplumber failed: %s", fallback_error)
        
        return images_by_page
    
    @staticmethod
    def _extract_images_with_pdfplumber(
        file_path: str,
        ocr_available: bool,
        specific_page: Optional[int] = None
    ) -> Dict[int, List[ImageContent]]:
        """
        Extract images from PDF using pdfplumber as fallback method.
        
        Args:
            file_path: Path to PDF file
            ocr_available: Whether OCR is available
            specific_page: If provided, only extract from this page
            
        Returns:
            Dictionary mapping page numbers to list of ImageContent
        """
        images_by_page = {}
        
        try:
            import pdfplumber
        except ImportError:
            return images_by_page
        
        try:
            from PIL import Image
            import io
        except ImportError:
            return images_by_page
        
        try:
            with pdfplumber.open(file_path) as pdf:
                pages_to_process = [specific_page] if specific_page else range(1, len(pdf.pages) + 1)
                
                for page_num in pages_to_process:
                    if page_num < 1 or page_num > len(pdf.pages):
                        continue
                    
                    page = pdf.pages[page_num - 1]
                    image_contents = []
                    
                    # Extract images from the page
                    images = page.images
                    
                    for img_idx, img in enumerate(images):
                        try:
                            # Try to extract image data
                            # pdfplumber provides image bounding box, but we need the actual image
                            # For now, we'll use OCR on the page region if OCR is available
                            if ocr_available:
                                try:
                                    import pytesseract
                                    # Crop the page to the image region
                                    bbox = (img['x0'], img['top'], img['x1'], img['bottom'])
                                    cropped = page.crop(bbox)
                                    
                                    # Convert to image and perform OCR
                                    # Note: pdfplumber doesn't directly give us image bytes,
                                    # so we use the cropped region for OCR
                                    # This is a workaround - ideally we'd extract the actual image
                                    try:
                                        # Try to get the actual image from the page
                                        # This is a limitation of pdfplumber - it gives us metadata but not the image bytes
                                        # For now, we'll record that an image was found
                                        image_content = ImageContent(
                                            page_number=page_num,
                                            image_index=img_idx,
                                            ocr_text=None,  # Can't easily extract image bytes with pdfplumber
                                            description=f"Image at ({img['x0']:.1f}, {img['top']:.1f})"
                                        )
                                        image_contents.append(image_content)
                                    except Exception as e:
                                        logger.debug(
                                            "Failed to record image %d on page %d: %s",
                                            img_idx,
                                            page_num,
                                            e,
                                        )
                                except Exception as e:
                                    logger.debug(
                                        "Failed to process image %d on page %d: %s",
                                        img_idx,
                                        page_num,
                                        e,
                                    )
                            else:
                                # No OCR, but still record the image
                                image_content = ImageContent(
                                    page_number=page_num,
                                    image_index=img_idx,
                                    ocr_text=None,
                                    description=f"Image at ({img['x0']:.1f}, {img['top']:.1f})"
                                )
                                image_contents.append(image_content)
                        except Exception as e:
                            logger.debug(
                                "Failed to extract image %d from page %d: %s",
                                img_idx,
                                page_num,
                                e,
                            )
                            continue
                    
                    if image_contents:
                        images_by_page[page_num] = image_contents
                        
        except Exception as e:
            logger.warning("Failed to extract images with pdfplumber: %s", e)
        
        return images_by_page

    @staticmethod
    def _is_text_garbled(text: str) -> bool:
        """
        Detect if PDF text extraction produced garbled output (e.g. Identity-H encoding).

        Heuristic: German text typically has vowels in most words. Garbled text from
        broken PDF encodings often produces consonant-heavy nonsense like "NZbgvdhr".
        """
        if not text or len(text.strip()) < 50:
            return False
        # Known garbled patterns from Identity-H / broken CMap PDFs
        garbled_patterns = [
            r'\bNZbgvdhr\b', r'\bUdpvdmctmf\b', r'\bAtryZgktmf\b',
            r'\bImipZesspdsdm\b', r'\bAtödpipZesspdsdm\b', r'\bAmkZfd\b',
        ]
        for pattern in garbled_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        # Statistical: words (4+ chars) with very low vowel ratio
        vowel_re = re.compile(r'[aäeëiouöüy]', re.IGNORECASE)
        words = re.findall(r'\b[a-zA-ZäöüßÄÖÜ]{4,}\b', text)
        if len(words) < 5:
            return False
        suspicious = 0
        for w in words:
            vowels = len(vowel_re.findall(w))
            if vowels / len(w) < 0.2:  # < 20% vowels = suspicious
                suspicious += 1
        return suspicious / len(words) > 0.2  # > 20% suspicious words = garbled

    @staticmethod
    def _extract_text_via_ocr(file_path: str, num_pages: int) -> Optional[List[str]]:
        """
        Extract text from PDF by rendering pages to images and running OCR.

        Fallback when normal text extraction produces garbled output (Identity-H etc.).
        Requires: pdf2image (poppler), pytesseract (Tesseract with deu).
        """
        try:
            from pdf2image import convert_from_path
            import pytesseract
        except ImportError as e:
            logger.debug("OCR fallback not available (pdf2image/pytesseract): %s", e)
            return None
        try:
            images = convert_from_path(file_path, first_page=1, last_page=num_pages)
        except Exception as e:
            logger.warning("pdf2image failed (poppler required): %s", e)
            return None
        page_texts = []
        for img in images:
            try:
                text = pytesseract.image_to_string(img, lang='deu')
                page_texts.append(text.strip() if text else '')
            except Exception as e:
                logger.debug("Tesseract OCR failed for page: %s", e)
                page_texts.append('')
        return page_texts if any(t for t in page_texts) else None

    @staticmethod
    def _extract_pdf_outline(reader: Any) -> List[Dict[str, Any]]:
        """
        Extract PDF outline (bookmarks) as heading hierarchy.
        Returns list of {'text': str, 'level': int} in document order.
        """
        headings: List[Dict[str, Any]] = []
        outline = getattr(reader, 'outline', None) or []

        def _flatten_outline(items: List, level: int = 1) -> None:
            for item in items:
                if isinstance(item, list):
                    _flatten_outline(item, level + 1)
                else:
                    title = getattr(item, 'title', None)
                    if title and isinstance(title, str) and title.strip():
                        headings.append({'text': title.strip(), 'level': level})

        try:
            _flatten_outline(outline)
        except Exception as e:
            logger.debug("Failed to extract PDF outline: %s", e)

        return headings

    @staticmethod
    def parse_pdf(
        file_path: str,
        extract_tables: bool = True,
        extract_images: bool = True
    ) -> DocumentContent:
        """
        Parse PDF file and extract text, metadata, structure, tables, and images.
        
        Args:
            file_path: Path to PDF file
            extract_tables: Whether to extract tables using pdfplumber (default: True)
            extract_images: Whether to extract images and perform OCR (default: True)
            
        Returns:
            DocumentContent with extracted information including tables and images
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf is required for PDF parsing. Install with: pip install pypdf")
        
        reader = PdfReader(file_path)
        
        # Extract metadata
        metadata = reader.metadata or {}
        doc_metadata = {
            'title': metadata.get('/Title', ''),
            'subject': metadata.get('/Subject', ''),
            'creator': metadata.get('/Creator', ''),
            'producer': metadata.get('/Producer', ''),
            'creation_date': str(metadata.get('/CreationDate', '')),
            'modification_date': str(metadata.get('/ModDate', '')),
        }
        
        # Extract tables if requested
        tables_by_page = {}
        all_tables = []
        if extract_tables:
            tables_by_page = DocumentParser._extract_tables_with_pdfplumber(file_path)
            # Flatten all tables
            for page_tables in tables_by_page.values():
                all_tables.extend(page_tables)
        
        # Extract images if requested
        images_by_page = {}
        all_images = []
        if extract_images:
            images_by_page = DocumentParser._extract_images_with_ocr(file_path)
            # Flatten all images
            for page_images in images_by_page.values():
                all_images.extend(page_images)
        
        # Extract text per page and integrate tables/images
        # Try pdfplumber first for better encoding handling, fallback to pypdf
        pages = []
        page_mapping = {}
        full_text_parts = []
        structure = {
            'headings': DocumentParser._extract_pdf_outline(reader),
            'sections': []
        }
        
        # Try to use pdfplumber for text extraction (better encoding handling)
        use_pdfplumber = False
        pdfplumber_pages = {}
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text()
                    if page_text:
                        pdfplumber_pages[page_num] = page_text
                use_pdfplumber = len(pdfplumber_pages) > 0
        except (ImportError, Exception) as e:
            logger.debug("pdfplumber not available for text extraction, using pypdf: %s", e)
        
        for page_num, page in enumerate(reader.pages, start=1):
            # Prefer pdfplumber text if available (better encoding handling)
            if use_pdfplumber and page_num in pdfplumber_pages:
                page_text = pdfplumber_pages[page_num]
            else:
                # Fallback to pypdf
                page_text = page.extract_text()
            
            # Get tables and images for this page
            page_tables = tables_by_page.get(page_num, [])
            page_images = images_by_page.get(page_num, [])
            
            # Build enhanced page text with tables and images
            enhanced_text_parts = [page_text] if page_text and page_text.strip() else []
            
            # Add tables to page text
            for table in page_tables:
                enhanced_text_parts.append(f"\n\n[TABELLE auf Seite {page_num}]\n{table.markdown}\n")
            
            # Add image OCR text to page text
            for image in page_images:
                if image.ocr_text:
                    enhanced_text_parts.append(f"\n\n[BILD auf Seite {page_num} - OCR Text]:\n{image.ocr_text}\n")
            
            enhanced_page_text = "\n".join(enhanced_text_parts)
            
            pages.append(PageContent(
                page_number=page_num,
                text=enhanced_page_text,
                metadata={},
                tables=page_tables,
                images=page_images
            ))
            page_mapping[page_num] = enhanced_page_text
            full_text_parts.append(enhanced_page_text)
        
        full_text = '\n\n'.join(full_text_parts)

        # OCR fallback when text extraction produced garbled output (Identity-H etc.)
        if DocumentParser._is_text_garbled(full_text):
            logger = logging.getLogger(__name__)
            logger.info("Detected garbled PDF text, attempting OCR fallback")
            ocr_page_texts = DocumentParser._extract_text_via_ocr(file_path, len(reader.pages))
            if ocr_page_texts:
                # Rebuild full_text_parts, pages, page_mapping with OCR text
                for i, page_num in enumerate(range(1, len(reader.pages) + 1)):
                    ocr_text = ocr_page_texts[i] if i < len(ocr_page_texts) else ''
                    page_tables = tables_by_page.get(page_num, [])
                    page_images = images_by_page.get(page_num, [])
                    enhanced_parts = [ocr_text] if ocr_text else []
                    for table in page_tables:
                        enhanced_parts.append(f"\n\n[TABELLE auf Seite {page_num}]\n{table.markdown}\n")
                    for image in page_images:
                        if image.ocr_text:
                            enhanced_parts.append(f"\n\n[BILD auf Seite {page_num} - OCR Text]:\n{image.ocr_text}\n")
                    enhanced_page_text = "\n".join(enhanced_parts)
                    full_text_parts[i] = enhanced_page_text
                    pages[i] = PageContent(
                        page_number=page_num,
                        text=enhanced_page_text,
                        metadata={'source': 'ocr_fallback'},
                        tables=page_tables,
                        images=page_images
                    )
                    page_mapping[page_num] = enhanced_page_text
                full_text = '\n\n'.join(full_text_parts)
                logger.info("OCR fallback succeeded, using OCR-extracted text")
            else:
                logger.warning("OCR fallback failed, retaining original (garbled) text")

        # Fix PDF encoding errors (CID references, encoding mistakes)
        # This should be done early, before other processing
        try:
            from ..chunking.text_preprocessor import TextPreprocessor
            full_text = TextPreprocessor.fix_pdf_encoding_errors(full_text)
        except Exception as e:
            logger.debug("TextPreprocessor encoding fix unavailable, continuing without fix: %s", e)

        # Remove duplicate table text
        full_text = DocumentParser._remove_duplicate_table_text(full_text, all_tables)

        # Remove duplicate paragraphs
        full_text = DocumentParser._remove_duplicate_paragraphs(full_text)

        return DocumentContent(
            text=full_text,
            metadata=doc_metadata,
            pages=pages,
            structure=structure,
            page_mapping=page_mapping,
            file_path=file_path,
            file_type='pdf',
            tables=all_tables,
            images=all_images
        )
    
    @staticmethod
    def parse_docx(file_path: str) -> DocumentContent:
        """
        Parse DOCX file and extract text, metadata, and structure.
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            DocumentContent with extracted information
        """
        try:
            from docx import Document
        except ImportError:
            raise ImportError("python-docx is required for DOCX parsing. Install with: pip install python-docx")
        
        doc = Document(file_path)
        
        # Extract metadata
        core_props = doc.core_properties
        doc_metadata = {
            'title': core_props.title or '',
            'subject': core_props.subject or '',
            'keywords': core_props.keywords or '',
            'comments': core_props.comments or '',
            'creation_date': str(core_props.created) if core_props.created else '',
            'modification_date': str(core_props.modified) if core_props.modified else '',
        }
        
        # Extract text and structure
        # text_parts: full document in order (headings + body) so chunking can match headings
        text_parts = []
        headings = []
        sections = []
        current_heading = None
        current_section = []
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            
            # Check if it's a heading
            if para.style.name.startswith('Heading'):
                level = int(para.style.name.split()[-1]) if para.style.name.split()[-1].isdigit() else 1
                headings.append({
                    'text': text,
                    'level': level
                })
                if current_heading:
                    sections.append({
                        'heading': current_heading,
                        'content': '\n'.join(current_section)
                    })
                current_heading = text
                current_section = []
                text_parts.append(text)
            else:
                if current_section is not None:
                    current_section.append(text)
                text_parts.append(text)
        
        # Add last section
        if current_heading and current_section:
            sections.append({
                'heading': current_heading,
                'content': '\n'.join(current_section)
            })
        
        full_text = '\n\n'.join(text_parts)
        
        # Create page-like structure (DOCX doesn't have explicit pages)
        # Estimate pages based on content length (rough estimate: 500 words per page)
        words = len(full_text.split())
        estimated_pages = max(1, words // 500)
        
        pages = []
        page_mapping = {}
        if estimated_pages > 1:
            words_per_page = words // estimated_pages
            words_list = full_text.split()
            for page_num in range(1, estimated_pages + 1):
                start_idx = (page_num - 1) * words_per_page
                end_idx = page_num * words_per_page if page_num < estimated_pages else len(words_list)
                page_text = ' '.join(words_list[start_idx:end_idx])
                pages.append(PageContent(
                    page_number=page_num,
                    text=page_text,
                    metadata={}
                ))
                page_mapping[page_num] = page_text
        else:
            pages.append(PageContent(
                page_number=1,
                text=full_text,
                metadata={}
            ))
            page_mapping[1] = full_text
        
        return DocumentContent(
            text=full_text,
            metadata=doc_metadata,
            pages=pages,
            structure={
                'headings': headings,
                'sections': sections
            },
            page_mapping=page_mapping,
            file_path=file_path,
            file_type='docx'
        )
    
    @staticmethod
    def parse_txt(file_path: str) -> DocumentContent:
        """
        Parse plain text file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            DocumentContent with extracted information
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Simple page estimation (500 words per page)
        words = len(text.split())
        estimated_pages = max(1, words // 500)
        
        pages = []
        page_mapping = {}
        if estimated_pages > 1:
            words_per_page = words // estimated_pages
            words_list = text.split()
            for page_num in range(1, estimated_pages + 1):
                start_idx = (page_num - 1) * words_per_page
                end_idx = page_num * words_per_page if page_num < estimated_pages else len(words_list)
                page_text = ' '.join(words_list[start_idx:end_idx])
                pages.append(PageContent(
                    page_number=page_num,
                    text=page_text,
                    metadata={}
                ))
                page_mapping[page_num] = page_text
        else:
            pages.append(PageContent(
                page_number=1,
                text=text,
                metadata={}
            ))
            page_mapping[1] = text
        
        return DocumentContent(
            text=text,
            metadata={
                'title': Path(file_path).stem,
            },
            pages=pages,
            structure={},
            page_mapping=page_mapping,
            file_path=file_path,
            file_type='txt'
        )
    
    @staticmethod
    def parse_markdown(file_path: str) -> DocumentContent:
        """
        Parse Markdown file and extract text and structure.
        
        Args:
            file_path: Path to Markdown file
            
        Returns:
            DocumentContent with extracted information
        """
        try:
            import markdown
        except ImportError:
            raise ImportError("markdown is required for Markdown parsing. Install with: pip install markdown")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            md_text = f.read()
        
        # Extract headings using regex
        import re
        headings = []
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        for line in md_text.split('\n'):
            match = re.match(heading_pattern, line)
            if match:
                level = len(match.group(1))
                heading_text = match.group(2).strip()
                headings.append({
                    'text': heading_text,
                    'level': level
                })
        
        # Convert markdown to plain text (remove markdown syntax)
        md = markdown.Markdown()
        html = md.convert(md_text)
        
        # Extract plain text from HTML
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text(separator='\n\n', strip=True)
        except ImportError:
            # Fallback: simple markdown text extraction
            text = re.sub(r'^#{1,6}\s+', '', md_text, flags=re.MULTILINE)
            text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
            text = re.sub(r'\*(.+?)\*', r'\1', text)
            text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
        
        # Page estimation
        words = len(text.split())
        estimated_pages = max(1, words // 500)
        
        pages = []
        page_mapping = {}
        if estimated_pages > 1:
            words_per_page = words // estimated_pages
            words_list = text.split()
            for page_num in range(1, estimated_pages + 1):
                start_idx = (page_num - 1) * words_per_page
                end_idx = page_num * words_per_page if page_num < estimated_pages else len(words_list)
                page_text = ' '.join(words_list[start_idx:end_idx])
                pages.append(PageContent(
                    page_number=page_num,
                    text=page_text,
                    metadata={}
                ))
                page_mapping[page_num] = page_text
        else:
            pages.append(PageContent(
                page_number=1,
                text=text,
                metadata={}
            ))
            page_mapping[1] = text
        
        return DocumentContent(
            text=text,
            metadata={
                'title': Path(file_path).stem,
            },
            pages=pages,
            structure={
                'headings': headings
            },
            page_mapping=page_mapping,
            file_path=file_path,
            file_type='md'
        )
    
    @staticmethod
    def parse_html(file_path: str) -> DocumentContent:
        """
        Parse HTML file and extract text and structure.
        
        Args:
            file_path: Path to HTML file
            
        Returns:
            DocumentContent with extracted information
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("beautifulsoup4 is required for HTML parsing. Install with: pip install beautifulsoup4")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title
        title = soup.find('title')
        title_text = title.string if title else ''
        
        # Extract headings
        headings = []
        for i in range(1, 7):
            for heading in soup.find_all(f'h{i}'):
                headings.append({
                    'text': heading.get_text(strip=True),
                    'level': i
                })
        
        # Extract text (remove scripts and styles)
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text(separator='\n\n', strip=True)
        
        # Page estimation
        words = len(text.split())
        estimated_pages = max(1, words // 500)
        
        pages = []
        page_mapping = {}
        if estimated_pages > 1:
            words_per_page = words // estimated_pages
            words_list = text.split()
            for page_num in range(1, estimated_pages + 1):
                start_idx = (page_num - 1) * words_per_page
                end_idx = page_num * words_per_page if page_num < estimated_pages else len(words_list)
                page_text = ' '.join(words_list[start_idx:end_idx])
                pages.append(PageContent(
                    page_number=page_num,
                    text=page_text,
                    metadata={}
                ))
                page_mapping[page_num] = page_text
        else:
            pages.append(PageContent(
                page_number=1,
                text=text,
                metadata={}
            ))
            page_mapping[1] = text
        
        return DocumentContent(
            text=text,
            metadata={
                'title': title_text or Path(file_path).stem,
            },
            pages=pages,
            structure={
                'headings': headings
            },
            page_mapping=page_mapping,
            file_path=file_path,
            file_type='html'
        )
    
    @staticmethod
    def parse_file(file_path: str) -> DocumentContent:
        """
        Parse a file based on its type.
        
        Args:
            file_path: Path to the file
            
        Returns:
            DocumentContent with extracted information
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_type = DocumentParser.detect_file_type(file_path)
        
        parser_map = {
            'pdf': DocumentParser.parse_pdf,
            'docx': DocumentParser.parse_docx,
            'txt': DocumentParser.parse_txt,
            'md': DocumentParser.parse_markdown,
            'html': DocumentParser.parse_html,
        }
        
        parser = parser_map.get(file_type)
        if not parser:
            # Fallback to text parser
            parser = DocumentParser.parse_txt
        
        return parser(file_path)

