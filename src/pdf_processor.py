"""
PDF Processing Module
Downloads and processes academic PDFs to extract structured content
"""
import requests
import fitz  # PyMuPDF
import pdfplumber
import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse
import os
import tempfile
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessedPaper:
    """Represents a processed academic paper with extracted content"""
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    sections: Dict[str, str]  # section_name -> content
    full_text: str
    references: List[str]
    sentences: List[Dict]  # List of sentences with metadata
    published_date: str  # Added: Publication date from original paper
    pdf_path: Optional[str] = None

class PDFProcessor:
    """Processes academic PDFs to extract structured content"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"PDF processor initialized with temp dir: {self.temp_dir}")
    
    def process_paper(self, paper) -> Optional[ProcessedPaper]:
        """
        Process a paper object to extract structured content
        """
        if not paper.pdf_url:
            logger.warning(f"No PDF URL for paper: {paper.title}")
            return None
        
        try:
            # Download PDF
            pdf_path = self._download_pdf(paper.pdf_url, paper.id)
            if not pdf_path:
                return None
            
            # Extract content
            content = self._extract_content(pdf_path)
            if not content:
                return None
            
            # Parse sections
            sections = self._parse_sections(content['full_text'])
            
            # Extract sentences with metadata
            sentences = self._extract_sentences_with_metadata(content['full_text'], paper)
            
            processed_paper = ProcessedPaper(
                paper_id=paper.id,
                title=paper.title,
                authors=paper.authors,
                abstract=paper.abstract or content.get('abstract', ''),
                sections=sections,
                full_text=content['full_text'],
                references=content.get('references', []),
                sentences=sentences,
                published_date=paper.published_date,  # Copy publication date from original paper
                pdf_path=pdf_path
            )
            
            logger.info(f"Successfully processed paper: {paper.title[:50]}...")
            return processed_paper
            
        except Exception as e:
            logger.error(f"Error processing paper {paper.title}: {e}")
            return None
    
    def _download_pdf(self, pdf_url: str, paper_id: str) -> Optional[str]:
        """Download PDF from URL"""
        try:
            # Clean the paper_id for filename
            safe_id = re.sub(r'[^\w\-_.]', '_', paper_id)
            pdf_path = os.path.join(self.temp_dir, f"{safe_id}.pdf")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(
                pdf_url, 
                headers=headers,
                timeout=Config.PDF_DOWNLOAD_TIMEOUT,
                stream=True
            )
            response.raise_for_status()
            
            # Check file size
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > Config.MAX_PDF_SIZE_MB * 1024 * 1024:
                logger.warning(f"PDF too large: {content_length} bytes")
                return None
            
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded PDF: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Error downloading PDF from {pdf_url}: {e}")
            return None
    
    def _extract_content(self, pdf_path: str) -> Optional[Dict]:
        """Extract text content from PDF"""
        try:
            content = {
                'full_text': '',
                'abstract': '',
                'references': []
            }
            
            # Try with PyMuPDF first (faster)
            try:
                doc = fitz.open(pdf_path)
                full_text = ""
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text()
                    full_text += text + "\n"
                
                doc.close()
                content['full_text'] = full_text
                
            except Exception as e:
                logger.warning(f"PyMuPDF failed, trying pdfplumber: {e}")
                
                # Fallback to pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    full_text = ""
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            full_text += text + "\n"
                    
                    content['full_text'] = full_text
            
            # Extract abstract if not already available
            if not content['abstract']:
                content['abstract'] = self._extract_abstract(content['full_text'])
            
            # Extract references
            content['references'] = self._extract_references(content['full_text'])
            
            return content
            
        except Exception as e:
            logger.error(f"Error extracting content from {pdf_path}: {e}")
            return None
    
    def _extract_abstract(self, full_text: str) -> str:
        """Extract abstract from full text"""
        # Look for abstract section
        abstract_patterns = [
            r'ABSTRACT\s*\n(.*?)(?=\n\s*(?:INTRODUCTION|1\.|I\.|Keywords|Key words))',
            r'Abstract\s*\n(.*?)(?=\n\s*(?:Introduction|1\.|I\.|Keywords|Key words))',
            r'abstract\s*\n(.*?)(?=\n\s*(?:introduction|1\.|I\.|keywords|key words))'
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # Clean up the abstract
                abstract = re.sub(r'\s+', ' ', abstract)
                if len(abstract) > 50:  # Reasonable abstract length
                    return abstract
        
        return ""
    
    def _extract_references(self, full_text: str) -> List[str]:
        """Extract references from full text"""
        references = []
        
        # Look for references section
        ref_patterns = [
            r'REFERENCES\s*\n(.*?)(?=\n\s*(?:APPENDIX|Appendix|\Z))',
            r'References\s*\n(.*?)(?=\n\s*(?:Appendix|\Z))',
            r'BIBLIOGRAPHY\s*\n(.*?)(?=\n\s*(?:APPENDIX|Appendix|\Z))'
        ]
        
        for pattern in ref_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
            if match:
                ref_section = match.group(1)
                # Split references by common patterns
                ref_lines = re.split(r'\n(?=\[\d+\]|\d+\.|\[.*?\])', ref_section)
                references = [ref.strip() for ref in ref_lines if ref.strip() and len(ref.strip()) > 20]
                break
        
        return references[:50]  # Limit to 50 references
    
    def _parse_sections(self, full_text: str) -> Dict[str, str]:
        """Parse text into sections"""
        sections = {}
        
        # Common academic paper sections
        section_patterns = {
            'introduction': r'(?:INTRODUCTION|Introduction|1\.\s*INTRODUCTION|1\.\s*Introduction)',
            'methodology': r'(?:METHODOLOGY|Methodology|METHOD|Method|METHODS|Methods)',
            'results': r'(?:RESULTS|Results|FINDINGS|Findings)',
            'discussion': r'(?:DISCUSSION|Discussion|ANALYSIS|Analysis)',
            'conclusion': r'(?:CONCLUSION|Conclusion|CONCLUSIONS|Conclusions)',
            'related_work': r'(?:RELATED WORK|Related Work|BACKGROUND|Background)',
            'evaluation': r'(?:EVALUATION|Evaluation|EXPERIMENTS|Experiments)'
        }
        
        # Find section boundaries
        section_boundaries = []
        for section_name, pattern in section_patterns.items():
            matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
            for match in matches:
                section_boundaries.append((match.start(), section_name, match.group()))
        
        # Sort by position in text
        section_boundaries.sort(key=lambda x: x[0])
        
        # Extract section content
        for i, (start_pos, section_name, section_title) in enumerate(section_boundaries):
            # Find the end position (start of next section or end of text)
            if i + 1 < len(section_boundaries):
                end_pos = section_boundaries[i + 1][0]
            else:
                end_pos = len(full_text)
            
            # Extract section content
            section_content = full_text[start_pos:end_pos]
            # Remove the section title from content
            section_content = re.sub(re.escape(section_title), '', section_content, count=1)
            section_content = section_content.strip()
            
            if section_content and len(section_content) > 50:
                sections[section_name] = section_content
        
        return sections
    
    def _extract_sentences_with_metadata(self, full_text: str, paper) -> List[Dict]:
        """Extract sentences with metadata for citation mapping"""
        sentences = []
        
        # Simple sentence splitting (can be improved with NLTK)
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        raw_sentences = re.split(sentence_pattern, full_text)
        
        for i, sentence in enumerate(raw_sentences):
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 1000:  # Filter reasonable sentences
                sentence_data = {
                    'id': f"{paper.id}_{i}",
                    'text': sentence,
                    'paper_id': paper.id,
                    'paper_title': paper.title,
                    'authors': paper.authors,
                    'source': paper.source,
                    'position': i
                }
                sentences.append(sentence_data)
        
        logger.info(f"Extracted {len(sentences)} sentences from paper")
        return sentences

if __name__ == "__main__":
    # Test the processor
    from academic_retriever import AcademicRetriever, Paper
    
    # Create a test paper
    test_paper = Paper(
        id="test",
        title="Test Paper",
        authors=["Test Author"],
        abstract="Test abstract",
        pdf_url="https://arxiv.org/pdf/1706.03762.pdf",  # Attention is All You Need
        published_date="2017",
        source="arxiv"
    )
    
    processor = PDFProcessor()
    processed = processor.process_paper(test_paper)
    
    if processed:
        print(f"Processed paper: {processed.title}")
        print(f"Sections found: {list(processed.sections.keys())}")
        print(f"Total sentences: {len(processed.sentences)}")
        print(f"Full text length: {len(processed.full_text)} characters")
