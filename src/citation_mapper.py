"""
Citation Mapping System
Maps claims to exact sentences in source papers for precise citations
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Citation:
    """Represents a citation with supporting evidence"""
    sentence_id: str
    text: str
    paper_id: str
    paper_title: str
    authors: List[str]
    similarity_score: float
    citation_format: str
    section: Optional[str] = None

class CitationMapper:
    """Maps claims to supporting sentences for precise citations"""
    
    def __init__(self):
        logger.info("Loading sentence transformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.citation_database = {}
        self.sentence_embeddings = {}
        
    def build_citation_database(self, processed_papers: List) -> None:
        """Build searchable database of sentences from processed papers"""
        logger.info(f"Building citation database from {len(processed_papers)} papers...")
        
        all_sentences = []
        sentence_metadata = []
        
        for paper in processed_papers:
            for sentence_data in paper.sentences:
                all_sentences.append(sentence_data['text'])
                sentence_metadata.append({
                    'id': sentence_data['id'],
                    'paper_id': paper.paper_id,
                    'paper_title': paper.title,
                    'authors': paper.authors,
                    'section': self._detect_section(sentence_data['text'], paper.sections),
                    'citation_format': self._format_citation(paper)
                })
        
        if not all_sentences:
            logger.warning("No sentences found in processed papers")
            return
        
        # Generate embeddings for all sentences
        logger.info(f"Generating embeddings for {len(all_sentences)} sentences...")
        embeddings = self.model.encode(all_sentences, show_progress_bar=True)
        
        # Store in database
        for i, (sentence, metadata) in enumerate(zip(all_sentences, sentence_metadata)):
            self.citation_database[metadata['id']] = {
                'text': sentence,
                'embedding': embeddings[i],
                **metadata
            }
        
        logger.info(f"Citation database built with {len(self.citation_database)} sentences")
    
    def find_supporting_citations(self, claim: str, top_k: int = 3, min_similarity: float = None) -> List[Citation]:
        """Find sentences that support a given claim"""
        if not self.citation_database:
            logger.warning("Citation database is empty")
            return []
        
        min_similarity = min_similarity or Config.CITATION_SIMILARITY_THRESHOLD
        
        # Generate embedding for the claim
        claim_embedding = self.model.encode([claim])
        
        # Calculate similarities
        similarities = []
        for sentence_id, data in self.citation_database.items():
            similarity = cosine_similarity(claim_embedding, [data['embedding']])[0][0]
            if similarity >= min_similarity:
                similarities.append((sentence_id, similarity, data))
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Create Citation objects
        citations = []
        for sentence_id, similarity, data in similarities[:top_k]:
            citation = Citation(
                sentence_id=sentence_id,
                text=data['text'],
                paper_id=data['paper_id'],
                paper_title=data['paper_title'],
                authors=data['authors'],
                similarity_score=similarity,
                citation_format=data['citation_format'],
                section=data.get('section')
            )
            citations.append(citation)
        
        return citations
    
    def create_cited_claim(self, original_claim: str, citations: List[Citation]) -> str:
        """Create a claim with integrated citations"""
        if not citations:
            return original_claim
        
        # Sort citations by similarity score
        citations.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Create citation text
        citation_texts = []
        for i, citation in enumerate(citations):
            # Use different citation formats
            if len(citations) == 1:
                citation_text = citation.citation_format
            else:
                citation_text = citation.citation_format
            citation_texts.append(citation_text)
        
        # Integrate citations into the claim
        if len(citations) == 1:
            cited_claim = f"{original_claim} {citation_texts[0]}"
        else:
            citations_str = "; ".join(citation_texts)
            cited_claim = f"{original_claim} ({citations_str})"
        
        return cited_claim
    
    def get_citation_details(self, citations: List[Citation]) -> Dict:
        """Get detailed information about citations for references section"""
        citation_details = {}
        
        for citation in citations:
            key = citation.paper_id
            if key not in citation_details:
                citation_details[key] = {
                    'title': citation.paper_title,
                    'authors': citation.authors,
                    'citation_format': citation.citation_format,
                    'supporting_sentences': []
                }
            
            citation_details[key]['supporting_sentences'].append({
                'text': citation.text[:200] + "..." if len(citation.text) > 200 else citation.text,
                'section': citation.section,
                'similarity': citation.similarity_score
            })
        
        return citation_details
    
    def _detect_section(self, sentence: str, sections: Dict[str, str]) -> Optional[str]:
        """Detect which section a sentence belongs to"""
        for section_name, section_content in sections.items():
            if sentence in section_content:
                return section_name
        return None
    
    def _format_citation(self, paper) -> str:
        """Format citation in academic style"""
        # Get first author's last name
        if paper.authors:
            first_author = paper.authors[0].split()[-1]  # Get last name
            if len(paper.authors) == 1:
                author_part = first_author
            elif len(paper.authors) == 2:
                second_author = paper.authors[1].split()[-1]
                author_part = f"{first_author} & {second_author}"
            else:
                author_part = f"{first_author} et al."
        else:
            author_part = "Unknown"
        
        # Extract year from paper (assuming it's available)
        # Extract year from paper's published_date
        year = "2023"  # Default fallback
        if hasattr(paper, 'published_date') and paper.published_date:
            try:
                year = paper.published_date.split('-')[0]
            except:
                year = "2023"
        
        return f"({author_part}, {year})"
    
    def analyze_citation_coverage(self, claims: List[str]) -> Dict:
        """Analyze how well claims are supported by citations"""
        coverage_stats = {
            'total_claims': len(claims),
            'claims_with_citations': 0,
            'average_similarity': 0.0,
            'coverage_by_paper': {}
        }
        
        total_similarity = 0.0
        
        for claim in claims:
            citations = self.find_supporting_citations(claim, top_k=1)
            if citations:
                coverage_stats['claims_with_citations'] += 1
                total_similarity += citations[0].similarity_score
                
                # Track coverage by paper
                paper_id = citations[0].paper_id
                if paper_id not in coverage_stats['coverage_by_paper']:
                    coverage_stats['coverage_by_paper'][paper_id] = {
                        'title': citations[0].paper_title,
                        'claims_supported': 0
                    }
                coverage_stats['coverage_by_paper'][paper_id]['claims_supported'] += 1
        
        if coverage_stats['claims_with_citations'] > 0:
            coverage_stats['average_similarity'] = total_similarity / coverage_stats['claims_with_citations']
        
        coverage_stats['coverage_percentage'] = (
            coverage_stats['claims_with_citations'] / coverage_stats['total_claims'] * 100
            if coverage_stats['total_claims'] > 0 else 0
        )
        
        return coverage_stats

class ClaimExtractor:
    """Extract claims from generated content for citation mapping"""
    
    @staticmethod
    def extract_claims(text: str) -> List[str]:
        """Extract factual claims from text"""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Filter out non-factual sentences
            if ClaimExtractor._is_factual_claim(sentence):
                claims.append(sentence)
        
        return claims
    
    @staticmethod
    def _is_factual_claim(sentence: str) -> bool:
        """Determine if a sentence contains a factual claim"""
        # Skip very short sentences
        if len(sentence.split()) < 5:
            return False
        
        # Skip questions
        if sentence.strip().endswith('?'):
            return False
        
        # Skip transition sentences
        transition_words = ['however', 'moreover', 'furthermore', 'therefore', 'thus', 'hence']
        if any(sentence.lower().startswith(word) for word in transition_words):
            return True  # These often contain important claims
        
        # Look for factual indicators
        factual_indicators = [
            'research shows', 'studies indicate', 'evidence suggests',
            'findings reveal', 'results demonstrate', 'analysis shows',
            'data indicates', 'experiments show', 'observations suggest'
        ]
        
        sentence_lower = sentence.lower()
        if any(indicator in sentence_lower for indicator in factual_indicators):
            return True
        
        # Include sentences with numbers/percentages (likely factual)
        if re.search(r'\b\d+%|\b\d+\.\d+\b|\b\d+,\d+\b', sentence):
            return True
        
        # Default to including most sentences (better to over-include)
        return len(sentence.split()) >= 8

if __name__ == "__main__":
    # Test the citation mapper
    from pdf_processor import ProcessedPaper
    
    # Create test data
    test_sentences = [
        {
            'id': 'test_1',
            'text': 'Machine learning models require large amounts of training data to achieve good performance.',
            'paper_id': 'paper1',
            'paper_title': 'Deep Learning Fundamentals',
            'authors': ['John Smith', 'Jane Doe'],
            'source': 'arxiv',
            'position': 0
        }
    ]
    
    test_paper = type('MockPaper', (), {
        'paper_id': 'paper1',
        'title': 'Deep Learning Fundamentals',
        'authors': ['John Smith', 'Jane Doe'],
        'sentences': test_sentences,
        'sections': {'introduction': 'This paper discusses machine learning...'}
    })()
    
    mapper = CitationMapper()
    mapper.build_citation_database([test_paper])
    
    # Test citation finding
    claim = "Training data is essential for machine learning performance"
    citations = mapper.find_supporting_citations(claim)
    
    print(f"Found {len(citations)} citations for claim: {claim}")
    for citation in citations:
        print(f"  - {citation.text[:100]}... (similarity: {citation.similarity_score:.3f})")
