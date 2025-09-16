"""
Academic Paper Retrieval System
Searches and retrieves papers from arXiv and Semantic Scholar
"""
import requests
import arxiv
import time
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from urllib.parse import quote

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Paper:
    """Represents an academic paper"""
    id: str
    title: str
    authors: List[str]
    abstract: str
    pdf_url: Optional[str]
    published_date: str
    source: str  # 'arxiv' or 'semantic_scholar'
    venue: Optional[str] = None
    citation_count: int = 0
    doi: Optional[str] = None

class AcademicRetriever:
    """Retrieves academic papers from multiple sources"""
    
    def __init__(self):
        self.arxiv_client = arxiv.Client()
        self.semantic_scholar_base_url = "https://api.semanticscholar.org/graph/v1"
        
    def search_papers(self, topic: str, max_papers: int = 20) -> List[Paper]:
        """
        Search for papers across multiple academic databases
        """
        logger.info(f"Searching for papers on topic: {topic}")
        
        all_papers = []
        
        # Search arXiv
        try:
            arxiv_papers = self._search_arxiv(topic, max_papers // 2)
            all_papers.extend(arxiv_papers)
            logger.info(f"Retrieved {len(arxiv_papers)} papers from arXiv")
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
        
        # Search Semantic Scholar
        try:
            ss_papers = self._search_semantic_scholar(topic, max_papers // 2)
            all_papers.extend(ss_papers)
            logger.info(f"Retrieved {len(ss_papers)} papers from Semantic Scholar")
        except Exception as e:
            logger.error(f"Error searching Semantic Scholar: {e}")
        
        # Remove duplicates and rank by relevance
        unique_papers = self._deduplicate_papers(all_papers)
        ranked_papers = self._rank_papers_by_relevance(unique_papers, topic)
        
        logger.info(f"Total unique papers found: {len(ranked_papers)}")
        return ranked_papers[:max_papers]
    
    def _search_arxiv(self, topic: str, max_results: int) -> List[Paper]:
        """Search arXiv for papers"""
        # First search by recency to get recent papers
        search_recent = arxiv.Search(
            query=topic,
            max_results=max_results // 2,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        # Then search by relevance for comprehensive coverage
        search_relevant = arxiv.Search(
            query=topic,
            max_results=max_results // 2,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        
        # Collect papers from both searches
        for search in [search_recent, search_relevant]:
            for result in self.arxiv_client.results(search):
                paper = Paper(
                    id=result.entry_id,
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    pdf_url=result.pdf_url,
                    published_date=result.published.strftime("%Y-%m-%d"),
                    source="arxiv",
                    venue=result.primary_category
                )
                papers.append(paper)
        
        return papers
    
    def _search_semantic_scholar(self, topic: str, max_results: int) -> List[Paper]:
        """Search Semantic Scholar for papers"""
        url = f"{self.semantic_scholar_base_url}/paper/search"
        
        params = {
            'query': topic,
            'limit': max_results,
            'fields': 'paperId,title,authors,abstract,venue,year,citationCount,openAccessPdf,externalIds'
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        papers = []
        
        for item in data.get('data', []):
            # Extract PDF URL if available
            pdf_url = None
            if item.get('openAccessPdf') and item['openAccessPdf'].get('url'):
                pdf_url = item['openAccessPdf']['url']
            
            # Extract authors
            authors = []
            if item.get('authors'):
                authors = [author['name'] for author in item['authors']]
            
            paper = Paper(
                id=item['paperId'],
                title=item.get('title', ''),
                authors=authors,
                abstract=item.get('abstract', ''),
                pdf_url=pdf_url,
                published_date=str(item.get('year', '')),
                source="semantic_scholar",
                venue=item.get('venue', ''),
                citation_count=item.get('citationCount', 0),
                doi=item.get('externalIds', {}).get('DOI')
            )
            papers.append(paper)
        
        return papers
    
    def _deduplicate_papers(self, papers: List[Paper]) -> List[Paper]:
        """Remove duplicate papers based on title similarity"""
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            # Normalize title for comparison
            normalized_title = paper.title.lower().strip()
            
            # Check if we've seen a very similar title
            is_duplicate = False
            for seen_title in seen_titles:
                if self._title_similarity(normalized_title, seen_title) > 0.9:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_papers.append(paper)
                seen_titles.add(normalized_title)
        
        return unique_papers
    
    def _title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles (simple word overlap)"""
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _rank_papers_by_relevance(self, papers: List[Paper], topic: str) -> List[Paper]:
        """Rank papers by relevance to the topic"""
        topic_words = set(topic.lower().split())
        
        def calculate_relevance_score(paper: Paper) -> float:
            score = 0.0
            
            # Title relevance (highest weight)
            title_words = set(paper.title.lower().split())
            title_overlap = len(topic_words.intersection(title_words))
            score += title_overlap * 3.0
            
            # Abstract relevance
            if paper.abstract:
                abstract_words = set(paper.abstract.lower().split())
                abstract_overlap = len(topic_words.intersection(abstract_words))
                score += abstract_overlap * 1.0
            
            # Citation count bonus (for Semantic Scholar papers)
            if paper.citation_count > 0:
                score += min(paper.citation_count / 100, 2.0)  # Cap at 2.0 bonus
            
            # PDF availability bonus
            if paper.pdf_url:
                score += 1.0
            
            return score
        
        # Sort by relevance score (descending)
        papers_with_scores = [(paper, calculate_relevance_score(paper)) for paper in papers]
        papers_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [paper for paper, score in papers_with_scores]

if __name__ == "__main__":
    # Test the retriever
    retriever = AcademicRetriever()
    papers = retriever.search_papers("machine learning transformers", max_papers=5)
    
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper.title}")
        print(f"   Authors: {', '.join(paper.authors[:3])}...")
        print(f"   Source: {paper.source}")
        print(f"   PDF: {'Available' if paper.pdf_url else 'Not available'}")
