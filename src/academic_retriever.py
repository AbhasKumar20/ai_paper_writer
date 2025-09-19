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
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
        
        # Initialize semantic similarity model
        logger.info("🧠 Loading semantic similarity model...")
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Semantic similarity model loaded")
        
    def search_papers(self, topic: str, max_papers: int = 20) -> List[Paper]:
        """
        Search for papers across multiple academic databases with detailed tracking
        
        Args:
            topic: The research topic to search for
            max_papers: Maximum number of papers to retrieve
            
        Returns:
            List of Paper objects ranked by relevance
        """
        logger.info(f"🔍 Starting comprehensive paper search for topic: '{topic}'")
        logger.info(f"📊 Target: {max_papers} papers from multiple sources")
        
        all_papers = []
        retrieval_stats = {
            'topic': topic,
            'sources': {},
            'total_retrieved': 0,
            'duplicates_removed': 0,
            'final_count': 0
        }
        
        # Search arXiv with expanded pool for better ranking
        try:
            logger.info("📚 Searching arXiv database...")
            arxiv_target = max_papers * 3  # Fetch 3x more papers for better selection
            arxiv_papers = self._search_arxiv(topic, arxiv_target)
            all_papers.extend(arxiv_papers)
            retrieval_stats['sources']['arxiv'] = {
                'count': len(arxiv_papers),
                'papers': [{'title': p.title[:80] + '...', 'year': p.published_date[:4], 'id': p.id} for p in arxiv_papers[:5]]
            }
            logger.info(f"✅ arXiv: Retrieved {len(arxiv_papers)} papers (target: {arxiv_target})")
        except Exception as e:
            logger.error(f"❌ arXiv search failed: {e}")
            retrieval_stats['sources']['arxiv'] = {'count': 0, 'error': str(e)}
        
        # Search Semantic Scholar with expanded pool
        try:
            logger.info("🎓 Searching Semantic Scholar database...")
            ss_target = max_papers * 2  # Fetch 2x more papers for better selection
            ss_papers = self._search_semantic_scholar_with_retry(topic, ss_target)
            all_papers.extend(ss_papers)
            retrieval_stats['sources']['semantic_scholar'] = {
                'count': len(ss_papers),
                'papers': [{'title': p.title[:80] + '...', 'year': p.published_date[:4], 'citations': p.citation_count} for p in ss_papers[:5]]
            }
            logger.info(f"✅ Semantic Scholar: Retrieved {len(ss_papers)} papers (target: {ss_target})")
        except Exception as e:
            logger.warning(f"⚠️  Semantic Scholar search failed: {e}")
            logger.info("   📚 Continuing with arXiv papers only...")
            retrieval_stats['sources']['semantic_scholar'] = {'count': 0, 'error': str(e)}
        
        retrieval_stats['total_retrieved'] = len(all_papers)
        logger.info(f"📋 Total papers retrieved: {len(all_papers)}")
        
        # Remove duplicates and rank by relevance
        logger.info("🔄 Removing duplicates and ranking by semantic relevance...")
        unique_papers = self._deduplicate_papers(all_papers)
        retrieval_stats['duplicates_removed'] = len(all_papers) - len(unique_papers)
        
        logger.info(f"🎯 Ranking {len(unique_papers)} papers to select top {max_papers}...")
        ranked_papers = self._rank_papers_by_relevance(unique_papers, topic)
        final_papers = ranked_papers[:max_papers]
        retrieval_stats['final_count'] = len(final_papers)
        
        # Log detailed retrieval statistics
        self._log_retrieval_statistics(retrieval_stats, final_papers)
        
        return final_papers
    
    def _search_arxiv(self, topic: str, max_results: int) -> List[Paper]:
        """Search arXiv for papers with improved search strategy"""
        logger.info(f"   🔍 Searching arXiv with multiple strategies...")
        
        # Strategy 1: Relevance search (primary - 70% of allocation)
        search_relevant = arxiv.Search(
            query=topic,
            max_results=int(max_results * 0.7),  # 70% for relevance
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        # Strategy 2: Recent papers search (secondary - 30% of allocation)  
        search_recent = arxiv.Search(
            query=topic,
            max_results=int(max_results * 0.3),  # 30% for recency
            sort_by=arxiv.SortCriterion.SubmittedDate
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
        """Rank papers by semantic relevance to the topic using transformer embeddings"""
        logger.info(f"🧠 Computing semantic relevance for {len(papers)} papers...")
        
        def calculate_semantic_relevance_score(paper: Paper) -> float:
            try:
                # Create embeddings for topic, title, and abstract
                topic_embedding = self.similarity_model.encode([topic])
                
                # Title semantic similarity (highest weight)
                title_text = paper.title or ""
                if title_text:
                    title_embedding = self.similarity_model.encode([title_text])
                    title_similarity = cosine_similarity(topic_embedding, title_embedding)[0][0]
                else:
                    title_similarity = 0.0
                
                # Abstract semantic similarity  
                abstract_text = paper.abstract or ""
                if abstract_text and len(abstract_text.strip()) > 10:
                    # Truncate very long abstracts for efficiency
                    abstract_truncated = abstract_text[:1000] if len(abstract_text) > 1000 else abstract_text
                    abstract_embedding = self.similarity_model.encode([abstract_truncated])
                    abstract_similarity = cosine_similarity(topic_embedding, abstract_embedding)[0][0]
                else:
                    abstract_similarity = 0.0
                
                # Combined semantic score (weighted)
                semantic_score = (title_similarity * 3.0 + abstract_similarity * 2.0)
                
                # Citation impact bonus (scaled and capped)
                citation_bonus = 0.0
                if paper.citation_count and paper.citation_count > 0:
                    # Logarithmic scaling for citation impact
                    citation_bonus = min(np.log10(paper.citation_count + 1) * 0.5, 2.0)
                
                # Recency bonus (prefer recent papers within reason)
                recency_bonus = 0.0
                if paper.published_date:
                    try:
                        year = int(paper.published_date[:4])
                        current_year = 2025
                        if year >= 2020:  # Recent papers get bonus
                            recency_bonus = min((year - 2020) * 0.1, 0.5)
                    except (ValueError, IndexError):
                        pass
                
                # PDF availability bonus
                pdf_bonus = 0.3 if paper.pdf_url else 0.0
                
                # Final score
                total_score = semantic_score + citation_bonus + recency_bonus + pdf_bonus
                
                return total_score
                
            except Exception as e:
                logger.warning(f"Error calculating semantic relevance for paper '{paper.title[:50]}...': {e}")
                # Fallback to simple word matching
                return self._calculate_fallback_score(paper, topic)
        
        # Calculate scores for all papers
        papers_with_scores = []
        for paper in papers:
            score = calculate_semantic_relevance_score(paper)
            papers_with_scores.append((paper, score))
            
        # Sort by relevance score (descending)
        papers_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Log top scores for debugging
        logger.info("🏆 Top 3 semantic relevance scores:")
        for i, (paper, score) in enumerate(papers_with_scores[:3], 1):
            logger.info(f"   {i}. Score: {score:.3f} - {paper.title[:60]}...")
        
        return [paper for paper, score in papers_with_scores]
    
    def _calculate_fallback_score(self, paper: Paper, topic: str) -> float:
        """Fallback scoring method using simple word matching"""
        topic_words = set(topic.lower().split())
        score = 0.0
        
        # Title word matching
        if paper.title:
            title_words = set(paper.title.lower().split())
            title_overlap = len(topic_words.intersection(title_words))
            score += title_overlap * 1.0
        
        # Abstract word matching
        if paper.abstract:
            abstract_words = set(paper.abstract.lower().split())
            abstract_overlap = len(topic_words.intersection(abstract_words))
            score += abstract_overlap * 0.5
        
        # Basic bonuses
        if paper.citation_count and paper.citation_count > 0:
            score += min(paper.citation_count / 1000, 1.0)
        if paper.pdf_url:
            score += 0.2
            
        return score
    
    def _log_retrieval_statistics(self, stats: dict, final_papers: List[Paper]) -> None:
        """Log detailed statistics about paper retrieval"""
        logger.info("📊 PAPER RETRIEVAL STATISTICS")
        logger.info("=" * 50)
        logger.info(f"🎯 Topic: {stats['topic']}")
        logger.info(f"📈 Total Retrieved: {stats['total_retrieved']}")
        logger.info(f"🔄 Duplicates Removed: {stats['duplicates_removed']}")
        logger.info(f"✅ Final Count: {stats['final_count']}")
        
        # Source breakdown
        logger.info("\n📚 SOURCE BREAKDOWN:")
        for source, data in stats['sources'].items():
            if 'error' in data:
                logger.info(f"  ❌ {source.replace('_', ' ').title()}: {data['error']}")
            else:
                logger.info(f"  ✅ {source.replace('_', ' ').title()}: {data['count']} papers")
                
                # Show sample papers from each source
                if data['papers']:
                    logger.info(f"     📋 Sample papers:")
                    for i, paper in enumerate(data['papers'][:3], 1):
                        year_info = f" ({paper['year']})" if 'year' in paper else ""
                        citation_info = f" - {paper['citations']} citations" if 'citations' in paper else ""
                        logger.info(f"       {i}. {paper['title']}{year_info}{citation_info}")
        
        # Final paper details
        logger.info(f"\n🏆 TOP {min(5, len(final_papers))} SELECTED PAPERS:")
        for i, paper in enumerate(final_papers[:5], 1):
            year = paper.published_date[:4] if paper.published_date else "Unknown"
            source_icon = "📚" if paper.source == "arxiv" else "🎓"
            logger.info(f"  {i}. {source_icon} [{paper.source.upper()}] {paper.title[:100]}... ({year})")
            if paper.citation_count > 0:
                logger.info(f"     📊 Citations: {paper.citation_count}")
        
        # Quality metrics
        if final_papers:
            avg_year = self._calculate_average_year(final_papers)
            total_citations = sum(p.citation_count for p in final_papers)
            arxiv_count = sum(1 for p in final_papers if p.source == "arxiv")
            ss_count = sum(1 for p in final_papers if p.source == "semantic_scholar")
            
            logger.info(f"\n📊 QUALITY METRICS:")
            logger.info(f"  📅 Average Publication Year: {avg_year:.1f}")
            logger.info(f"  📈 Total Citations: {total_citations}")
            logger.info(f"  📚 arXiv Papers: {arxiv_count} ({arxiv_count/len(final_papers)*100:.1f}%)")
            logger.info(f"  🎓 Semantic Scholar Papers: {ss_count} ({ss_count/len(final_papers)*100:.1f}%)")
        
        logger.info("=" * 50)
    
    def _calculate_average_year(self, papers: List[Paper]) -> float:
        """Calculate average publication year of papers"""
        years = []
        for paper in papers:
            try:
                year = int(paper.published_date[:4]) if paper.published_date else 2020
                years.append(year)
            except (ValueError, IndexError):
                years.append(2020)  # Default year for invalid dates
        return sum(years) / len(years) if years else 2020.0
    
    def _search_semantic_scholar_with_retry(self, topic: str, max_results: int, max_retries: int = 3) -> List[Paper]:
        """Search Semantic Scholar with retry logic for rate limiting"""
        import time
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                    logger.info(f"   ⏳ Rate limited. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                
                return self._search_semantic_scholar(topic, max_results)
                
            except Exception as e:
                if "429" in str(e):  # Rate limit error
                    if attempt == max_retries - 1:
                        logger.warning(f"   ❌ Max retries reached. Semantic Scholar unavailable.")
                        return []
                    continue
                else:
                    # Other errors, don't retry
                    raise e
        
        return []


if __name__ == "__main__":
    # Test the retriever
    retriever = AcademicRetriever()
    papers = retriever.search_papers("machine learning transformers", max_papers=5)
    
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper.title}")
        print(f"   Authors: {', '.join(paper.authors[:3])}...")
        print(f"   Source: {paper.source}")
        print(f"   PDF: {'Available' if paper.pdf_url else 'Not available'}")
