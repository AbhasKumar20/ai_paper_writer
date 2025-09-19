"""
Main Research Paper Generation Pipeline
Orchestrates the complete process from topic to final paper
"""
import os
import logging
import time
from typing import Optional
from datetime import datetime

from config import Config
from academic_retriever import AcademicRetriever
from pdf_processor import PDFProcessor
from citation_mapper import CitationMapper
from content_generator import ContentGenerator, PaperAssembler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchPaperGenerator:
    """Main pipeline for generating research papers"""
    
    def __init__(self):
        """Initialize all components"""
        logger.info("Initializing Research Paper Generator...")
        
        self.retriever = AcademicRetriever()
        self.pdf_processor = PDFProcessor()
        self.citation_mapper = CitationMapper()
        self.content_generator = ContentGenerator()
        self.paper_assembler = PaperAssembler()
        
        # Set up output directory
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        
        logger.info("All components initialized successfully")
    
    def generate_paper(self, topic: str, max_papers: int = None) -> str:
        """
        Generate a complete research paper for the given topic
        
        Args:
            topic: The research topic
            max_papers: Maximum number of papers to retrieve (default from config)
            
        Returns:
            Path to the generated paper file
        """
        start_time = time.time()
        max_papers = max_papers or Config.MAX_PAPERS_TO_RETRIEVE
        
        logger.info(f"Starting paper generation for topic: '{topic}'")
        logger.info(f"Target: {Config.TARGET_PAPER_LENGTH} words, {max_papers} source papers")
        
        try:
            # Step 1: Search and retrieve academic papers
            step1_start = time.time()
            logger.info("Step 1: Searching for relevant academic papers...")
            papers = self.retriever.search_papers(topic, max_papers)
            step1_time = time.time() - step1_start
            logger.info(f"Step 1 completed in {step1_time:.1f} seconds")
            
            if not papers:
                raise ValueError("No papers found for the given topic")
            
            logger.info(f"Found {len(papers)} relevant papers")
            
            # Step 2: Process PDFs and extract content
            step2_start = time.time()
            logger.info("Step 2: Processing PDFs and extracting content...")
            processed_papers = []
            
            for i, paper in enumerate(papers, 1):
                logger.info(f"Processing paper {i}/{len(papers)}: {paper.title[:60]}...")
                processed_paper = self.pdf_processor.process_paper(paper)
                
                if processed_paper:
                    processed_papers.append(processed_paper)
                else:
                    logger.warning(f"Failed to process paper: {paper.title}")
            
            if not processed_papers:
                raise ValueError("No papers could be processed successfully")
            
            logger.info(f"Successfully processed {len(processed_papers)} papers")
            step2_time = time.time() - step2_start
            logger.info(f"Step 2 completed in {step2_time:.1f} seconds")
            
            # Step 3: Build citation database
            step3_start = time.time()
            logger.info("Step 3: Building citation database...")
            self.citation_mapper.build_citation_database(processed_papers)
            self.content_generator.set_citation_mapper(self.citation_mapper)
            step3_time = time.time() - step3_start
            logger.info(f"Step 3 completed in {step3_time:.1f} seconds")
            
            # Step 4: Generate complete paper (OPTIMIZED - Single LLM call)
            step4_start = time.time()
            logger.info("Step 4: Generating complete paper with citations...")
            generated_sections = self.content_generator.generate_complete_paper(topic, processed_papers)
            
            # Create default outlines for paper assembly
            section_outlines = [
                type('SectionOutline', (), {'title': 'Introduction', 'name': 'introduction'})(),
                type('SectionOutline', (), {'title': 'Literature Review', 'name': 'literature_review'})(),
                type('SectionOutline', (), {'title': 'Current Research and Findings', 'name': 'current_research'})(),
                type('SectionOutline', (), {'title': 'Discussion and Implications', 'name': 'discussion'})(),
                type('SectionOutline', (), {'title': 'Conclusion', 'name': 'conclusion'})()
            ]
            
            for i, section_content in enumerate(generated_sections):
                section_name = section_outlines[i].title if i < len(section_outlines) else f"Section {i+1}"
                logger.info(f"{section_name}: {section_content.word_count} words "
                           f"with {len(section_content.citations_used)} citations "
                           f"({section_content.citation_coverage:.1%} coverage)")
            
            step4_time = time.time() - step4_start
            logger.info(f"Step 4 completed in {step4_time:.1f} seconds")
            
            # Step 5: Collect citation details
            step5_start = time.time()
            logger.info("Step 5: Collecting citation details...")
            all_citations = []
            for section in generated_sections:
                logger.info(f"Section citations: {section.citations_used}")
                for citation_ref in section.citations_used:
                    all_citations.append(citation_ref)
            
            logger.info(f"All citations collected: {all_citations}")
            citation_details = self._collect_citation_details(processed_papers, all_citations)
            logger.info(f"Citation details: {citation_details}")
            step5_time = time.time() - step5_start
            logger.info(f"Step 5 completed in {step5_time:.1f} seconds")
            
            # Step 6: Assemble final paper
            step6_start = time.time()
            logger.info("Step 6: Assembling final paper...")
            final_paper = self.paper_assembler.assemble_paper(
                topic, generated_sections, section_outlines, citation_details
            )
            step6_time = time.time() - step6_start
            logger.info(f"Step 6 completed in {step6_time:.1f} seconds")
            
            # Step 7: Save paper
            step7_start = time.time()
            output_path = self._save_paper(topic, final_paper)
            step7_time = time.time() - step7_start
            logger.info(f"Step 7 completed in {step7_time:.1f} seconds")
            
            # Log completion statistics
            end_time = time.time()
            total_time = end_time - start_time
            total_words = sum(section.word_count for section in generated_sections)
            
            logger.info("Paper generation completed successfully!")
            logger.info(f"Total time: {total_time:.1f} seconds")
            logger.info(f"Total words: {total_words}")
            logger.info(f"Unique citations: {len(citation_details)}")
            logger.info(f"Source papers processed: {len(processed_papers)}")
            logger.info(f"Output saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating paper: {e}")
            raise
    
    def _collect_citation_details(self, processed_papers, citation_refs) -> dict:
        """Collect detailed information about citations used"""
        citation_details = {}
        
        # Always include all processed papers in references
        # This ensures we have a references section even if citation matching fails
        for paper in processed_papers:
            # Extract year from published_date (e.g., "2024-03-15" -> "2024")
            year = "Unknown"  # Default fallback (changed from 2023)
            if hasattr(paper, 'published_date') and paper.published_date:
                try:
                    # Handle different date formats:
                    # "2024-03-15" -> "2024" (arXiv format)  
                    # "2024" -> "2024" (Semantic Scholar format)
                    if '-' in paper.published_date:
                        year = paper.published_date.split('-')[0]
                    else:
                        year = str(paper.published_date)
                    # Validate year is reasonable
                    year_int = int(year)
                    if year_int < 1900 or year_int > 2030:
                        year = "Unknown"
                except (ValueError, IndexError, AttributeError):
                    year = "Unknown"
            
            citation_details[paper.paper_id] = {
                'title': paper.title,
                'authors': paper.authors,
                'published_date': paper.published_date if hasattr(paper, 'published_date') else '',
                'citation_format': f"({paper.authors[0].split()[-1] if paper.authors else 'Unknown'}, {year})"
            }
        
        return citation_details
    
    def _save_paper(self, topic: str, paper_content: str) -> str:
        """Save the generated paper to file"""
        # Create safe filename
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_topic = safe_topic.replace(' ', '_')[:50]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_paper_{safe_topic}_{timestamp}.tex"
        output_path = os.path.join(Config.OUTPUT_DIR, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(paper_content)
        
        return output_path
    
    def generate_paper_with_stats(self, topic: str, max_papers: int = None) -> dict:
        """Generate paper and return detailed statistics"""
        output_path = self.generate_paper(topic, max_papers)
        
        # Read the generated paper
        with open(output_path, 'r', encoding='utf-8') as f:
            paper_content = f.read()
        
        # Calculate statistics
        word_count = len(paper_content.split())
        citation_count = len([line for line in paper_content.split('\n') if line.strip().startswith('1.') or line.strip().startswith('2.')])
        
        return {
            'output_path': output_path,
            'word_count': word_count,
            'citation_count': citation_count,
            'paper_content': paper_content
        }

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate research papers from topics')
    parser.add_argument('topic', help='Research topic to generate paper for')
    parser.add_argument('--max-papers', type=int, default=Config.MAX_PAPERS_TO_RETRIEVE,
                       help='Maximum number of source papers to use')
    parser.add_argument('--output-dir', default=Config.OUTPUT_DIR,
                       help='Output directory for generated papers')
    
    args = parser.parse_args()
    
    # Update config if needed
    Config.OUTPUT_DIR = args.output_dir
    Config.MAX_PAPERS_TO_RETRIEVE = args.max_papers
    
    # Generate paper
    generator = ResearchPaperGenerator()
    
    try:
        output_path = generator.generate_paper(args.topic)
        print(f"\nResearch paper generated successfully!")
        print(f"Output file: {output_path}")
        
        # Show preview
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"\nPaper preview (first 500 characters):")
            print("=" * 60)
            print(content[:500] + "...")
            print("=" * 60)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
