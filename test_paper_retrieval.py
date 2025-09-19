#!/usr/bin/env python3
"""
Test script for paper retrieval component
Runs only the paper retrieval and saves results for analysis
"""
import os
import sys
import json
import time
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from academic_retriever import AcademicRetriever
from pdf_processor import PDFProcessor

def save_paper_metadata(papers, topic, output_dir):
    """Save paper metadata to JSON for analysis"""
    metadata = {
        'topic': topic,
        'timestamp': datetime.now().isoformat(),
        'total_papers': len(papers),
        'papers': []
    }
    
    for i, paper in enumerate(papers, 1):
        paper_data = {
            'rank': i,
            'title': paper.title or 'Unknown Title',
            'authors': paper.authors or [],
            'abstract': (paper.abstract[:500] + '...' if paper.abstract and len(paper.abstract) > 500 
                        else paper.abstract or 'No abstract available'),
            'source': paper.source or 'unknown',
            'published_date': paper.published_date or 'Unknown',
            'citation_count': paper.citation_count or 0,
            'venue': paper.venue or 'Unknown',
            'pdf_url': paper.pdf_url or '',
            'paper_id': paper.id or f'unknown_{i}',
            'has_pdf': bool(paper.pdf_url)
        }
        metadata['papers'].append(paper_data)
    
    # Save metadata
    metadata_file = os.path.join(output_dir, f'paper_metadata_{topic.replace(" ", "_")}.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return metadata_file

def create_paper_summary_report(papers, topic, output_dir):
    """Create a human-readable summary report"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report_file = os.path.join(output_dir, f'paper_summary_{topic.replace(" ", "_")}.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"PAPER RETRIEVAL ANALYSIS REPORT\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Topic: {topic}\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Total Papers Retrieved: {len(papers)}\n\n")
        
        # Source breakdown
        sources = {}
        for paper in papers:
            sources[paper.source] = sources.get(paper.source, 0) + 1
        
        f.write("SOURCE BREAKDOWN:\n")
        f.write("-" * 20 + "\n")
        for source, count in sources.items():
            percentage = (count / len(papers)) * 100
            f.write(f"{source.replace('_', ' ').title()}: {count} papers ({percentage:.1f}%)\n")
        
        # Year analysis
        years = []
        for paper in papers:
            try:
                if paper.published_date and len(paper.published_date) >= 4:
                    year = int(paper.published_date[:4])
                    years.append(year)
                else:
                    years.append(2020)
            except (ValueError, IndexError):
                years.append(2020)
        
        if years:
            f.write(f"\nPUBLICATION YEARS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Average Year: {sum(years)/len(years):.1f}\n")
            f.write(f"Latest Paper: {max(years)}\n")
            f.write(f"Oldest Paper: {min(years)}\n")
        
        # Citation analysis
        total_citations = sum(p.citation_count for p in papers)
        cited_papers = [p for p in papers if p.citation_count > 0]
        
        f.write(f"\nCITATION ANALYSIS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Citations: {total_citations}\n")
        f.write(f"Papers with Citations: {len(cited_papers)}/{len(papers)}\n")
        if cited_papers:
            avg_citations = sum(p.citation_count for p in cited_papers) / len(cited_papers)
            f.write(f"Average Citations (cited papers): {avg_citations:.1f}\n")
        
        # PDF availability
        pdf_available = sum(1 for p in papers if p.pdf_url)
        f.write(f"\nPDF AVAILABILITY:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Papers with PDF: {pdf_available}/{len(papers)} ({pdf_available/len(papers)*100:.1f}%)\n")
        
        f.write(f"\n{'=' * 50}\n")
        f.write("DETAILED PAPER LIST:\n")
        f.write(f"{'=' * 50}\n\n")
        
        for i, paper in enumerate(papers, 1):
            title = paper.title or 'Unknown Title'
            authors = paper.authors or []
            source = paper.source or 'unknown'
            published_date = paper.published_date or 'Unknown'
            citation_count = paper.citation_count or 0
            pdf_url = paper.pdf_url or ''
            venue = paper.venue or ''
            abstract = paper.abstract or 'No abstract available'
            paper_id = paper.id or f'unknown_{i}'
            
            f.write(f"{i}. {title}\n")
            f.write(f"   Authors: {', '.join(authors[:3])}{'...' if len(authors) > 3 else ''}\n")
            f.write(f"   Source: {source.replace('_', ' ').title()}\n")
            f.write(f"   Year: {published_date[:4] if len(published_date) >= 4 else 'Unknown'}\n")
            f.write(f"   Citations: {citation_count}\n")
            f.write(f"   PDF Available: {'Yes' if pdf_url else 'No'}\n")
            if venue:
                f.write(f"   Venue: {venue}\n")
            f.write(f"   Abstract: {abstract[:200]}...\n")
            f.write(f"   Paper ID: {paper_id}\n")
            f.write("-" * 50 + "\n")
    
    return report_file

def test_paper_retrieval_only(topic, max_papers=15):
    """Test only the paper retrieval component"""
    print(f"🔍 Testing Paper Retrieval for: '{topic}'")
    print(f"📊 Target Papers: {max_papers}")
    print("=" * 60)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"paper_analysis_{topic.replace(' ', '_')}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output Directory: {output_dir}")
    
    try:
        # Initialize retriever
        print("\n🚀 Initializing Academic Retriever...")
        retriever = AcademicRetriever()
        
        # Search for papers
        print(f"\n🔍 Searching for papers on '{topic}'...")
        start_time = time.time()
        papers = retriever.search_papers(topic, max_papers)
        search_time = time.time() - start_time
        
        print(f"\n✅ Paper retrieval completed in {search_time:.2f} seconds")
        print(f"📋 Retrieved {len(papers)} papers")
        
        # Save results
        print("\n💾 Saving analysis results...")
        metadata_file = save_paper_metadata(papers, topic, output_dir)
        summary_file = create_paper_summary_report(papers, topic, output_dir)
        
        print(f"✅ Metadata saved to: {metadata_file}")
        print(f"✅ Summary report saved to: {summary_file}")
        
        # Show quick summary
        print(f"\n📊 QUICK SUMMARY:")
        print(f"   Total Papers: {len(papers)}")
        
        sources = {}
        for paper in papers:
            sources[paper.source] = sources.get(paper.source, 0) + 1
        
        for source, count in sources.items():
            print(f"   {source.replace('_', ' ').title()}: {count} papers")
        
        pdf_count = sum(1 for p in papers if p.pdf_url)
        print(f"   Papers with PDF: {pdf_count}/{len(papers)} ({pdf_count/len(papers)*100:.1f}%)")
        
        total_citations = sum(p.citation_count for p in papers)
        print(f"   Total Citations: {total_citations}")
        
        print(f"\n🎯 TOP 3 PAPERS:")
        for i, paper in enumerate(papers[:3], 1):
            year = paper.published_date[:4] if paper.published_date else "Unknown"
            source_icon = "📚" if paper.source == "arxiv" else "🎓"
            print(f"   {i}. {source_icon} {paper.title[:80]}... ({year})")
            if paper.citation_count > 0:
                print(f"      📊 {paper.citation_count} citations")
        
        return output_dir, papers
        
    except Exception as e:
        print(f"❌ Error during paper retrieval: {e}")
        raise

def main():
    """Main function for interactive testing"""
    print("🔬 PAPER RETRIEVAL ANALYSIS TOOL")
    print("=" * 50)
    
    # Get topic from user
    default_topics = [
        "quantum machine learning",
        "transformer architecture optimization", 
        "graph neural networks",
        "federated learning privacy",
        "multimodal deep learning"
    ]
    
    print("📝 Enter a research topic or choose from examples:")
    for i, topic in enumerate(default_topics, 1):
        print(f"   {i}. {topic}")
    
    user_input = input("\n🎯 Enter topic (or number): ").strip()
    
    # Parse input
    if user_input.isdigit() and 1 <= int(user_input) <= len(default_topics):
        topic = default_topics[int(user_input) - 1]
    elif user_input:
        topic = user_input
    else:
        topic = default_topics[0]  # Default
    
    print(f"\n🔍 Selected topic: '{topic}'")
    
    # Get number of papers
    try:
        max_papers = input("📊 Number of papers to retrieve (default 15): ").strip()
        max_papers = int(max_papers) if max_papers else 15
        max_papers = max(1, min(max_papers, 50))  # Limit between 1-50
    except ValueError:
        max_papers = 15
    
    # Run the test
    try:
        output_dir, papers = test_paper_retrieval_only(topic, max_papers)
        
        print(f"\n🎉 Analysis Complete!")
        print(f"📁 Check the '{output_dir}' folder for detailed results")
        print(f"📄 Open the summary report to review paper quality and sources")
        
    except Exception as e:
        print(f"❌ Failed to complete analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
