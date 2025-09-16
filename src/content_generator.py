"""
Content Generation System
Generates academic content with precise citations using LLMs
"""
import openai
import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from config import Config
from citation_mapper import CitationMapper, ClaimExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SectionOutline:
    """Represents an outline for a paper section"""
    name: str
    title: str
    key_points: List[str]
    target_word_count: int
    required_citations: int

@dataclass
class GeneratedContent:
    """Represents generated content with citations"""
    text: str
    citations_used: List
    word_count: int
    citation_coverage: float

class ContentGenerator:
    """Generates academic content with precise citations"""
    
    def __init__(self):
        Config.validate()
        self.client = Config.get_openai_client()
        self.citation_mapper = None
        
    def set_citation_mapper(self, citation_mapper: CitationMapper):
        """Set the citation mapper for finding supporting evidence"""
        self.citation_mapper = citation_mapper
    
    def generate_complete_paper(self, topic: str, processed_papers: List) -> List[GeneratedContent]:
        """Generate complete paper in a single LLM call - OPTIMIZED VERSION"""
        logger.info(f"Generating complete paper for topic: {topic} (Single LLM call)")
        
        if not self.citation_mapper:
            raise ValueError("Citation mapper not set. Call set_citation_mapper() first.")
        
        # Analyze available papers to understand key themes
        paper_summaries = self._summarize_papers(processed_papers)
        
        # Single comprehensive prompt for entire paper
        complete_paper_prompt = f"""
        Write a complete 2-page research paper on: "{topic}" in LaTeX format.
        
        Available research papers cover these themes:
        {paper_summaries}
        
        Generate a complete academic paper with these EXACT sections:
        
        \\section{{Introduction}} (200 words)
        - Define {topic} with precise mathematical formulations where applicable
        - Present formal problem statements and technical challenges
        - State main contributions with quantitative metrics
        
        \\section{{Literature Review}} (300 words)  
        - Review existing algorithms, models, and mathematical frameworks
        - Compare computational complexities and performance bounds
        - Identify theoretical gaps and methodological limitations
        
        \\section{{Current Research and Findings}} (400 words)
        - Present novel algorithms, mathematical models, or technical solutions
        - Include formal definitions, theorems, and proofs where relevant
        - Analyze computational complexity and performance metrics
        
        \\section{{Discussion and Implications}} (150 words)
        - Discuss theoretical implications and computational trade-offs
        - Address scalability, convergence, and optimization aspects
        - Propose future algorithmic improvements
        
        \\section{{Conclusion}} (100 words)
        - Summarize technical contributions and theoretical advances
        - Quantify improvements over existing methods
        - Outline future research directions
        
        Requirements:
        - Write in VALID LaTeX format with proper commands
        - HIGHLY TECHNICAL and MATHEMATICAL writing style focused on REAL research
        - Use \\section{{}} for section headers
        - Include mathematical equations using \\begin{{equation}} and \\end{{equation}}
        - Use \\textbf{{}} for bold text, \\textit{{}} for italics
        - Use proper LaTeX math mode: $...$ for inline, \\[...\\] for display
        - Include \\begin{{algorithm}} and \\begin{{algorithmic}} for algorithms
        - ONLY include mathematics and algorithms that are DIRECTLY relevant to the source papers
        - Base all technical content on the provided research papers, not generic formulations
        - Use technical terminology and precise mathematical notation from the source papers
        - Total target: ~1150 words (much more concise)
        - NO citations yet (will be added automatically later)
        - Write as original technical research, not survey/review
        - Focus on specific algorithms, models, and methods from the source papers
        - Include complexity analysis and performance bounds from the actual papers
        - ENSURE all LaTeX syntax is correct and compilable
        - Stay focused on the specific topic: {topic}
        
        Output the complete LaTeX paper now:
        """
        
        try:
            if Config.USE_AZURE_OPENAI:
                response = self.client.chat.completions.create(
                    model=Config.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are an expert academic researcher who writes comprehensive, well-structured research papers."},
                        {"role": "user", "content": complete_paper_prompt}
                    ],
                    max_tokens=8000,  # Increased for full paper
                    temperature=Config.TEMPERATURE
                )
                full_paper_content = response.choices[0].message.content.strip()
            else:
                response = openai.ChatCompletion.create(
                    model=Config.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are an expert academic researcher who writes comprehensive, well-structured research papers."},
                        {"role": "user", "content": complete_paper_prompt}
                    ],
                    max_tokens=8000,
                    temperature=Config.TEMPERATURE
                )
                full_paper_content = response.choices[0].message.content.strip()
            
            # Parse the generated paper into sections
            sections_content = self._parse_complete_paper(full_paper_content)
            
            # Add citations to each section
            generated_sections = []
            section_names = ["Introduction", "Literature Review", "Current Research and Findings", "Discussion and Implications", "Conclusion"]
            target_word_counts = [400, 600, 700, 250, 150]
            
            for i, (section_name, content) in enumerate(sections_content.items()):
                if content.strip():
                    # Extract claims and add citations
                    claims = ClaimExtractor.extract_claims(content)
                    logger.info(f"Extracted {len(claims)} claims for {section_name}")
                    
                    cited_content = self._add_citations_to_content(content, claims)
                    word_count = len(cited_content.split())
                    citations_used = self._extract_citations_from_content(cited_content)
                    coverage = len(citations_used) / len(claims) if claims else 0
                    
                    generated_section = GeneratedContent(
                        text=cited_content,
                        citations_used=citations_used,
                        word_count=word_count,
                        citation_coverage=coverage
                    )
                    generated_sections.append(generated_section)
                    
                    logger.info(f"{section_name}: {word_count} words, {len(citations_used)} citations, {coverage:.1%} coverage")
            
            logger.info(f"Complete paper generated in 1 LLM call: {sum(s.word_count for s in generated_sections)} total words")
            return generated_sections
            
        except Exception as e:
            logger.error(f"Error generating complete paper: {e}")
            # Fallback to original method
            logger.info("Falling back to multi-call approach")
            return self._generate_paper_sections_fallback(topic, processed_papers)
    
    def _parse_complete_paper(self, paper_content: str) -> Dict[str, str]:
        """Parse the complete paper into individual sections"""
        sections = {}
        
        # Split by LaTeX section headers first
        latex_section_pattern = r'\\section\{([^}]+)\}'
        parts = re.split(latex_section_pattern, paper_content)
        
        if len(parts) > 1:
            # Skip the first part (usually empty or title)
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    section_name = parts[i].strip()
                    section_content = parts[i + 1].strip()
                    sections[section_name] = section_content
        else:
            # Fallback: try markdown section headers
            section_pattern = r'##\s*([^#\n]+)'
            parts = re.split(section_pattern, paper_content)
            
            if len(parts) > 1:
                for i in range(1, len(parts), 2):
                    if i + 1 < len(parts):
                        section_name = parts[i].strip()
                        section_content = parts[i + 1].strip()
                        sections[section_name] = section_content
            else:
                # Final fallback: try to split by common section names
                section_names = ["Introduction", "Literature Review", "Current Research", "Discussion", "Conclusion"]
                current_content = paper_content
                
                for section_name in section_names:
                    # Try both LaTeX and plain text patterns
                    patterns = [
                        rf'\\section\{{{section_name}\}}(.*?)(?=\\section\{{|$)',
                        rf'{section_name}[^\n]*\n(.*?)(?={"|".join(section_names[section_names.index(section_name)+1:])}|$)'
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, current_content, re.DOTALL | re.IGNORECASE)
                        if match:
                            sections[section_name] = match.group(1).strip()
                            break
        
        # Ensure we have all expected sections
        expected_sections = ["Introduction", "Literature Review", "Current Research and Findings", "Discussion and Implications", "Conclusion"]
        for section_name in expected_sections:
            if section_name not in sections:
                # Try partial matches
                for key in sections.keys():
                    if any(word in key.lower() for word in section_name.lower().split()):
                        sections[section_name] = sections[key]
                        break
                else:
                    sections[section_name] = f"Content for {section_name} section was not properly generated."
        
        return sections
    
    def _generate_paper_sections_fallback(self, topic: str, processed_papers: List) -> List[GeneratedContent]:
        """Fallback to original multi-call approach if single call fails"""
        outline = self.generate_paper_outline(topic, processed_papers)
        sections = []
        
        for section_outline in outline:
            section_content = self.generate_section_content(section_outline, topic)
            sections.append(section_content)
        
        return sections
    
    def generate_paper_outline(self, topic: str, processed_papers: List) -> List[SectionOutline]:
        """Generate a structured outline for the research paper"""
        logger.info(f"Generating outline for topic: {topic}")
        
        # Analyze available papers to understand key themes
        paper_summaries = self._summarize_papers(processed_papers)
        
        outline_prompt = f"""
        Create a detailed outline for a 2-page research paper on: "{topic}"
        
        Available research papers cover these themes:
        {paper_summaries}
        
        Create an outline with the following sections:
        1. Introduction (400 words)
        2. Literature Review (600 words)
        3. Current Research and Findings (700 words)
        4. Discussion and Implications (250 words)
        5. Conclusion (150 words)
        
        For each section, provide:
        - A clear title
        - 3-5 key points to cover
        - Number of citations needed
        
        Format as JSON with this structure:
        {{
            "sections": [
                {{
                    "name": "introduction",
                    "title": "Introduction",
                    "key_points": ["point1", "point2", "point3"],
                    "target_word_count": 400,
                    "required_citations": 3
                }}
            ]
        }}
        """
        
        try:
            if Config.USE_AZURE_OPENAI:
                response = self.client.chat.completions.create(
                    model=Config.OPENAI_MODEL,
                    messages=[{"role": "user", "content": outline_prompt}],
                    max_tokens=2000,
                    temperature=Config.TEMPERATURE
                )
                outline_text = response.choices[0].message.content
            else:
                response = openai.ChatCompletion.create(
                    model=Config.OPENAI_MODEL,
                    messages=[{"role": "user", "content": outline_prompt}],
                    max_tokens=2000,
                    temperature=Config.TEMPERATURE
                )
                outline_text = response.choices[0].message.content
            outline_data = self._parse_outline_response(outline_text)
            
            sections = []
            for section_data in outline_data.get('sections', []):
                section = SectionOutline(
                    name=section_data['name'],
                    title=section_data['title'],
                    key_points=section_data['key_points'],
                    target_word_count=section_data['target_word_count'],
                    required_citations=section_data['required_citations']
                )
                sections.append(section)
            
            logger.info(f"Generated outline with {len(sections)} sections")
            return sections
            
        except Exception as e:
            logger.error(f"Error generating outline: {e}")
            return self._create_default_outline()
    
    def generate_section_content(self, section: SectionOutline, topic: str) -> GeneratedContent:
        """Generate content for a specific section with citations"""
        logger.info(f"Generating content for section: {section.title}")
        
        if not self.citation_mapper:
            raise ValueError("Citation mapper not set. Call set_citation_mapper() first.")
        
        # Generate initial content
        initial_content = self._generate_initial_content(section, topic)
        
        # Extract claims that need citations
        claims = ClaimExtractor.extract_claims(initial_content)
        logger.info(f"Extracted {len(claims)} claims for citation")
        
        # Find citations for each claim
        cited_content = self._add_citations_to_content(initial_content, claims)
        
        # Count words and calculate citation coverage
        word_count = len(cited_content.split())
        citations_used = self._extract_citations_from_content(cited_content)
        coverage = len(citations_used) / len(claims) if claims else 0
        
        return GeneratedContent(
            text=cited_content,
            citations_used=citations_used,
            word_count=word_count,
            citation_coverage=coverage
        )
    
    def _generate_initial_content(self, section: SectionOutline, topic: str) -> str:
        """Generate initial content without citations"""
        key_points_str = "\n".join(f"- {point}" for point in section.key_points)
        
        content_prompt = f"""
        Write the {section.title} section for a research paper on "{topic}".
        
        Key points to cover:
        {key_points_str}
        
        Requirements:
        - Target length: {section.target_word_count} words
        - Academic writing style
        - Clear, factual statements
        - Logical flow between points
        - No citations yet (will be added later)
        
        Write in a scholarly tone appropriate for an academic research paper.
        """
        
        try:
            if Config.USE_AZURE_OPENAI:
                response = self.client.chat.completions.create(
                    model=Config.OPENAI_MODEL,
                    messages=[{"role": "user", "content": content_prompt}],
                    max_tokens=Config.MAX_TOKENS_PER_REQUEST,
                    temperature=Config.TEMPERATURE
                )
                return response.choices[0].message.content.strip()
            else:
                response = openai.ChatCompletion.create(
                    model=Config.OPENAI_MODEL,
                    messages=[{"role": "user", "content": content_prompt}],
                    max_tokens=Config.MAX_TOKENS_PER_REQUEST,
                    temperature=Config.TEMPERATURE
                )
                return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating initial content: {e}")
            return f"Error generating content for {section.title}: {e}"
    
    def _add_citations_to_content(self, content: str, claims: List[str]) -> str:
        """Add citations to content based on claims"""
        cited_content = content
        citations_added = []
        
        for claim in claims:
            # Find supporting citations
            supporting_citations = self.citation_mapper.find_supporting_citations(claim, top_k=2)
            
            if supporting_citations:
                # Create citation text
                citation_text = self._format_inline_citations(supporting_citations)
                
                # Find the claim in the content and add citation
                # Simple approach: add citation at the end of sentences containing key words from claim
                claim_keywords = self._extract_keywords(claim)
                
                # Look for sentences in content that match the claim
                sentences = re.split(r'(?<=[.!?])\s+', cited_content)
                for i, sentence in enumerate(sentences):
                    if self._sentence_matches_claim(sentence, claim_keywords):
                        # Add citation to this sentence
                        if not any(cite in sentence for cite in citations_added):
                            sentences[i] = sentence.rstrip('.') + f" {citation_text}."
                            citations_added.extend([cite.citation_format for cite in supporting_citations])
                            break
                
                cited_content = " ".join(sentences)
        
        return cited_content
    
    def _format_inline_citations(self, citations: List) -> str:
        """Format citations for inline use"""
        if not citations:
            return ""
        
        if len(citations) == 1:
            return citations[0].citation_format
        else:
            citation_formats = [cite.citation_format for cite in citations]
            return f"({'; '.join(citation_formats).replace('(', '').replace(')', '')})"
    
    def _extract_keywords(self, claim: str) -> List[str]:
        """Extract key words from a claim for matching"""
        # Remove common words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'cannot', 'this', 'that', 'these', 'those'}
        
        words = re.findall(r'\b\w+\b', claim.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        return keywords[:5]  # Top 5 keywords
    
    def _sentence_matches_claim(self, sentence: str, claim_keywords: List[str]) -> bool:
        """Check if a sentence matches a claim based on keywords"""
        sentence_lower = sentence.lower()
        matches = sum(1 for keyword in claim_keywords if keyword in sentence_lower)
        return matches >= 2  # At least 2 keywords must match
    
    def _extract_citations_from_content(self, content: str) -> List[str]:
        """Extract citation references from content"""
        # Look for citation patterns like (Author, Year) or (Author et al., Year)
        citation_pattern = r'\([^)]*\d{4}[^)]*\)'
        citations = re.findall(citation_pattern, content)
        return list(set(citations))  # Remove duplicates
    
    def _summarize_papers(self, processed_papers: List) -> str:
        """Create a summary of available papers for outline generation"""
        summaries = []
        for paper in processed_papers[:10]:  # Limit to top 10 papers
            summary = f"- {paper.title}: {paper.abstract[:150]}..."
            summaries.append(summary)
        return "\n".join(summaries)
    
    def _parse_outline_response(self, response: str) -> Dict:
        """Parse the JSON response from outline generation"""
        try:
            import json
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning(f"Could not parse JSON outline: {e}")
        
        # Return default structure if parsing fails
        return {"sections": []}
    
    def _create_default_outline(self) -> List[SectionOutline]:
        """Create a default outline if generation fails"""
        return [
            SectionOutline("introduction", "Introduction", 
                          ["Mathematical formulation", "Problem statement", "Technical contributions"], 
                          200, 2),
            SectionOutline("literature_review", "Literature Review", 
                          ["Algorithmic approaches", "Complexity analysis", "Methodological gaps"], 
                          300, 4),
            SectionOutline("current_research", "Current Research and Findings", 
                          ["Novel algorithms", "Theoretical results", "Performance analysis"], 
                          400, 6),
            SectionOutline("discussion", "Discussion and Implications", 
                          ["Computational trade-offs", "Scalability analysis", "Future improvements"], 
                          150, 2),
            SectionOutline("conclusion", "Conclusion", 
                          ["Technical contributions", "Quantitative improvements", "Future work"], 
                          100, 1)
        ]

class PaperAssembler:
    """Assembles the final research paper with proper formatting"""
    
    def __init__(self):
        pass
    
    def assemble_paper(self, topic: str, sections: List[GeneratedContent], 
                      section_outlines: List[SectionOutline], 
                      citation_details: Dict) -> str:
        """Assemble the complete research paper in LaTeX format"""
        logger.info("Assembling final research paper in LaTeX format")
        
        # Create title
        title = self._generate_title(topic)
        
        # LaTeX document structure
        paper_content = []
        
        # LaTeX preamble
        paper_content.append("\\documentclass[12pt]{article}")
        paper_content.append("\\usepackage[utf8]{inputenc}")
        paper_content.append("\\usepackage{amsmath}")
        paper_content.append("\\usepackage{amsfonts}")
        paper_content.append("\\usepackage{amssymb}")
        paper_content.append("\\usepackage{cite}")
        paper_content.append("\\usepackage{url}")
        paper_content.append("\\usepackage{algorithm}")
        paper_content.append("\\usepackage{algorithmic}")
        paper_content.append("")
        paper_content.append(f"\\title{{{title}}}")
        paper_content.append("\\author{AI}")
        paper_content.append("\\date{\\today}")
        paper_content.append("")
        paper_content.append("\\begin{document}")
        paper_content.append("")
        paper_content.append("\\maketitle")
        paper_content.append("")
        
        # Add abstract
        abstract = self._generate_abstract(topic, sections)
        paper_content.append("\\begin{abstract}")
        paper_content.append(abstract)
        paper_content.append("\\end{abstract}")
        paper_content.append("")
        
        # Add main sections
        for section_content, outline in zip(sections, section_outlines):
            paper_content.append(f"\\section{{{outline.title}}}")
            paper_content.append("")
            paper_content.append(section_content.text)
            paper_content.append("")
        
        # Add references
        references = self._format_references(citation_details)
        if references:
            paper_content.append("\\section{References}")
            paper_content.append("")
            paper_content.append("\\begin{enumerate}")
            for ref in references.split('\n'):
                if ref.strip():
                    # Remove the number prefix and format as LaTeX item
                    clean_ref = re.sub(r'^\d+\.\s*', '', ref.strip())
                    paper_content.append(f"\\item {clean_ref}")
            paper_content.append("\\end{enumerate}")
            paper_content.append("")
        
        paper_content.append("\\end{document}")
        
        final_paper = "\n".join(paper_content)
        
        # Clean up any duplicate \end{document} commands
        final_paper = re.sub(r'\\end\{document\}.*?\\end\{document\}', r'\\end{document}', final_paper, flags=re.DOTALL)
        
        # Remove any content after the final \end{document}
        final_paper = re.sub(r'(\\end\{document\}).*', r'\1', final_paper, flags=re.DOTALL)
        
        # Add statistics
        total_words = sum(section.word_count for section in sections)
        total_citations = len(citation_details)
        
        logger.info(f"Paper assembled: {total_words} words, {total_citations} unique citations")
        
        return final_paper
    
    def _generate_title(self, topic: str) -> str:
        """Generate a title for the paper"""
        # Generate more varied, non-survey titles
        topic_words = topic.lower().split()
        
        # Different title patterns to avoid always using "review" or "survey"
        title_patterns = [
            f"{topic.title()}: Advances and Applications",
            f"Novel Approaches in {topic.title()}",
            f"{topic.title()}: Methods and Implementation",
            f"Innovations in {topic.title()}",
            f"{topic.title()}: Current Developments and Future Directions",
            f"Exploring {topic.title()}: Techniques and Solutions",
            f"{topic.title()}: Progress and Challenges"
        ]
        
        # Simple selection based on topic length
        pattern_index = len(topic_words) % len(title_patterns)
        return title_patterns[pattern_index]
    
    def _generate_abstract(self, topic: str, sections: List[GeneratedContent]) -> str:
        """Generate an abstract based on the paper content"""
        # Extract first sentence from each section for abstract
        key_points = []
        for section in sections:
            # Get first complete sentence (up to first period followed by space or end)
            sentences = section.text.split('. ')
            if sentences:
                first_sentence = sentences[0]
                if not first_sentence.endswith('.'):
                    first_sentence += '.'
                if len(first_sentence) > 20 and len(first_sentence) < 200:
                    key_points.append(first_sentence)
        
        abstract = f"This paper provides a comprehensive technical analysis of {topic}. " + " ".join(key_points[:3])
        
        # Ensure abstract ends with a complete sentence, no truncation
        if len(abstract) > 800:
            # Find last complete sentence within reasonable length
            sentences = abstract.split('. ')
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence + '. ') <= 800:
                    truncated += sentence + '. '
                else:
                    break
            abstract = truncated.strip()
        
        return abstract
    
    def _format_references(self, citation_details: Dict) -> str:
        """Format the references section"""
        references = []
        for i, (paper_id, details) in enumerate(citation_details.items(), 1):
            authors_str = ", ".join(details['authors'][:3])
            if len(details['authors']) > 3:
                authors_str += " et al."
            
            reference = f"{i}. {authors_str} {details['title']}."
            references.append(reference)
        
        return "\n".join(references)

if __name__ == "__main__":
    # Test content generation
    generator = ContentGenerator()
    
    # Test outline generation
    test_topic = "machine learning in healthcare"
    outline = generator.generate_paper_outline(test_topic, [])
    
    print(f"Generated outline with {len(outline)} sections:")
    for section in outline:
        print(f"  - {section.title} ({section.target_word_count} words)")
        print(f"    Key points: {section.key_points}")
        print()
