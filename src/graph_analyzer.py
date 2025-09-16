"""
Optional: Graph and Chart Analysis from PDFs
Extracts and analyzes graphs, charts, and figures from research papers
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import fitz  # PyMuPDF
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import os
import tempfile
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractedFigure:
    """Represents an extracted figure from a PDF"""
    figure_id: str
    image_path: str
    caption: str
    page_number: int
    figure_type: str  # 'chart', 'graph', 'diagram', 'table', 'unknown'
    analysis: Optional[str] = None

class GraphAnalyzer:
    """Analyzes graphs and charts from PDF research papers"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Graph analyzer initialized with temp dir: {self.temp_dir}")
    
    def extract_figures_from_pdf(self, pdf_path: str) -> List[ExtractedFigure]:
        """Extract all figures from a PDF"""
        logger.info(f"Extracting figures from PDF: {pdf_path}")
        
        figures = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract images from page
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n < 5:  # GRAY or RGB
                        # Save image
                        img_filename = f"figure_p{page_num}_{img_index}.png"
                        img_path = os.path.join(self.temp_dir, img_filename)
                        pix.save(img_path)
                        
                        # Extract caption (simple approach - look for nearby text)
                        caption = self._extract_figure_caption(page, img_index)
                        
                        # Classify figure type
                        figure_type = self._classify_figure_type(img_path)
                        
                        # Analyze if it's a chart/graph
                        analysis = None
                        if figure_type in ['chart', 'graph']:
                            analysis = self._analyze_chart(img_path)
                        
                        figure = ExtractedFigure(
                            figure_id=f"{os.path.basename(pdf_path)}_p{page_num}_{img_index}",
                            image_path=img_path,
                            caption=caption,
                            page_number=page_num,
                            figure_type=figure_type,
                            analysis=analysis
                        )
                        
                        figures.append(figure)
                    
                    pix = None
            
            doc.close()
            logger.info(f"Extracted {len(figures)} figures from PDF")
            
        except Exception as e:
            logger.error(f"Error extracting figures: {e}")
        
        return figures
    
    def _extract_figure_caption(self, page, img_index: int) -> str:
        """Extract caption text near a figure (simplified approach)"""
        try:
            # Get all text from the page
            text_dict = page.get_text("dict")
            
            # Look for common caption patterns
            caption_patterns = [
                f"Figure {img_index + 1}",
                f"Fig. {img_index + 1}",
                f"Chart {img_index + 1}",
                "Figure:",
                "Fig:",
                "Chart:"
            ]
            
            page_text = page.get_text()
            
            for pattern in caption_patterns:
                if pattern in page_text:
                    # Extract text around the pattern
                    start_idx = page_text.find(pattern)
                    if start_idx != -1:
                        # Get next 100 characters as potential caption
                        caption_text = page_text[start_idx:start_idx + 100]
                        # Clean up
                        caption_text = caption_text.replace('\n', ' ').strip()
                        return caption_text[:150]  # Limit length
            
            return "Figure caption not found"
            
        except Exception as e:
            logger.warning(f"Error extracting caption: {e}")
            return "Caption extraction failed"
    
    def _classify_figure_type(self, img_path: str) -> str:
        """Classify the type of figure using basic image analysis"""
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                return "unknown"
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Basic heuristics for classification
            height, width = gray.shape
            
            # Look for chart/graph characteristics
            
            # 1. Check for grid lines (common in charts)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
            
            if lines is not None and len(lines) > 10:
                # Many straight lines suggest a chart/graph
                
                # 2. Check for text (axis labels)
                # Simple text detection using contours
                contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                text_like_contours = [c for c in contours if 10 < cv2.contourArea(c) < 1000]
                
                if len(text_like_contours) > 5:
                    return "chart"
                else:
                    return "graph"
            
            # 3. Check aspect ratio for tables
            aspect_ratio = width / height
            if 1.2 < aspect_ratio < 3.0:
                # Look for table-like structure (rectangular regions)
                contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rectangular_contours = []
                for c in contours:
                    epsilon = 0.02 * cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, epsilon, True)
                    if len(approx) == 4:  # Rectangle
                        rectangular_contours.append(c)
                
                if len(rectangular_contours) > 3:
                    return "table"
            
            # 4. Default classification based on size and content
            total_pixels = height * width
            if total_pixels > 100000:  # Large image
                return "diagram"
            else:
                return "unknown"
                
        except Exception as e:
            logger.warning(f"Error classifying figure type: {e}")
            return "unknown"
    
    def _analyze_chart(self, img_path: str) -> str:
        """Analyze chart/graph content and extract insights"""
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                return "Chart analysis failed: Could not load image"
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            analysis_parts = []
            
            # 1. Basic image properties
            analysis_parts.append(f"Chart dimensions: {width}x{height} pixels")
            
            # 2. Color analysis
            if len(img.shape) == 3:
                # Analyze color distribution
                colors = cv2.split(img)
                dominant_colors = []
                for i, color in enumerate(['Blue', 'Green', 'Red']):
                    mean_val = np.mean(colors[i])
                    if mean_val > 100:  # Threshold for dominant color
                        dominant_colors.append(color.lower())
                
                if dominant_colors:
                    analysis_parts.append(f"Dominant colors: {', '.join(dominant_colors)}")
            
            # 3. Structure analysis
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=30)
            
            if lines is not None:
                horizontal_lines = []
                vertical_lines = []
                
                for line in lines:
                    rho, theta = line[0]
                    if abs(theta) < 0.1 or abs(theta - np.pi) < 0.1:  # Horizontal
                        horizontal_lines.append(line)
                    elif abs(theta - np.pi/2) < 0.1:  # Vertical
                        vertical_lines.append(line)
                
                analysis_parts.append(f"Grid structure: {len(horizontal_lines)} horizontal, {len(vertical_lines)} vertical lines")
                
                # Suggest chart type based on structure
                if len(horizontal_lines) > 5 and len(vertical_lines) > 2:
                    analysis_parts.append("Likely chart type: Bar chart or line graph with grid")
                elif len(vertical_lines) > len(horizontal_lines):
                    analysis_parts.append("Likely chart type: Bar chart")
                else:
                    analysis_parts.append("Likely chart type: Line graph or scatter plot")
            
            # 4. Content density
            non_zero_pixels = np.count_nonzero(gray < 240)  # Non-white pixels
            density = non_zero_pixels / (width * height)
            analysis_parts.append(f"Content density: {density:.2%}")
            
            if density > 0.3:
                analysis_parts.append("High information density - complex chart with multiple data series")
            elif density > 0.1:
                analysis_parts.append("Medium information density - standard chart")
            else:
                analysis_parts.append("Low information density - simple chart or mostly text")
            
            return " | ".join(analysis_parts)
            
        except Exception as e:
            logger.warning(f"Error analyzing chart: {e}")
            return f"Chart analysis failed: {e}"
    
    def generate_figure_insights(self, figures: List[ExtractedFigure]) -> Dict[str, str]:
        """Generate insights about the figures for inclusion in the paper"""
        insights = {}
        
        # Overall statistics
        total_figures = len(figures)
        figure_types = {}
        
        for figure in figures:
            figure_types[figure.figure_type] = figure_types.get(figure.figure_type, 0) + 1
        
        # Generate summary
        summary_parts = [f"Analysis of {total_figures} figures extracted from source papers"]
        
        if figure_types:
            type_summary = ", ".join([f"{count} {ftype}s" for ftype, count in figure_types.items()])
            summary_parts.append(f"Figure types identified: {type_summary}")
        
        # Analyze charts specifically
        charts = [f for f in figures if f.figure_type in ['chart', 'graph'] and f.analysis]
        if charts:
            summary_parts.append(f"{len(charts)} charts/graphs were analyzed for data visualization patterns")
            
            # Extract common patterns
            complex_charts = [c for c in charts if "complex chart" in c.analysis.lower()]
            if complex_charts:
                summary_parts.append(f"{len(complex_charts)} figures show complex multi-series data visualizations")
        
        insights['figure_analysis_summary'] = ". ".join(summary_parts) + "."
        
        # Individual figure insights
        for figure in figures[:5]:  # Limit to top 5 figures
            insight_key = f"figure_{figure.figure_id}"
            insight_parts = [f"Figure from page {figure.page_number + 1}"]
            
            if figure.caption and "not found" not in figure.caption.lower():
                insight_parts.append(f"Caption: {figure.caption[:100]}")
            
            if figure.analysis:
                insight_parts.append(f"Analysis: {figure.analysis}")
            
            insights[insight_key] = " | ".join(insight_parts)
        
        return insights

def create_sample_visualization(topic: str, data_insights: Dict) -> str:
    """Create a sample data visualization related to the topic"""
    try:
        # Generate sample data based on topic
        if "machine learning" in topic.lower():
            # Sample ML performance data
            algorithms = ['SVM', 'Random Forest', 'Neural Network', 'Naive Bayes']
            accuracy = [0.85, 0.92, 0.94, 0.78]
        elif "climate" in topic.lower():
            # Sample climate data
            algorithms = ['Solar', 'Wind', 'Hydro', 'Nuclear']
            accuracy = [0.15, 0.25, 0.20, 0.30]
        else:
            # Generic data
            algorithms = ['Method A', 'Method B', 'Method C', 'Method D']
            accuracy = [0.75, 0.82, 0.89, 0.71]
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        bars = plt.bar(algorithms, accuracy, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        
        plt.title(f'Performance Comparison - {topic.title()}', fontsize=14, fontweight='bold')
        plt.xlabel('Methods/Approaches', fontsize=12)
        plt.ylabel('Performance Score', fontsize=12)
        plt.ylim(0, 1.0)
        
        # Add value labels on bars
        for bar, value in zip(bars, accuracy):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save visualization
        output_dir = "generated_papers"
        os.makedirs(output_dir, exist_ok=True)
        viz_path = os.path.join(output_dir, f"visualization_{topic.replace(' ', '_')}.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return viz_path
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        return ""

if __name__ == "__main__":
    # Test the graph analyzer
    analyzer = GraphAnalyzer()
    
    # Test with a sample PDF (you would need to provide a real PDF path)
    # figures = analyzer.extract_figures_from_pdf("sample_paper.pdf")
    # insights = analyzer.generate_figure_insights(figures)
    
    # Test visualization creation
    viz_path = create_sample_visualization("machine learning in healthcare", {})
    if viz_path:
        print(f"Sample visualization created: {viz_path}")
    
    print("Graph analyzer test completed")
