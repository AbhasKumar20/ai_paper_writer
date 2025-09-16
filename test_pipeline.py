"""
Test script for the Research Paper Generator Pipeline
Validates core functionality without requiring API keys
"""
import os
import sys
import tempfile
from unittest.mock import Mock, patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_academic_retriever():
    """Test the academic retriever component"""
    print("🔍 Testing Academic Retriever...")
    
    try:
        from academic_retriever import AcademicRetriever, Paper
        
        # Test paper object creation
        test_paper = Paper(
            id="test_123",
            title="Test Paper on Machine Learning",
            authors=["John Doe", "Jane Smith"],
            abstract="This is a test abstract about machine learning applications.",
            pdf_url="https://example.com/test.pdf",
            published_date="2023",
            source="test"
        )
        
        assert test_paper.title == "Test Paper on Machine Learning"
        assert len(test_paper.authors) == 2
        
        # Test retriever initialization
        retriever = AcademicRetriever()
        assert retriever is not None
        
        print("✅ Academic Retriever tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Academic Retriever test failed: {e}")
        return False

def test_pdf_processor():
    """Test the PDF processor component"""
    print("📄 Testing PDF Processor...")
    
    try:
        from pdf_processor import PDFProcessor
        from academic_retriever import Paper
        
        # Create test paper
        test_paper = Paper(
            id="test_pdf",
            title="Test PDF Paper",
            authors=["Test Author"],
            abstract="Test abstract",
            pdf_url=None,  # No PDF URL for this test
            published_date="2023",
            source="test"
        )
        
        processor = PDFProcessor()
        assert processor is not None
        
        # Test with no PDF URL (should return None gracefully)
        result = processor.process_paper(test_paper)
        assert result is None  # Expected behavior for papers without PDF
        
        print("✅ PDF Processor tests passed")
        return True
        
    except Exception as e:
        print(f"❌ PDF Processor test failed: {e}")
        return False

def test_citation_mapper():
    """Test the citation mapper component"""
    print("🔗 Testing Citation Mapper...")
    
    try:
        from citation_mapper import CitationMapper, ClaimExtractor
        
        # Test claim extraction
        test_text = "Machine learning is important. It helps solve complex problems. This is a question? Short."
        claims = ClaimExtractor.extract_claims(test_text)
        
        # Should extract factual claims, not questions or short sentences
        assert len(claims) >= 1
        assert "Machine learning is important" in claims[0] or "It helps solve complex problems" in claims[0]
        
        # Test citation mapper initialization
        mapper = CitationMapper()
        assert mapper is not None
        
        print("✅ Citation Mapper tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Citation Mapper test failed: {e}")
        return False

def test_content_generator():
    """Test the content generator component (without API calls)"""
    print("✍️ Testing Content Generator...")
    
    try:
        from content_generator import ContentGenerator, SectionOutline, PaperAssembler
        
        # Test section outline creation
        section = SectionOutline(
            name="introduction",
            title="Introduction",
            key_points=["Point 1", "Point 2", "Point 3"],
            target_word_count=400,
            required_citations=3
        )
        
        assert section.name == "introduction"
        assert section.target_word_count == 400
        assert len(section.key_points) == 3
        
        # Test paper assembler
        assembler = PaperAssembler()
        assert assembler is not None
        
        # Test title generation
        title = assembler._generate_title("machine learning")
        assert "machine learning" in title.lower()
        
        print("✅ Content Generator tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Content Generator test failed: {e}")
        return False

def test_graph_analyzer():
    """Test the optional graph analyzer component"""
    print("📊 Testing Graph Analyzer...")
    
    try:
        from graph_analyzer import GraphAnalyzer, create_sample_visualization
        
        # Test analyzer initialization
        analyzer = GraphAnalyzer()
        assert analyzer is not None
        
        # Test sample visualization creation
        viz_path = create_sample_visualization("test topic", {})
        
        if viz_path and os.path.exists(viz_path):
            print(f"   Sample visualization created: {viz_path}")
        
        print("✅ Graph Analyzer tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Graph Analyzer test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("⚙️ Testing Configuration...")
    
    try:
        from config import Config
        
        # Test default values
        assert Config.MAX_PAPERS_TO_RETRIEVE > 0
        assert Config.TARGET_PAPER_LENGTH > 0
        assert Config.OPENAI_MODEL is not None
        
        print("✅ Configuration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def run_integration_test():
    """Run a mock integration test"""
    print("🧪 Running Integration Test (Mock)...")
    
    try:
        # Mock the components that require API calls
        with patch('openai.ChatCompletion.create') as mock_openai:
            mock_openai.return_value.choices = [
                Mock(message=Mock(content="Test generated content"))
            ]
            
            from paper_generator import ResearchPaperGenerator
            
            # This will fail at the API call stage, but we can test initialization
            generator = ResearchPaperGenerator()
            assert generator is not None
            assert generator.retriever is not None
            assert generator.pdf_processor is not None
            assert generator.citation_mapper is not None
            
        print("✅ Integration test passed (components initialized)")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Research Paper Generator - Test Suite")
    print("=" * 50)
    
    tests = [
        test_configuration,
        test_academic_retriever,
        test_pdf_processor,
        test_citation_mapper,
        test_content_generator,
        test_graph_analyzer,
        run_integration_test
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The pipeline is ready.")
        print("\nNext steps:")
        print("1. Set your OPENAI_API_KEY in a .env file")
        print("2. Run: python demo.py")
        print("3. Generate your first research paper!")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
