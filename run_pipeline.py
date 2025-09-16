"""
Simple script to run the research paper generation pipeline with Azure OpenAI
"""
import os
import sys

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Validate that required environment variables are set
from config import Config
try:
    Config.validate_configuration()
    print("✅ Environment configuration validated successfully")
except ValueError as e:
    print(f"❌ Configuration Error: {e}")
    print("\n💡 Please create a .env file with your API credentials:")
    print("   cp .env.example .env")
    print("   # Edit .env with your actual API keys")
    exit(1)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_quick_demo():
    """Run a quick demo of the paper generator"""
    print("🔬 Research Paper Generator - Technical Paper Generation")
    print("=" * 60)
    
    try:
        # Get topic directly from user input
        print("Enter a research topic for technical paper generation.")
        print("Examples: 'deep neural networks', 'quantum algorithms', 'optimization theory'")
        
        selected_topic = input("\n📝 Enter your research topic: ").strip()
        
        if not selected_topic:
            selected_topic = "machine learning optimization"  # Default
            print(f"Using default topic: '{selected_topic}'")
        
        print(f"\n🎯 Topic: '{selected_topic}'")
        print("\n🚀 Generating technical research paper...")
        print("⚡ Optimized pipeline: 1 LLM call, ~1 minute generation time")
        print("📐 Technical focus: Mathematical formulations and algorithms")
        
        # Import and run the generator
        from paper_generator import ResearchPaperGenerator
        
        generator = ResearchPaperGenerator()
        result = generator.generate_paper_with_stats(selected_topic, max_papers=5)
        
        print(f"\n✅ Technical paper generated successfully!")
        print(f"📊 Statistics:")
        print(f"  - Word count: {result['word_count']} words (~1150 target)")
        print(f"  - Citations: {result['citation_count']} references")
        print(f"  - Format: LaTeX (.tex file)")
        print(f"  - Output file: {result['output_path']}")
        
        # Show preview
        print(f"\n📖 LaTeX Paper preview:")
        print("=" * 60)
        preview = result['paper_content'][:800]
        print(preview + "..." if len(result['paper_content']) > 800 else preview)
        print("=" * 60)
        
        print(f"\n🎉 Complete LaTeX paper saved to:")
        print(f"   {os.path.abspath(result['output_path'])}")
        print(f"\n💡 To compile: pdflatex {os.path.basename(result['output_path'])}")
        
        return True
        
    except KeyboardInterrupt:
        print("\n👋 Demo cancelled by user")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nThis might be due to:")
        print("1. Internet connection issues")
        print("2. Azure OpenAI service issues") 
        print("3. Missing dependencies")
        return False

def run_simple_test():
    """Run a simple test without full paper generation"""
    print("🧪 Running Simple Test...")
    
    try:
        from content_generator import ContentGenerator, SectionOutline
        
        generator = ContentGenerator()
        
        # Test content generation
        test_section = SectionOutline(
            name="introduction",
            title="Introduction", 
            key_points=["Define the topic", "Explain importance"],
            target_word_count=150,
            required_citations=1
        )
        
        content = generator._generate_initial_content(test_section, "artificial intelligence")
        
        print("✅ Test passed!")
        print(f"📄 Generated content ({len(content.split())} words):")
        print("-" * 40)
        print(content[:300] + "..." if len(content) > 300 else content)
        print("-" * 40)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main function"""
    print("🔵 Azure OpenAI Technical Research Paper Generator")
    print("=" * 55)
    
    print("Choose an option:")
    print("1. Generate technical research paper (~1 minute)")
    print("2. Run simple test (30 seconds)")
    print("3. Exit")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            return run_quick_demo()
        elif choice == "2":
            return run_simple_test()
        elif choice == "3":
            print("👋 Goodbye!")
            return True
        else:
            print("Invalid choice. Generating technical paper...")
            return run_quick_demo()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
