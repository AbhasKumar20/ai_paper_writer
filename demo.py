"""
Demo script for the Research Paper Generator
Shows how to use the pipeline to generate research papers
"""
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from paper_generator import ResearchPaperGenerator
from config import Config

def main():
    """Run demo of the research paper generator"""
    
    # Load environment variables
    load_dotenv()
    
    print("🔬 Research Paper Generator Demo")
    print("=" * 50)
    
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not found!")
        print("Please set your OpenAI API key in a .env file or environment variable")
        print("\nExample .env file:")
        print("OPENAI_API_KEY=your_openai_api_key_here")
        return
    
    # Initialize generator
    print("🚀 Initializing paper generator...")
    try:
        generator = ResearchPaperGenerator()
        print("✅ Generator initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing generator: {e}")
        return
    
    # Demo topics
    demo_topics = [
        "machine learning in healthcare",
        "quantum computing applications",
        "natural language processing transformers",
        "climate change and renewable energy",
        "artificial intelligence ethics"
    ]
    
    print(f"\n📚 Available demo topics:")
    for i, topic in enumerate(demo_topics, 1):
        print(f"  {i}. {topic}")
    
    # Get user choice
    try:
        choice = input(f"\nSelect a topic (1-{len(demo_topics)}) or enter your own: ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(demo_topics):
            topic = demo_topics[int(choice) - 1]
        else:
            topic = choice if choice else demo_topics[0]
        
        print(f"\n🎯 Selected topic: '{topic}'")
        
    except KeyboardInterrupt:
        print("\n👋 Demo cancelled by user")
        return
    
    # Generate paper
    print(f"\n📝 Generating research paper...")
    print("This may take a few minutes...")
    
    try:
        result = generator.generate_paper_with_stats(topic, max_papers=10)
        
        print(f"\n✅ Paper generated successfully!")
        print(f"📊 Statistics:")
        print(f"  - Word count: {result['word_count']} words")
        print(f"  - Citations: {result['citation_count']} references")
        print(f"  - Output file: {result['output_path']}")
        
        # Show preview
        print(f"\n📖 Paper preview:")
        print("=" * 60)
        preview = result['paper_content'][:1000]
        print(preview + "..." if len(result['paper_content']) > 1000 else preview)
        print("=" * 60)
        
        print(f"\n🎉 Demo completed! Check the full paper at:")
        print(f"   {os.path.abspath(result['output_path'])}")
        
    except Exception as e:
        print(f"❌ Error generating paper: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Verify your OpenAI API key is valid")
        print("3. Try a simpler topic")
        print("4. Check the logs for more details")

def quick_test():
    """Quick test with minimal output"""
    load_dotenv()
    
    if not os.getenv('OPENAI_API_KEY'):
        print("OPENAI_API_KEY required")
        return False
    
    try:
        generator = ResearchPaperGenerator()
        result = generator.generate_paper_with_stats("machine learning", max_papers=5)
        
        print(f"✅ Test passed! Generated {result['word_count']} words with {result['citation_count']} citations")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        quick_test()
    else:
        main()
