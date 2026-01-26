"""
Receipt Recognition & Optimization Main Entry Point
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Insert src into python path
sys.path.insert(0, str(Path(__file__).parent))

from crew import ReceiptProcessingCrew


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <absolute_image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        sys.exit(1)
        
    print(f"Starting processing for: {image_path}")
    
    try:
        crew = ReceiptProcessingCrew()
        result = crew.run(image_path)
        print("\nFinal Result:")
        print(result)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
