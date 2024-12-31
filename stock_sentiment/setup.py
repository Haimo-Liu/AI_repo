import nltk
import sys

def setup_nltk():
    """Download required NLTK data."""
    required_packages = [
        'punkt',
        'averaged_perceptron_tagger',
        'vader_lexicon'
    ]
    
    for package in required_packages:
        try:
            nltk.download(package, quiet=True)
            print(f"Successfully downloaded {package}")
        except Exception as e:
            print(f"Error downloading {package}: {e}")
            sys.exit(1)

if __name__ == "__main__":
    setup_nltk()
