import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
from bs4 import BeautifulSoup

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class EmailPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def remove_html_tags(self, text):
        """Remove HTML tags from email content"""
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()
    
    def remove_urls(self, text):
        """Remove URLs from text"""
        return re.sub(r'http\S+|www.\S+', '', text)
    
    def remove_email_addresses(self, text):
        """Remove email addresses from text"""
        return re.sub(r'\S*@\S*\s?', '', text)
    
    def remove_special_chars(self, text):
        """Remove special characters but keep basic punctuation"""
        # Keep basic sentence structure punctuation
        text = re.sub(r'[^a-zA-Z\s\.\?!,]', '', text)
        return text
    
    def normalize_text(self, text):
        """Convert to lowercase and remove extra whitespace"""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize_text(self, text):
        """Tokenize text into words"""
        return nltk.word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from token list"""
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_tokens(self, tokens):
        """Apply stemming to tokens"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens):
        """Apply lemmatization to tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def full_preprocess(self, text, use_lemmatization=True):
        """Complete preprocessing pipeline"""
        # Remove HTML
        text = self.remove_html_tags(text)
        
        # Remove URLs and emails
        text = self.remove_urls(text)
        text = self.remove_email_addresses(text)
        
        # Normalize
        text = self.normalize_text(text)
        
        # Remove special characters
        text = self.remove_special_chars(text)
        
        # Tokenize
        tokens = self.tokenize_text(text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Apply stemming or lemmatization
        if use_lemmatization:
            tokens = self.lemmatize_tokens(tokens)
        else:
            tokens = self.stem_tokens(tokens)
            
        return ' '.join(tokens)

# Example usage
if __name__ == "__main__":
    preprocessor = EmailPreprocessor()
    sample_text = "Hello! Please check this website: https://example.com. Contact me at john@email.com."
    processed = preprocessor.full_preprocess(sample_text)
    print("Original:", sample_text)
    print("Processed:", processed)