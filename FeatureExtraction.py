from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
import numpy as np
import joblib

class FeatureExtractor:
    def __init__(self, max_features=5000, method='tfidf'):
        self.max_features = max_features
        self.method = method
        self.vectorizer = None
        self.fitted = False
        
    def fit(self, texts):
        """Fit the feature extractor on training texts"""
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.8
            )
        elif self.method == 'count':
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.8
            )
        else:
            raise ValueError("Method must be 'tfidf' or 'count'")
            
        self.vectorizer.fit(texts)
        self.fitted = True
        
    def transform(self, texts):
        """Transform texts to features"""
        if not self.fitted:
            raise ValueError("Feature extractor must be fitted first")
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        """Fit and transform in one step"""
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names(self):
        """Get feature names"""
        if not self.fitted:
            raise ValueError("Feature extractor must be fitted first")
        return self.vectorizer.get_feature_names_out()
    
    def reduce_dimensionality(self, features, n_components=100, method='lda'):
        """Reduce feature dimensionality"""
        if method == 'lda':
            reducer = LatentDirichletAllocation(n_components=n_components, random_state=42)
        elif method == 'svd':
            reducer = TruncatedSVD(n_components=n_components, random_state=42)
        else:
            raise ValueError("Method must be 'lda' or 'svd'")
            
        return reducer.fit_transform(features)
    
    def save(self, filepath):
        """Save the fitted vectorizer"""
        if not self.fitted:
            raise ValueError("Cannot save unfitted vectorizer")
        joblib.dump(self.vectorizer, filepath)
    
    def load(self, filepath):
        """Load a fitted vectorizer"""
        self.vectorizer = joblib.load(filepath)
        self.fitted = True

# Example usage
if __name__ == "__main__":
    # Sample texts
    texts = [
        "Meeting scheduled for tomorrow at 2 PM",
        "WIN A FREE IPHONE! CLICK HERE NOW!!!",
        "50% OFF all products this weekend only!",
        "Hey, want to grab dinner this Friday?"
    ]
    
    # Initialize and fit feature extractor
    extractor = FeatureExtractor(method='tfidf', max_features=1000)
    features = extractor.fit_transform(texts)
    
    print("Feature matrix shape:", features.shape)
    print("First few feature names:", extractor.get_feature_names()[:10])