from preprocess import EmailPreprocessor
from FeatureExtraction import FeatureExtractor
from train_from_sample import EmailClassifierTrainer
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import numpy as np

class EmailClassificationPipeline:
    def __init__(self):
        self.preprocessor = EmailPreprocessor()
        self.feature_extractor = None
        self.trainer = EmailClassifierTrainer()
        self.label_encoder = None
        self.class_names = ['spam', 'promotions', 'social']
    
    def prepare_data(self, texts, labels, test_size=0.2, random_state=42):
        """Prepare and split the data"""
        # Preprocess texts
        print("Preprocessing texts...")
        processed_texts = [self.preprocessor.full_preprocess(text) for text in texts]
        
        # Extract features
        print("Extracting features...")
        self.feature_extractor = FeatureExtractor(method='tfidf', max_features=5000)
        X = self.feature_extractor.fit_transform(processed_texts)
        
        # Convert labels to numerical values
        if self.label_encoder is None:
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(labels)
        else:
            y = self.label_encoder.transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, texts, labels, tune_hyperparameters=False):
        """Train the complete pipeline"""
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(texts, labels)
        
        # Train models
        print("Training models...")
        results = self.trainer.train(X_train, y_train)
        
        # Hyperparameter tuning if requested
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            self.trainer.hyperparameter_tuning(X_train, y_train, self.trainer.best_model_name)
        
        # Evaluate on test set
        print("Evaluating on test set...")
        y_pred, y_prob = self.trainer.evaluate(
            self.trainer.best_model, 
            X_test, 
            y_test,
            self.class_names
        )
        
        return results, (X_test, y_test, y_pred, y_prob)
    
    def predict(self, new_texts):
        """Predict categories for new texts"""
        if self.feature_extractor is None or self.trainer.best_model is None:
            raise ValueError("Pipeline must be trained first")
        
        # Preprocess new texts
        processed_texts = [self.preprocessor.full_preprocess(text) for text in new_texts]
        
        # Extract features
        X_new = self.feature_extractor.transform(processed_texts)
        
        # Predict
        predictions = self.trainer.best_model.predict(X_new)
        probabilities = self.trainer.best_model.predict_proba(X_new)
        
        # Convert back to original labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        # Create results with confidence scores
        results = []
        for i, text in enumerate(new_texts):
            result = {
                'text': text,
                'predicted_category': predicted_labels[i],
                'confidence': probabilities[i].max(),
                'probabilities': dict(zip(self.class_names, probabilities[i]))
            }
            results.append(result)
        
        return results
    
    def save_pipeline(self, filepath):
        """Save the complete pipeline"""
        pipeline_data = {
            'feature_extractor': self.feature_extractor,
            'model': self.trainer.best_model,
            'label_encoder': self.label_encoder,
            'class_names': self.class_names
        }
        joblib.dump(pipeline_data, filepath)
        print(f"Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath):
        """Load a complete pipeline"""
        pipeline_data = joblib.load(filepath)
        self.feature_extractor = pipeline_data['feature_extractor']
        self.trainer.best_model = pipeline_data['model']
        self.label_encoder = pipeline_data['label_encoder']
        self.class_names = pipeline_data['class_names']
        print(f"Pipeline loaded from {filepath}")
        
# Example usage
if __name__ == "__main__":
    # Sample data
    sample_texts = [
        "Meeting scheduled for tomorrow at 2 PM regarding project deadline",
        "WIN A FREE IPHONE! CLICK HERE NOW!!!",
        "50% OFF all products this weekend only!",
        "Hey, want to grab dinner this Friday?",
        "Quarterly report submission required by EOD",
        "Your account statement is ready for review",
        "Party at my place this Saturday, bring friends!",
        "URGENT: Your account has been compromised",
        "Project status update meeting moved to 3 PM",
        "Limited time offer: Buy one get one free!"
    ]
    
    sample_labels = [
        'work', 'spam', 'promotions', 'social',
        'work', 'promotions', 'social', 'spam',
        'work', 'promotions'
    ]
    
    # Initialize and train pipeline
    pipeline = EmailClassificationPipeline()
    results, evaluation = pipeline.train(sample_texts, sample_labels)
    
    # Test prediction
    new_emails = [
        "Team lunch next Wednesday at 12 PM",
        "CONGRATULATIONS! You've won a luxury cruise!",
        "Sale on all electronics this weekend",
        "Can you review the document I sent yesterday?"
    ]
    
    predictions = pipeline.predict(new_emails)
    print("\nPredictions for new emails:")
    for pred in predictions:
        print(f"Text: {pred['text'][:50]}...")
        print(f"Predicted: {pred['predicted_category']} (Confidence: {pred['confidence']:.2f})")
        print("---")