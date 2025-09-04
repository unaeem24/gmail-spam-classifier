from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class EmailClassifierTrainer:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
<<<<<<< HEAD
            # 'naive_bayes': MultinomialNB(),
            # 'svm': SVC(kernel='linear', probability=True, random_state=42),
=======
            'naive_bayes': MultinomialNB(),
            'svm': SVC(kernel='linear', probability=True, random_state=42),
>>>>>>> e716f77073d2f53e33f5f733a04aba61b70601a3
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        self.best_model = None
        self.best_model_name = None
        
    def train(self, X_train, y_train, cv=5):
        """Train and evaluate all models using cross-validation"""
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, n_jobs=-1)
            results[name] = {
                'model': model,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores)
            }
            print(f"{name} - CV Accuracy: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")
            
            # Fit the model on full training data
            model.fit(X_train, y_train)
        
        # Select the best model
        self.best_model_name = max(results, key=lambda x: results[x]['cv_mean'])
        self.best_model = results[self.best_model_name]['model']
        
        print(f"\nBest model: {self.best_model_name} with CV accuracy: {results[self.best_model_name]['cv_mean']:.3f}")
        
        return results
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='random_forest'):
        """Perform hyperparameter tuning for a specific model"""
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
<<<<<<< HEAD
            # 'naive_bayes': {
            #     'alpha': [0.1, 0.5, 1.0, 2.0]
            # },
            # 'svm': {
            #     'C': [0.1, 1, 10],
            #     'kernel': ['linear', 'rbf']
            # },
=======
            'naive_bayes': {
                'alpha': [0.1, 0.5, 1.0, 2.0]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            },
>>>>>>> e716f77073d2f53e33f5f733a04aba61b70601a3
            'logistic_regression': {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            }
        }
        
        if model_name not in param_grids:
            raise ValueError(f"No parameter grid defined for {model_name}")
        
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grids[model_name],
            cv=5,
            n_jobs=-1,
            scoring='accuracy'
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best CV accuracy: {grid_search.best_score_:.3f}")
        
        # Update the model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        return grid_search
    
    def evaluate(self, model, X_test, y_test, class_names):
        """Evaluate model on test set"""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        return y_pred, y_prob
    
    def save_model(self, model, filepath):
        """Save trained model to file"""
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from file"""
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model

# Example usage
if __name__ == "__main__":
    # Example data
    X_train = ...  # your features
    y_train = ["spam", "not spam", "spam", "not spam", ...]  # your labels

    X_test = ...  # your test features
    y_test = ["spam", "not spam", ...]  # your test labels

    trainer = EmailClassifierTrainer()
    trainer.train(X_train, y_train)
<<<<<<< HEAD
    trainer.evaluate(trainer.best_model, X_test, y_test, class_names=["not spam", "spam"])

=======
    trainer.evaluate(trainer.best_model, X_test, y_test, class_names=["not spam", "spam"])
>>>>>>> e716f77073d2f53e33f5f733a04aba61b70601a3
