# Amazon Product Review Classification
- This project was focused on the development of a machine learning-based classifier designed to accurately categorize Amazon product reviews.
- By leveraging advanced algorithms such as Naive Bayes, SVM, Logistic Regression, XGBoost, and Random Forest, the classifier is able to achieve <b>89% accuracy</b> and an <b>87% F1-score</b>.
- In this project, I have used TF-IDF vectorizer for feature extraction to interpret and classify vast quantities of text data effectively.

<hr />

### Project Overview

- **Objective**: To accurately classify Amazon product reviews using machine learning algorithms.
- **Algorithms Used**: Naive Bayes, SVM, Logistic Regression, XGBoost, and Random Forest.
- **Feature Extraction**: Utilized TF-IDF.
- **Validation Method**: 5-fold cross-validation.
- **Dataset**: 220,909 Amazon product reviews ([Link]('/AMAZON_FASHION_sample.json.gz')).


### Technologies Used

- **Python**: The primary programming language used for model development.
- **Jupyter Notebook**: For documenting the development process and performing data analysis.
- **Scikit-learn**: Utilized for model training, feature extraction, and evaluation.
- **XGBoost**: For implementing the XGBoost algorithm.
- **Pandas & NumPy**: For data manipulation and numerical calculations.
- **NLTK**: For text preprocessing and natural language processing tasks such as tokenization, stemming, lemmatization and parsing.

### Algorithm Performance



| Algorithm          | Accuracy (%) | F1-Score (%) |
|--------------------|--------------|--------------|
| Naive Bayes        | 88           | 83           |
| Random Forest      | 88           | 82           |
| XGBoost            | 88           | 85           |
| Logistic Regression| 89           | 86           |
| SVM                | 89           | 87           |


