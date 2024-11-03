# News2Stock-Predictor-Basic-
Natural Language Processing project to analyze the impact of global news on stock price fluctuations. Employs a Bag-of-Words model with N-gram features to classify daily stock movements using Random-Forest-Classifier.
# Stock Market Movement Prediction Using News Headlines

## Project Overview
This project analyzes the relationship between stock price movements and top 25 world news headlines. The analysis uses natural language processing (NLP) techniques to predict stock price movements based on news sentiment.

## Dataset Description
- **Source**: Kaggle, scraped from Yahoo Finance
- **Time Period**: January 2000 - September 2016
- **Features**: 
  - Daily top 25 news headlines
  - Binary labels (1: stock price increased, 0: unchanged/decreased)

## Methodology
### Data Preprocessing
- Dataset was loaded using pandas with ISO-8859-1 encoding
- Text data (headlines) were processed using Natural Language Processing techniques

### Feature Engineering
- Implemented Bag of Words (BoW) model using CountVectorizer
- Using RandomForestClassifier
- Experimented with different n-gram ranges:
  - Unigrams (ngram_range=(1,1))
  - Bigrams (ngram_range=(2,2))

### Model Evaluation
Tested two different train-test split configurations:

1. 80-20 Split:
   - Unigrams: 53% accuracy
   - Bigrams: 56% accuracy

2. 90-10 Split:
   - Unigrams: 84% accuracy
   - Bigrams: 87% accuracy

## Key Findings
- Bigram features consistently outperformed unigram features
- Larger training set (90-10 split) significantly improved model performance
- The substantial improvement with more training data suggests the model benefits from seeing more examples of news headline patterns

## Technical Implementation
```python
from sklearn.feature_extraction.text import CountVectorizer
countvector = CountVectorizer(ngram_range=(2,2))
traindataset = countvector.fit_transform(headlines)
```

## Limitations and Considerations
- The significant performance difference between split ratios suggests potential overfitting with the 90-10 split
- Results might be sensitive to the specific time period covered in the training data

The ngram_range parameter in CountVectorizer is crucial for text analysis as it determines how words are grouped together when creating features. Here's why it's important:

What are n-grams? -> Unigrams (n=1): Single words (e.g., "market", "crash")
Bigrams (n=2): Two consecutive words (e.g., "stock market", "market crash")
Trigrams (n=3): Three consecutive words (e.g., "stock market crash")


In my specific case: -> When using ngram_range=(2,2), you're exclusively using bigrams
This captures important phrase-level patterns in financial news
Example: "interest rates" has different implications than "interest" and "rates" separately


Why bigrams performed better:
Financial news often contains meaningful two-word phrases: "Federal Reserve", "interest rates", "market crash" These combinations carry more specific meaning than individual words so accuracy improved from 53% to 56% (80-20 split) and 84% to 87% (90-10 split) when using bigrams

Trade-offs:


Higher n-grams capture more context but increase feature space
More features require more training data
This explains why the 90-10 split worked better - more training data to learn these complex patterns
