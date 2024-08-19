# BBC News Articles Topic Modeling with NMF

This project demonstrates the application of Non-negative Matrix Factorization (NMF) to perform topic modeling on a dataset of BBC news articles. The goal is to identify distinct topics within the articles and assign appropriate labels based on the most frequent words associated with each topic.

## Project Overview

### Objectives

1. **Data Preprocessing**:
   - Clean and preprocess the news articles by removing numbers and non-alphabetic characters.
   - Apply tokenization and vectorization using `TfidfVectorizer`.

2. **Topic Modeling**:
   - Use Non-negative Matrix Factorization (NMF) to discover hidden topics within the news articles.
   - Identify the top 15 words for each topic and assign relevant labels.

3. **Topic Assignment**:
   - Assign the most relevant topic to each news article based on the NMF results.
   - Add these topics and their corresponding labels to the dataset.

## Tools and Libraries

- **Python**: Programming language used for the entire project.
- **Pandas**: Library used for data manipulation and analysis.
- **NumPy**: Library used for numerical computations.
- **Regular Expressions (re)**: Used for data cleaning and preprocessing.
- **Scikit-learn**: Machine learning library used for vectorization and topic modeling.

## Key Steps and Functions

### 1. Data Preprocessing

- **Loading Data**:
  - Load the dataset containing BBC news articles from a CSV file using Pandas.
  - Example:
    ```python
    news_articles_df = pd.read_csv('Resources/bbc_news_articles.csv')
    ```

- **Cleaning Data**:
  - Remove numbers and non-alphabetic characters from the `news_summary` column using regular expressions.
  - Example:
    ```python
    news_articles_df['news_summary'] = news_articles_df['news_summary'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x)))
    ```

### 2. Vectorization and Topic Modeling

- **Vectorization**:
  - Use `TfidfVectorizer` to convert the text data into a Document-Term Matrix (DTM) with specific parameters (`max_df=0.95`, `min_df=5`, and `stop_words='english'`).
  - Example:
    ```python
    tfidf = TfidfVectorizer(max_df=0.95, min_df=5, stop_words='english')
    dtm = tfidf.fit_transform(news_articles_df['news_summary'])
    ```

- **Non-negative Matrix Factorization (NMF)**:
  - Apply NMF with 5 topics to identify different themes within the news articles.
  - Example:
    ```python
    nmf_model = NMF(n_components=5, random_state=42)
    nmf_model.fit(dtm)
    ```

- **Top Words for Each Topic**:
  - Extract and print the top 15 words associated with each of the 5 topics identified by NMF.
  - Example Output:
    ```python
    The top 15 words for topic #1: ['government', 'rise', 'december', 'company', 'prices', ...]
    ```

### 3. Topic Assignment

- **Topic Transformation**:
  - Transform the DTM into topic probabilities for each document to determine the most likely topic for each article.
  - Example:
    ```python
    topic_results = nmf_model.transform(dtm)
    ```

- **Add Topic Labels**:
  - Assign topics and labels to each article by finding the topic with the highest probability and mapping it to a label.
  - Example Function:
    ```python
    def add_topic_labels(df, topic_results, topic_labels):
        df['topic'] = topic_results.argmax(axis=1) + 1
        df['topic_label'] = df['topic'].map(topic_labels)
    ```

- **Topic Labels**:
  - The topics have been labeled as follows:
    - Topic 1: Business
    - Topic 2: Entertainment
    - Topic 3: Politics
    - Topic 4: Sports
    - Topic 5: Technology

### 4. Final Output

- The DataFrame is updated with new columns for topics and their labels. The first and last 10 rows are displayed to validate that the articles have been appropriately categorized.

## Results

- **Topic Categorization**:
  - The NMF model successfully identified and categorized the news articles into distinct topics based on their content.
  - The assigned labels accurately reflect the main themes of the articles, as observed from the top words and the topics.

## Requirements

- Python 3.x
- Libraries: Pandas, NumPy, scikit-learn, re

## Installation

To install the required libraries, use the following pip command:

```bash
pip install pandas numpy scikit-learn
```

## How to Run

1. Ensure all required libraries are installed.
2. Load the dataset and preprocess the text data.
3. Apply `TfidfVectorizer` to create a Document-Term Matrix.
4. Use Non-negative Matrix Factorization (NMF) to identify topics.
5. Assign topics and labels to each article and display the results.

## Conclusion

This project demonstrates the effectiveness of using NMF for topic modeling on textual data. The model was able to identify distinct topics within the BBC news articles, and the results were appropriately labeled, providing a clear understanding of the underlying themes in the dataset.
