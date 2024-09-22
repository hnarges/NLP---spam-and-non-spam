# NLP---spam-and-non-spam
This project aims to build a robust email classification model using deep learning architectures, handling text preprocessing, imbalanced data, and model comparisons to identify the best approach for spam detection.
Email Spam Detection Project
Text Preprocessing Steps
Remove HTML Tags:

Use the BeautifulSoup library to remove HTML tags.
Remove URLs:

Use Regular Expressions (RegEx) to identify and remove URLs.
Remove Punctuation:

Use RegEx to remove punctuation marks.
Convert to Lowercase:

Convert all text to lowercase to maintain uniformity.
Remove Stopwords:

Use the nltk library to remove stopwords from the text.
Remove Emojis and Emoticons:

Use RegEx to identify and remove emojis and emoticons.
Frequent Words in Spam and Non-Spam Emails
Steps:
Split Data into Spam and Non-Spam:

Divide the dataset based on labels (spam vs non-spam).
Calculate Word Frequencies:

Calculate the frequency of words in each category (spam, non-spam).
Display Frequent Words:

Visualize the most frequent words in each category and analyze how these words can help distinguish between spam and non-spam emails.
Analysis:
In spam emails, words related to promotions, prizes, discounts, etc., are common, such as “win”, “free”, “prize”, and “cash”.
In non-spam emails, conversational words like "hi", "meeting", "thank", and "regards" are frequent.
Specific spam-indicative words, like “call,” are highly recurrent, which may suggest promotional content urging users to contact a specific center.
Some words, like “2,” appear frequently in both categories, indicating that not all frequent words are reliable for classification.
Sample Distribution:
The dataset shows imbalance, with significantly more non-spam (ham) emails compared to spam emails.
Handling Imbalanced Data
Oversampling (Increase Minority Class):

Techniques:
Random Oversampling: Randomly duplicate minority class samples.
SMOTE (Synthetic Minority Over-sampling Technique): Create synthetic samples for the minority class.
Undersampling (Reduce Majority Class):

Techniques:
Random Undersampling: Randomly remove majority class samples.
Tomek Links / NearMiss: More advanced sampling techniques.
Balanced Classifiers:

Algorithms like RandomForest and GradientBoosting have parameters for balancing class distributions.
Applying SMOTE for Balancing Data
Steps:

Label Conversion:

Convert spam and ham labels to numerical values (e.g., 0 and 1).
Text to Feature Vectors:

Use TfidfVectorizer to transform text into feature vectors.
Train-Test Split:

Split the data into training and testing sets.
Apply SMOTE:

Use SMOTE to oversample the minority class in the training data, making the dataset more balanced.
Comparison of Sampling Techniques: SMOTE vs. RandomUnderSampler
Oversampling (SMOTE):
Pros:
Preserves original data while generating synthetic samples.
Useful for small datasets.
Cons:
Increases dataset size, potentially leading to overfitting.
Increases training time.
Undersampling (RandomUnderSampler):
Pros:
Reduces dataset size, speeding up model training.
May improve generalization by removing redundant data.
Cons:
May lose valuable information by removing data.
Choosing Between SMOTE and RandomUnderSampler
Factors to consider:

Data Size:
If the dataset is small or significantly imbalanced, SMOTE is recommended.
If the dataset is large, RandomUnderSampler can reduce model complexity.
For a dataset of 6,000 samples (4,825 non-spam and 747 spam), SMOTE is generally a better choice, ensuring the minority class gets enough representation without reducing the dataset size.

Tokenization in NLP
Tokenization is a crucial step in text preprocessing for NLP tasks. Here’s why:

Text Analysis and Feature Extraction:

Splitting text into tokens allows for easier text analysis and conversion into numerical features for machine learning models.
Reducing Complexity:

Tokenization helps manage text length and complexity, improving model performance.
Converting Text to Numerical Features:

Tokens are converted into numerical vectors using techniques like TF-IDF and word embeddings.
Reducing Dimensionality:

Instead of processing the entire text, tokenization breaks the text into manageable tokens, helping to reduce feature space.
Improving Language Models:

Tokenization is key to training modern language models like BERT and GPT, allowing them to process language more efficiently.
Model Performance Comparison: LSTM, RNN, and GRU
Model Performance:
| Metric        | LSTM    | RNN     | GRU     |
|---------------|---------|---------|---------|
| **Precision (Class 0)** | 0.81    | 0.80    | 0.86    |
| **Precision (Class 1)** | 0.85    | 0.77    | 0.82    |
| **Recall (Class 0)**    | 0.86    | 0.75    | 0.81    |
| **Recall (Class 1)**    | 0.80    | 0.81    | 0.87    |
| **F1-Score (Class 0)**  | 0.83    | 0.78    | 0.83    |
| **F1-Score (Class 1)**  | 0.83    | 0.79    | 0.84    |
| **Accuracy**   | 0.83    | 0.78    | 0.84    |

Key Insights:
LSTM vs RNN:

LSTM performs better than RNN due to its ability to remember long-term dependencies in sequences.
LSTM vs GRU:

GRU slightly outperforms LSTM in accuracy and F1-score, with fewer parameters and faster training time.
RNN vs GRU:

GRU shows significant improvement over RNN, due to its more efficient architecture.
Optimizer and Learning Rate Analysis
Based on testing different optimizers and learning rates:

Best Optimizer: RMSprop
Best Learning Rate: 0.001
