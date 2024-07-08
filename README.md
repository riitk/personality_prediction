# personality_prediction
Personality Prediction Using ML

# Personality Prediction using Machine Learning

This project aims to predict the personality type of individuals based on various psychological traits using machine learning techniques. The dataset includes several attributes related to personality traits, and the goal is to classify the personality type.

## Dataset

The dataset consists of the following columns:

- `Gender`: Gender of the individual (Female/Male)
- `Age`: Age of the individual
- `openness`: Openness to experience score
- `neuroticism`: Neuroticism score
- `conscientiousness`: Conscientiousness score
- `agreeableness`: Agreeableness score
- `extraversion`: Extraversion score
- `Personality`: Personality type (target variable)

The train dataset has 709 rows, and the test dataset also has 709 rows, making a total of 1418 rows after concatenation.

## Data Preprocessing

1. **Concatenation**: 
   - Combined train and test datasets using `pd.concat([train, test], axis=0)`.

2. **Basic Information**: 
   - Explored the dataset to find the number of null values, shape, and descriptive statistics using `df.info()`, `df.shape`, and `df.describe()`.

3. **Value Counts**: 
   - Analyzed the distribution of the `Personality` column using `df['Personality'].value_counts()`.

4. **Data Visualization**: 
   - Visualized data using countplots, barplots, and histograms to study the relationships between columns.

5. **Gender Encoding**: 
   - Converted the categorical `Gender` column to numerical using `df["Gender"] = df['Gender'].map({"Female": 0, "Male": 1})`.

6. **Correlation Analysis**: 
   - Found the correlation of the `Personality` column with other features in a sorted manner using `df.corr()["Personality"].sort_values()`.
   - Plotted a heatmap to visualize the correlations.

7. **Train-Test Split**: 
   - Split the data into training and testing sets.

8. **Data Scaling**: 
   - Scaled the data using `MinMaxScaler` from `sklearn`.

## Models Used

1. **Linear Regression**: 
   - Trained a model using Linear Regression.
   
2. **Gaussian Naive Bayes (GaussianNB)**: 
   - Trained a model using GaussianNB.

3. **Random Forest Classifier**: 
   - Trained a model using Random Forest Classifier.

## Evaluation

The models were evaluated using the accuracy score.

### Results

- **Linear Regression**: Achieved an accuracy score of 0.0395.
- **Gaussian Naive Bayes**: Achieved an accuracy score of 0.3725.
- **Random Forest Classifier**: Achieved an accuracy score of 0.3529.

### Reasons for Low Accuracy

The low accuracy scores may be attributed to the following reasons:

1. **Imbalanced Dataset**: The dataset may have an imbalanced distribution of personality types, leading to biased model performance.
2. **Feature Relevance**: The selected features may not be strong predictors of personality, resulting in poor model performance.
3. **Model Choice**: Linear Regression is generally not suitable for classification tasks. Although GaussianNB and Random Forest are more appropriate, they may still struggle with the given feature set and data characteristics.

## How to Use

### Cloning the Repository

To clone the repository, use the following command:

```bash
git clone https://github.com/yourusername/personality-prediction.git
cd personality-prediction
