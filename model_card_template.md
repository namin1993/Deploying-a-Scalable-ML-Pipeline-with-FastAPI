# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
* Model Date: January 10, 2026
* Model Version: Version 1.0.0
* Model Type: This model is a Random Forest Classifier designed to predict whether an individual’s income exceeds $50,000 based on census data. The model includes hyperparameter tuning via GridSearchCV, which optimizes for the F1 score using a 5-fold cross-validation process.

## Intended Use
This model is intended for educational and research purposes to demonstrate the end-to-end process of building, training, and evaluating a machine learning model. It is also useful for practical applications in income classification tasks based on demographic and occupational features.

## Training Data
The training dataset is derived from the "U.S. Census Bureau Adult Income" dataset saved under the path ./data/census.csv. The dataset includes categorical and numerical features, such as:

* Categorical Features: workclass, education, marital-status, occupation, relationship, race, sex, native-country

* Numerical Features: age, fnlgt (final weight), education-num, capital-gain, capital-loss, hours-per-week

* Label (target): Binary classification of income level (salary > $50,000 or <= $50,000)

Categorical Features are handled as so:
* They are one-hot encoded, creating a binary feature for each category.

* One-hot encoding allows for the model to learn separate patterns per category without assuming relationships between them.

Numerical Features are handled as so:
* Values are passed directly to the model.

* Random Forest Classifiers handle raw numerical ranges well, so scaling is not required. This allows for the real-world meaning of numerical values to be preserved. 

The training dataset is 80% of the census.csv spreadsheet.

## Evaluation Data
The evaluation dataset is the test set split from the original dataset. It follows the same preprocessing steps as the training data by borrowing the functions defined in the ./ml/model.py file, thereby ensuring compatibility with the trained model. Performance metrics are computed on this test dataset.

The test dataset is 20% of the census.csv spreadsheet.

## Metrics
The model’s performance is evaluated using the following metrics:

* Precision: Measures the proportion of true positive predictions among all positive predictions. In this context, how often predictions of income >50K are correct.

* Recall: Measures the proportion of true positives identified among all actual positives. In this context, how many actual >50K cases are correctly identified.

* F1 Score (F-beta): The harmonic mean of precision and recall.

### Model Performance:
The model performance is computed on slices of the data based on unique values of each categorical feature (workclass, education, education-num, marital-status, occupation, relationship, race, sex, native-country). The results are saved in a text file slice_output.txt for further analysis. Based on the slice_output.txt file, the metrics computed per categorical slice include:

* Sex
    * Female: Precision: 0.7672 | Recall: 0.5918 | F1: 0.6682
    * Male: Precision: 0.7284 | Recall: 0.6447 | F1: 0.6840

* Education
    * Bachelor’s: Precision: 0.7456 | Recall: 0.7712 | F1: 0.7582
    * HS-grad: Precision: 0.5830 | Recall: 0.4041 | F1: 0.4774
    * Master’s: Precision: 0.8424 | Recall: 0.8516 | F1 ≈ 0.8470

* Race
    * White: Precision: 0.7335 | Recall: 0.6404 | F1: 0.6838
    * Black: Precision: 0.7746 | Recall: 0.5914 | F1: 0.6707

Some slices like "native-country", "education", and "occupation", have very small sample sizes and report perfect scores (F1 = 1.0), which are statistically unreliable and should be interpreted with caution .

## Ethical Considerations
This project uses a Random Forest classifier trained on U.S. Census data to predict whether an individual’s income exceeds $50,000 annually. While the model demonstrates reasonable overall performance, several ethical considerations must be addressed before any real-world use.

Slice-based evaluation of the model reveals that performance varies across demographic groups such as sex, race, education, marital status, and native country. Some groups experience lower recall or precision, and several slices contain very small sample sizes, meaning that the model may systematically under- or over-predict income for certain populations. These disparities indicate that the model may reflect historical and societal income inequalities present in the Census data rather than objective earning potential. 

In addition, the model uses sensitive demographic attributes, which raises fairness and discrimination concerns if used in real-world decision-making contexts. Although the categorical features are included to evaluate fairness, their use in an actual application could enable direct or proxy discrimination if not carefully governed. Non-technical Users might also question the use of using a Random Forest Model if there is no clear explanation on how the model is predicting salary outcomes.

## Caveats and Recommendations
The slice-based evaluation file shows that model performance varies across demographic subgroups, with some groups experiencing lower recall or precision and others having very small sample sizes that produce unstable metrics. Because the model is trained on historical Census data and uses sensitive attributes such as sex, race, marital status, and native country, it may reflect and reinforce existing income inequalities rather than objective earning potential. 

As a result, this model should be used for educational or analytical purposes only, with predictions serving as decision support rather than automated decision authority. Any real-world use would require continued slice-level monitoring, careful handling or removal of sensitive features, minimum sample thresholds for subgroup analysis, and human oversight to mitigate bias and misuse.