## Data Augmentation via Synthetic Oversampling (SMOTE)
### The most widely cited, empirically validated method to enlarge a training dataset—especially when facing class imbalance—is the Synthetic Minority Over-sampling Technique (SMOTE).

### SMOTE generates new, synthetic samples by interpolating between existing instances of the minority class. Instead of simply duplicating data points, SMOTE creates “in-between” records along the feature space, which has been shown to reduce overfitting and improve classifier generalization (https://www.jair.org/index.php/jair/article/view/10302)

#### pip install pandas numpy scikit-learn imbalanced-learn
#### python smote_augment.py --input your_data.csv --factor 1.5
#### python smote_augment.py --input 5G-NIDD-NS.csv --factor 100
#### python smote_augment.py --input 5G-NIDD-S.csv --factor 100


### How It Works
#### Missing numeric values are imputed with column means.
#### Missing categorical values become a single “MISSING” category.
#### Categorical features are ordinally encoded; numeric features are standardized.
#### SMOTE or SMOTENC oversamples the minority class to orig_count * factor.
#### Encodings and scalings are inverted so the output CSV preserves original data types.

### SMOTE’s sampling_strategy parameter is flexible. You’re not limited to a single minority label:
# Example: Oversample two classes to different targets
sampling_strategy = {
    "class_A": 500,   # bring class_A up to 500 samples
    "class_B": 300    # bring class_B up to 300 samples
}
sm = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
X_res, y_res = sm.fit_resample(X, y)

