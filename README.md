# Data Augmentation via Synthetic Oversampling (SMOTE)
## The most widely cited, empirically validated method to enlarge a training dataset—especially when facing class imbalance—is the Synthetic Minority Over-sampling Technique (SMOTE).

## SMOTE generates new, synthetic samples by interpolating between existing instances of the minority class. Instead of simply duplicating data points, SMOTE creates “in-between” records along the feature space, which has been shown to reduce overfitting and improve classifier generalization (https://www.jair.org/index.php/jair/article/view/10302)

### pip install pandas imbalanced-learn
### python smote_augment.py --input your_data.csv --factor 1.5
