# Depression Prediction using various Supervised Machine Learning Models
This projects aims to predict depression using various supervised machine learning models.

Data source: https://zindi.africa/competitions/busara-mental-health-prediction-challenge/data

##Overview

A recent NUHS study states that 3 in 4 NUS students are at risk of depression as a result of pandemic measures imposed (Ang, 2020). This highlights the growing concern of depression in our community. However, despite depression being a common illness, it is difficult to diagnose accurately as not all patients present with the same symptoms (Canales, 2019). Depression is also likely to have a greater impact on individuals from lower socioeconomic backgrounds, as they are less likely to seek professional help due to negative sentiments and consultation costs. This project aims to find the most reliable supervised machine learning classifier that is able to produce the best performance evaluation at predicting whether an individual in the low socioeconomic status is likely to suffer from depression. The data used is based on a 2015 study conducted by the Busara Center in rural Siaya County, western Kenya. Our findings indicate that Extreme Gradient Boosting Classifier provides the best performance with the highest f1 score in predicting whether an individual is likely to suffer from depression. The optimal model can be deployed as an open-source package for providing free and accessible preliminary diagnosis for individuals.

##Evaluation metrics

###Model performance
The accuracy, precision score, recall score, f1 score, ROC_AUC score and the confusion matrix is generated for each model when evaluating the performance. Although the imbalance training
dataset is rebalanced using random undersampling, accuracy is still not an appropriate performance metric due to the downside in predicting false negatives. For our problem, f1 score is the primary performance metric in model evaluation and hyper-parameter tuning.

Given the nature of our problem, both false negatives and false positives are undesirable and need to be minimized. In the case of false negatives, individuals will not seek treatment because they falsely believe that they do not have depression. In the case of false positives, individuals will incur unnecessary costs seeking professional consultation and diagnosis. As our target audience belongs to the lower socioeconomic class, the cost is an important concern and incurring these costs will have a negative impact on their lives.

Since f1 score is the weighted average between recall and precision, it will only be high when both precision and recall are high, and low when both are low. This allows us to gain information regarding both precision and recall using a single metric.

###Feature importance
To determine feature importance of the various features used in all our models, we utilize Shapely Additive Explanation (SHAP) values as opposed to odds ratio. The regression coefficient is the estimated increase in the log odds of depression per unit increase in the value of the feature (Szumilas, 2010). In essence, since the odds ratio is a monotonic transformation from probability (UCLA, 2021), a feature with larger odds ratio (more than 1) implies that an observation is more likely to be depressed given a larger value for the feature (Minitab, 2022). However, the drawback of odds ratio is that it is scaled with the scale of the feature value, thus leading to possible distortions. We are also unable to determine the local importance of the feature from the odds ratio.
SHAP value measures the marginal contribution of each feature for the model prediction. The properties of SHAP such as local accuracy, missingness and consistency make it a more desirable metric for feature importance (Lundberg, 2018c). With SHAP value, we can not only understand how much each feature contributes (either positively or negatively) towards the target variable i.e. global interpretability, but also dive deeper into the SHAP values for each observation as well i.e. local interpretability (Trevisan, 2022). At the same time, SHAP can be used to explain feature importance in both tree-based and non-tree based models while odds ratio is only applicable to linear models (Lundberg, 2018b).
