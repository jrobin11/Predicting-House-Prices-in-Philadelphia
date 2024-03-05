# Predicting-House-Prices-in-Philadelphia

# Data Collection & Preprocessing:
In my project, I gathered data from various real estate websites, public databases, and APIs to compile a dataset on Philadelphia houses, focusing on attributes like location, size, and other significant factors. One of the challenges I faced was dealing with missing values in the dataset. To tackle this, I employed imputation techniques to fill in these gaps, ensuring no valuable information was lost. Additionally, outliers posed a significant challenge as they could skew the results of my analysis. I addressed this by identifying and removing or adjusting these data points. Another crucial step was transforming categorical variables into a format that could be effectively utilized in modeling through encoding techniques. This preprocessing was vital in cleaning the data and making it ready for analysis, laying a solid foundation for the subsequent stages of my project.

#### Before Preprocessing:
Before preprocessing, my data was quite raw and cluttered, containing a wide range of columns, many of which had missing values (denoted as NaN) and the presence of outliers was evident. The dataset also included complex types like geometry data for locations (SRID and POINT data), and there was a significant amount of categorical data that wasn't yet suitable for analysis.

**Data:**
For proper view of the results, please refer to the juypter file.
Steps:
1. Click on the link to view the file: [View Initial Data Before Preprocessing](results/initial_data_before_preprocessing.md)
2. Once the file is open in GitHub's default markdown view, click on the "code" button to view the  content of the file.

#### After Processing:
After preprocessing, the dataset transformed significantly. I focused on retaining relevant columns, dropping NaN values to ensure completeness, and converting categorical variables into a more analysis-friendly format. This refinement process led to a cleaner, more concise dataset that emphasized crucial features like the object ID, assessment date, basements, building description, category description, and other attributes related to the house's size, value, and location. This cleaner version was far more amenable to the analytical and modeling processes that followed, facilitating a smoother transition into feature engineering and model development.

**Data:**
For proper view of the results, please refer to the juypter file.
Steps:
1. Click on the link to view the file: [View Cleaned Data After Preprocessing](results/cleaned_data_after_preprocessing.md)
2. Once the file is open in GitHub's default markdown view, click on the "code" button to view the  content of the file.

### One-Hot Encoding:
To ensure that the categorical variables in my dataset could be effectively utilized by machine learning algorithms, I performed one-hot encoding on variables such as 'basements', 'building_code_description', 'category_code_description', 'location', 'state_code', and 'street_name'. This technique transforms categorical columns into a format that provides a binary representation of categories, which is more suitable for the modeling algorithms to interpret.

#### One-Hot Encoded Data:
```
# One-hot encoding performed on these columns
categorical_cols = ['basements', 'building_code_description', 'category_code_description', 'location', 'state_code', 'street_name']
data_encoded_subset = pd.get_dummies(data_subset, columns=categorical_cols)
data_encoded_subset.head()

```
This encoding expanded my feature set, converting the categorical attributes into numerous binary features that represent the presence of each category with a 1 or 0.

Outlier Removal:
Additionally, I identified and addressed outliers in the numerical columns of the dataset by calculating Z-scores. The Z-score method enabled me to detect and remove data points that were several standard deviations away from the mean, which could have adversely affected the model's performance.

#### Data After Outlier Removal:
```
# Calculated Z-scores for numeric columns to identify outliers
numeric_cols = data_subset.select_dtypes(include=[np.number]).columns
z_scores = np.abs(stats.zscore(data_subset[numeric_cols]))
threshold = 3
data_no_outliers = data_subset[(z_scores < threshold).all(axis=1)]
data_no_outliers.head()
```
By setting a threshold, commonly 3, I was able to exclude anomalies that may represent rare or extreme market conditions, thus refining the dataset for more accurate predictive modeling.

These preprocessing steps were crucial for preparing the data for the predictive modeling phase of the project. They helped ensure that the input features accurately represented the information without being skewed by non-numeric data or extreme values that are not representative of the typical house in Philadelphia.

### Data Cleaning and Preparation:
In addition to handling outliers, I focused on ensuring the data quality by dealing with 'assessment_date'. I created a copy of the data subset to avoid any warnings and ensure data integrity. Then, I dropped any rows where 'assessment_date' was not available, as the accuracy of temporal information is critical for my analysis. Converting 'assessment_date' to a datetime format allowed me to work with it more effectively, and extracting the year provided me with an additional feature that could influence the model due to market changes over time.

### Data Sampling for Kernel Stability:
Due to the substantial size of the dataset and to ensure computational efficiency during the exploratory phase, I sampled a smaller fraction of the data, specifically 10%. This approach maintains the dataset's integrity while significantly improving the kernel's stability, allowing for faster iteration during model training and validation phases.

#### Sampled Data (10%):
By taking a 10% sample of the cleaned data (ensuring a random and representative subset with random_state=42), I was able to work with a more manageable dataset that facilitated a quick turnaround in exploratory data analysis and preliminary modeling. 

# Feature Engineering:
To bolster the model's predictive power, I engineered new features from the dataset. This included creating interaction terms to capture the combined effects of different features on the sale price. For instance, multiplying the number of bedrooms by the number of bathrooms provided a 'bed_bath_interaction' term, which offered a more nuanced attribute reflecting property utility. I also transformed certain skewed variables like 'market_value' and 'sale_price' by taking their logarithms, which helped in normalizing their distributions and reducing the impact of extreme values.

### Feature Engineering Enhanced:
My project's feature engineering phase was expanded by the creation of interaction terms like 'bed_bath_interaction'. This new feature encapsulates the combined effect of the number of bedrooms and bathrooms on the house price, offering a more comprehensive variable for my models to learn from.

Additionally, I applied logarithmic transformations to 'market_value' and 'sale_price' to address their skewed distributions. Logarithmic transformation is a powerful method when dealing with skewed data, as it helps in normalizing the data distribution and reducing the impact of extreme outliers.

These transformations, along with the extraction of dates and the creation of interaction terms, are sophisticated techniques that contribute significantly to the predictive power of my machine learning models.

# Final Data Preparation for Modeling:
Before training the models, I took steps to ensure that all features were numeric and no NaN values were present. This involved selecting only numerical data types and dropping any remaining NaN values. This stringent final data preparation guaranteed that the models received the cleanest possible data, free from any potential issues that could bias the results.

# Model Training:
With the data prepared, I proceeded to split it into training and testing sets, using 80% of the data for training and reserving 20% for testing. This split was performed to evaluate the model's performance on unseen data, ensuring that it generalizes well to new data outside of the training dataset. A LinearRegression model was then trained on the training set, which included all numeric features except for the target variable, 'sale_price'.

# Model Evaluation:
After training, the model's performance was evaluated on the testing set. The evaluation metric used was the Root Mean Squared Error (RMSE), which quantifies the model's prediction errors. An RMSE of 153,686.81 suggests the typical prediction error when estimating the sale price of houses. This value was considered relative to the average house price of $222,996.90 and a price range of $1,224,203.0 within the dataset, providing context to the magnitude of the RMSE.

By comparing the RMSE to the average house price and the price range, I could assess the error in the context of actual house prices in Philadelphia. It revealed that while the model captures the general trends, there is still room for improvement, possibly through more complex models, additional feature engineering, or tuning hyperparameters.


# Model Development:
For developing the predictive models, I initiated and trained a linear regression model as a baseline to understand the linear relationships within the data. To tackle the complexity of the housing market, I also employed a RandomForestRegressor. I performed hyperparameter tuning using grid search to optimize the model, exploring different configurations of n_estimators, max_depth, min_samples_split, and min_samples_leaf. This not only helped in improving the model's performance but also provided insights into the importance of each hyperparameter.

# Hyperparameter Tuning with RandomForestRegressor:
To improve upon my initial modeling efforts, I employed a RandomForestRegressor for its robustness and ability to model non-linear relationships. Recognizing the importance of hyperparameter tuning, I utilized GridSearchCV to systematically explore a range of hyperparameters across multiple folds of my data.

# Hyperparameter Search Space:
I defined a grid for hyperparameter tuning, including the number of trees (n_estimators), the maximum depth of trees (max_depth), the minimum number of samples required to split an internal node (min_samples_split), and the minimum number of samples required to be at a leaf node (min_samples_leaf). This comprehensive search was designed to find the best combination that minimizes the mean squared error of predictions.

# Cross-Validation and Grid Search Execution:
I performed the grid search with 5-fold cross-validation, which allowed the model to be more generalizable and reduced the risk of overfitting. The cross-validation process not only provided a robust estimate of the model's performance but also ensured that the best hyperparameters were selected based on a more reliable evaluation metric.

# Model Evaluation:
After the grid search, the best estimator was identified, and predictions were made on the testing set. The performance of the tuned RandomForestRegressor was evaluated using the RMSE, a common metric for regression problems. An RMSE of 144,879.80 was observed, which represents an improvement over my initial linear regression model. This improvement indicates that the RandomForestRegressor is capturing more complex patterns in the data.

The RMSE for the RandomForestRegressor was substantially lower than the linear model, suggesting that the ensemble method is better suited for this dataset. Moreover, the RMSE value relative to the average house price and the range of house prices provides a more nuanced understanding of the model's predictive capabilities across various segments of the housing market in Philadelphia.

# Reflections and Next Steps:
The reduction in RMSE demonstrates the value of hyperparameter tuning and model complexity. It encourages further exploration of ensemble methods and possibly even more advanced techniques like boosting or neural networks to push the boundaries of predictive accuracy.

In summary, the process of hyperparameter tuning has significantly enhanced the performance of my predictive model, as evidenced by the lower RMSE. It highlights the importance of exploring different model configurations to find the optimal solution for a given predictive task.

# Visualizations:
### Histogram:
![DistrubtionOfSalePrices](https://github.com/jrobin11/Predicting-House-Prices-in-Philadelphia/assets/73866458/d817ad8a-dbab-4a5b-af00-ae16d0e568be)

In analyzing the distribution of house sale prices within my dataset, as depicted in the accompanying histogram, I've observed a pronounced right-skewness indicating that the majority of homes in Philadelphia are sold at lower price ranges. This distribution is critical to my project as it highlights the commonality of more affordable housing and the relative scarcity of high-priced real estate transactions. 

The skew in the data also poses an interesting challenge for the predictive modeling aspect of the project. Traditional models like linear regression assume a normal distribution of the target variable and may not perform optimally with skewed data. Consequently, this insight has guided me to consider models that can handle non-normal distributions, such as decision trees and ensemble methods like random forests or gradient boosting machines.

Moreover, the long tail on the right side of the distribution suggests the presence of outliers, which could represent luxury properties or other atypical market transactions. These outliers could potentially distort the predictive accuracy of my models, prompting me to apply log transformations or use robust scaling techniques during the data preprocessing stage to mitigate their impact.

The overall sale price distribution is a valuable component of my exploratory data analysis, offering a clear visual representation of the market dynamics in Philadelphia and shaping the direction of my feature engineering and model selection processes.

### Histogram:
![distribution of residuals](https://github.com/jrobin11/Predicting-House-Prices-in-Philadelphia/assets/73866458/107431a1-6a2d-40db-921a-7e9917d93883)

The histogram above displays the distribution of residuals, which are the differences between the actual sale prices and the prices predicted by my linear regression model. Residual analysis is a critical aspect of regression diagnostics, providing insights into the model's accuracy and whether certain assumptions of the regression are met.

From the distribution, we can observe that the residuals are approximately normally distributed, centering around zero. This indicates that the model, on average, tends to predict the sale prices with reasonable accuracy. However, there are noticeable tails, especially in the positive direction, suggesting that the model underestimates the sale prices for some houses. This could be due to outliers in the data, or perhaps the model's inability to capture non-linear relationships within the more expensive properties in the dataset.

Moreover, the presence of residuals on both extremes implies that there is variability in the model's predictive performance across different price ranges. The model's tendency to both overestimate and underestimate the sale prices suggests areas where the model could potentially be improved, possibly by incorporating more features or applying different modeling techniques that can better handle the complexity of the data.

In conclusion, the residual distribution is a powerful diagnostic tool that has informed me about the efficacy of my linear regression model. It will guide me in further refining the model, either by re-evaluating feature selection, introducing regularization to reduce overfitting, or exploring more complex models that can account for the nuances in the Philadelphia housing market.

### Heatmap:
![heatmap](https://github.com/jrobin11/Predicting-House-Prices-in-Philadelphia/assets/73866458/e7cd317f-0312-4380-a3ab-8eadb3402686)

The correlation heatmap above has been instrumental in understanding the relationships between different variables in my Philadelphia housing dataset. This visualization has allowed me to pinpoint which features have a more significant impact on the sale price, which is the target variable for my predictive modeling project.

From the heatmap, we observe that 'market_value', 'number_of_bathrooms', 'number_of_bedrooms', and 'number_of_rooms' have relatively strong positive correlations with 'sale_price'. This suggests that as these features increase in value, the sale price of the houses tends to increase as well, which aligns with market expectations. On the other hand, 'objectid' and 'year_built' have a weak negative correlation with 'sale_price', indicating these variables do not significantly influence the sale price in the data that I have collected.

Interestingly, 'zip_code' shows a moderate negative correlation with many features, including 'market_value' and 'sale_price'. This could reflect the geographical differences in the housing market of Philadelphia, where certain areas have generally lower property values.

Understanding these correlations is critical for my project because it informs the feature selection phase for the machine learning models. I aim to focus on variables with a stronger correlation to 'sale_price' to enhance model accuracy. Additionally, the insight that location (as denoted by 'zip_code') may affect house prices could lead me to further investigate spatial analysis or include interaction terms that capture the essence of location more effectively in my models.

In conclusion, the heatmap has been a valuable tool for identifying the most relevant features to predict house prices in Philadelphia, and the resulting insights will be used to refine my predictive models further.

### Scatter Plot:
![scatterplot](https://github.com/jrobin11/Predicting-House-Prices-in-Philadelphia/assets/73866458/fcf98ba7-207f-4be4-a591-19fd5c345d5f)

This scatter plot represents a key part of the exploratory data analysis for my project, showing the relationship between 'Market Value' and 'Sale Price' of houses in Philadelphia. As the x-axis indicates the assessed market value, and the y-axis shows the sale price, we can observe a general positive correlation, suggesting that higher market values are often associated with higher sale prices.

However, the plot also reveals a non-linear pattern and a wide spread of sale prices at each market value level, indicating that other factors may be influencing the sale price beyond the assessed market value. This insight is crucial for the model development phase of my project, as it implies that a linear model may not fully capture the complexity of the market. Therefore, I'm considering using more sophisticated models that can handle non-linearity, such as polynomial regression or even ensemble methods that can tease out these nuanced relationships.

The presence of outliers, especially on the higher end of the sale price spectrum, poses additional questions. Are these luxury homes or simply anomalies? Addressing these outliers will be vital for improving model robustness and ensuring that it doesn't overfit to these extreme values.

Overall, the scatter plot drives home the point that while market value is a strong indicator of sale price, my predictive model will need to account for a multifaceted set of factors to accurately predict house prices in Philadelphia. The next steps in my project will involve delving deeper into feature selection and engineering to include variables that can help explain the variance in sale prices not accounted for by market value alone.





