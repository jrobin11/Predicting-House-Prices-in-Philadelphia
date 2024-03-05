# Predicting-House-Prices-in-Philadelphia

# Data Collection & Preprocessing:
In my project, I gathered data from various real estate websites, public databases, and APIs to compile a dataset on Philadelphia houses, focusing on attributes like location, size, and other significant factors. One of the challenges I faced was dealing with missing values in the dataset. To tackle this, I employed imputation techniques to fill in these gaps, ensuring no valuable information was lost. Additionally, outliers posed a significant challenge as they could skew the results of my analysis. I addressed this by identifying and removing or adjusting these data points. Another crucial step was transforming categorical variables into a format that could be effectively utilized in modeling through encoding techniques. This preprocessing was vital in cleaning the data and making it ready for analysis, laying a solid foundation for the subsequent stages of my project.

#### Before Preprocessing:
Before preprocessing, my data was quite raw and cluttered, containing a wide range of columns, many of which had missing values (denoted as NaN) and the presence of outliers was evident. The dataset also included complex types like geometry data for locations (SRID and POINT data), and there was a significant amount of categorical data that wasn't yet suitable for analysis.

**Data:**
Steps:
1. Click on the link to view the file: [View Initial Data Before Preprocessing](initial_data_before_preprocessing.md).
2. Once the file is open in GitHub's default markdown view, click on the "code" button to view the  content of the file.

#### After Processing:
After preprocessing, the dataset transformed significantly. I focused on retaining relevant columns, dropping NaN values to ensure completeness, and converting categorical variables into a more analysis-friendly format. This refinement process led to a cleaner, more concise dataset that emphasized crucial features like the object ID, assessment date, basements, building description, category description, and other attributes related to the house's size, value, and location. This cleaner version was far more amenable to the analytical and modeling processes that followed, facilitating a smoother transition into feature engineering and model development.

**Data:**




