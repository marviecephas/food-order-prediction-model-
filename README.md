# Food Order Prediction Model

A machine learning project to predict customer food orders based on preferences, location, and order history. This model uses supervised classification techniques to help restaurants and delivery apps forecast demand and personalize recommendations.



## 1. Project Overview

This project aims to build a classification model that predicts a customer's food order. By analyzing historical data—such as past orders, customer demographics, location, and time of day—the model learns patterns to forecast future choices. This can help businesses optimize inventory, staffing, and marketing efforts.

### Learning Outcomes
* **ML Workflow:** Gained a practical understanding of the end-to-end machine learning process.
* **Data Processing:** Learned to clean, transform, and prepare data for training, including handling missing values and encoding categorical variables.
* **Exploratory Data Analysis (EDA):** Used data visualization (Matplotlib/Seaborn) to understand feature relationships and inform model selection.
* **Model Selection:** Experimented with multiple supervised learning algorithms (e.g., Logistic Regression, Random Forest, XGBoost) to find the best performer.
* **Model Evaluation:** Assessed model performance using classification metrics like Accuracy, Precision, Recall, and F1-Score.

## 2. Dataset

The dataset used for this project is the **[Food Delivery Order History Data](https://www.kaggle.com/datasets/sujalsuthar/food-delivery-order-history-data)** from Kaggle. It contains 21,000+ order records with features such as:

* `Items in order`
* `Restaurant name`
* `Subzone` & `City`
* `Order Placed At`
* `Order Status`
* `Bill Subtotal`
* `Ratings` & `Reviews`

## 3. Tech Stack

* **Python 3.10**
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Matplotlib & Seaborn:** For data visualization.
* **Scikit-learn (sklearn):** For feature processing, model training, and evaluation.
* **Jupyter Notebook:** For interactive development and analysis.

## 4. Methodology

1.  **Data Cleaning:** Loaded the dataset, handled missing values, and corrected data types.
2.  **Feature Engineering:**
    * Extracted `time_of_day` (e.g., 'Breakfast', 'Lunch', 'Dinner') from the `Order Placed At` timestamp.
    * Processed the `Items in order` column to create a single, predictable target variable (e.g., 'primary_food_category').
    * Encoded categorical features like `Subzone` and `Restaurant name` using One-Hot Encoding.
3.  **Data Splitting:** Split the processed data into training (80%) and testing (20%) sets.
4.  **Model Training:** Trained and compared several classification models:
    * Logistic Regression (as a baseline)
    * Random Forest Classifier
    * XGBoost Classifier
5.  **Model Evaluation:** The Random Forest Classifier was selected as the final model based on the best balance of Precision and Recall (F1-Score) on the test set.

## 5. How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook Food_Order_Prediction.ipynb
    ```

## 6. Real-World Applications

* **Inventory Management:** Restaurants can forecast demand for specific items and optimize stock levels, reducing food waste.
* **Personalized Recommendations:** Food delivery apps can suggest items to users based on their learned preferences, improving user experience.
* **Staffing & Operations:** Kitchens can anticipate peak hours for specific types of orders (e.g., large pizza orders on a Friday night) and staff accordingly.
* **Targeted Marketing:** Marketers can develop personalized promotions based on customer preferences and order patterns.
