# General Instructions

## Plagiarism  
Plagiarism is the act of using another’s words, ideas, or organizational patterns without crediting or acknowledging the source. This includes work generated or modified by AI. In particular, the use of generative AI that violates the instructor’s articulated policy or using it to complete an assessment (e.g., project, midterm) in a way not explicitly permitted by the instructor will be considered a breach of academic integrity.  

# Specific Instructions  

To carry out the classification, clustering, and regression tasks, you may need to consider the following steps:  

1. **Data description & visualization** that aids in the comprehension of the problem.  
2. **Data pre-processing**.  
3. **Data/feature selection/evaluation**.  
4. **Decide how to split the data** between training and test sets (if not stipulated by the instructions).  
5. **Use multiple classifiers and evaluate the parameters of each classifier**: Try at least the following:  
   - Decision Trees  
   - One based on ensemble learning (especially consider Random Forests)  
   - Neural Networks  
6. **Use clustering algorithms and evaluate parameters**. Try at least two, e.g.:  
   - k-means  
   - Agglomerative (hierarchical) clustering  
7. **Regression**: Try at least  
   - Linear regression  
   - Polynomial regression  
   - Explore regularization  
8. **Evaluate**:  
   - The performance of each classifier: at least provide F1 measure, precision, recall, and ROC curves (if applicable) and AUC.  
   - Clusters based on criteria such as silhouette and inertia.  
   - Regression based on criteria such as the R score and others.  
9. **Observe findings and draw conclusions**.  
10. **Future work**: Also include things you might try/consider in the future.  

---

# 1. Anomaly Detection: Unsupervised Learning (25%)  

### Description  
In this problem, the task is to perform anomaly detection, i.e., to detect data points that may indicate unusual computer network behavior. The dataset pertains to computer network traffic and can be obtained from:  

- **Source Data**:  
  - [Kaggle](https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data/code)  
  - [UCI KDD Cup 1999 Dataset](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)  
- **Data file**: `kddcup.data_10_percent.gz`  

### Instructions  
- Summarize the dataset by discovering clusters, evaluating, and characterizing them.  
- Use **unsupervised learning methods**, such as:  
  - k-means  
  - DBSCAN  
- Detect possible abnormal behavior.  

---

# 2. Regression (20%)  

It is up to you to choose a regression problem, but you should inform the instructor and get approval for it.  

### Indicative Data Sources  
- [Kaggle](https://www.kaggle.com)  
- [Analytics Vidhya](https://www.analyticsvidhya.com/)  
- [Awesome Public Datasets](https://github.com/awesomedata/awesome-public-datasets)  
- [OpenML](https://www.openml.org/)  

The dataset should present some challenges, e.g., **size, missing values, or categorical features**.  

---

# 3. Predicting Outcome (30%)  

The dataset contains **120,919 movie plot text summaries** from the **Internet Movie Database (IMDB)** ([www.imdb.com](https://www.imdb.com)), labeled with one or more genres.  

## Task  
- **Predict the movie genre**.  
- Since there can be multiple genres per movie, this is a **multilabel classification problem**.  
- **Dataset Source**:  
  - [Dataset Link](https://www.uco.es/kdis/mllresources/#FoodtruckDesc)  
  - **Dataset Name**: IMDB  

---

# 4. Explainable AI (XAI)  

The explainability of machine learning models has become very important.  

### Task  
- Conduct research on **Shapley Additive Explanation (SHAP)**.  
- Examine the **explainability** of the best model found in Step 1 using SHAP.  
