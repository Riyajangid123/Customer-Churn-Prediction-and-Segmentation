## Customer Churn Prediction & Segmentation System 
 # Overview
  This project is a production-grade Machine Learning system that predicts customer churn and segments users into risk 
  categories. It combines:
  Churn Prediction using XGBoost
  Customer Segmentation using K-Means
  FastAPI for model serving
  Streamlit for interactive dashboard
  Docker for containerized deployment

  The system is designed to simulate a real-world ML pipeline from training → deployment → visualization.
 
# Key Features
- High-performance churn prediction (Recall: 0.81)
- Customer segmentation into Low, Medium, High risk
- REST API for real-time predictions
- Interactive dashboard for business users
- Fully containerized using Docker
- Efficient model loading using Joblib

# Machine Learning Pipeline
 1. Churn Prediction
Model: XGBoost Classifier
Metric: Recall = 0.81

Why Recall?
Missing churners is costly → high recall ensures maximum churn detection.

2. Customer Segmentation
Algorithm: K-Means Clustering
Clusters: 3

Segments:
Low Risk
Medium Risk
High Risk

3. Visualization

Scatter plots for cluster visualization
Helps stakeholders quickly interpret customer risk distribution

# Tech Stack

 Category	Tools Used
 Language	Python
 ML Libraries	Scikit-learn, XGBoost
 Data Handling	Pandas, NumPy
 Backend API	FastAPI
 Frontend UI	Streamlit
 Deployment	Docker, Docker Compose
 Model Storage	Joblib

 How to run docker
1. Build and Start Containers
docker-compose up --build
2. Access Services
   FastAPI: http://localhost:8000
   Streamlit: http://localhost:8501
   API Endpoints (FastAPI)
   Predict Churn
POST /predict
Sample Input:
{
  "feature1": 10,
  "feature2": 5,
  ...
}
Output:
{
  "churn_prediction": 1,
  "risk_segment": "High Risk"
}

Results
Metric	Value
Recall	0.81
Model	XGBoost
Clusters	3

Business Impact
- Identify high-risk customers early
- Reduce churn-related revenue loss
- Improve retention strategies
- Enable data-driven decision-making

# Key Highlights 
- End-to-End ML Pipeline
- Real-world Deployment (API + UI)
- Dockerized Microservices Architecture
- Clean and scalable code structure

# Future Enhancements
- Deploy on AWS (EC2)
- CI/CD pipeline integration
-Real-time streaming predictions

# Author
 Riya Jangid

