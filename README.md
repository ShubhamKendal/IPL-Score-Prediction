# IPL Score Prediction Using Deep Learning 🏏🤖

# Overview
This project uses deep learning (neural networks) to predict the final score of an IPL (Indian Premier League) match. The model is built using historical IPL data from 2007 to 2018 and predicts match scores based on several factors, such as team, opponent, batsman, bowler, and stadium. The aim is to provide accurate score predictions that can be used for strategic decision-making during matches and by fans to better understand potential outcomes.

# Project Objectives
The main goals of this project are:

1. Score Prediction: Develop a neural network model to predict the final score of an IPL match.
2. Factor Analysis: Understand the impact of various factors, such as team, players, and stadium, on the predicted score.
3. Model Accuracy: Improve prediction accuracy using advanced deep learning techniques.

# Key Technologies and Tools
  # Python: Main programming language used.
  # Libraries:
1. TensorFlow: Used for building, training, and optimizing the neural network.
2. Keras: Integrated with TensorFlow for creating and managing neural network layers.
3. NumPy: Numerical computing.
4. Pandas: Data manipulation and analysis.
5. Matplotlib & Seaborn: Data visualization.
6. Scikit-learn (sklearn): Preprocessing, evaluation metrics, and model selection.

# Data Description
The dataset contains the following key features:

1. Team: Name of the batting team.
2. Opponent Team: Name of the opposing team.
3. Batsman and Bowler: Names of the key players involved in the play.
4. Stadium: The stadium where the match took place.
5. Runs: Runs scored per ball at various stages of the match.
   
The dataset covers IPL matches from 2007 to 2018, providing extensive historical data for model training.

# Model Development
  # Data Preprocessing:
1. Data cleaning: Handling missing values, feature engineering (transforming categorical variables), and scaling numerical features.
2. Splitting the data into training and testing sets to evaluate model performance.

  # Neural Network Architecture:
A feedforward neural network with multiple hidden layers was built using TensorFlow and Keras.
1. Input layer: Takes in features such as team, batsman, bowler, and stadium.
2. Hidden layers: Several dense layers with activation functions like ReLU.
3. Output layer: Predicts the final match score.

  # Evaluation:
Metrics used: Root Mean Squared Error (RMSE) and R² score were used to assess model performance.
Results: The model achieved an RMSE of 12.5 and an accuracy rate of 85%, showing promising predictive ability.

# Analysis & Visualizations
  # Exploratory Data Analysis (EDA): 
    The dataset was analyzed using Pandas, Matplotlib, and Seaborn to understand key trends in the data, such as:
    1. Distribution of scores across different teams and stadiums.
    2. The impact of different players on the final match score.
  # Model Performance: 
    Visualized the learning curves, training/validation loss, and predicted vs actual scores using line and scatter plots.

# Key Findings
1. Team Influence: Certain teams had a consistent impact on match outcomes, showing higher average scores.
2. Player Influence: Top-performing batsmen and bowlers significantly influenced the predicted score.
3. Stadium Effect: Matches held in certain stadiums consistently showed higher scores due to pitch conditions.

# How to Use

# Dataset

# Future Work
1. Hyperparameter Tuning: Experiment with different neural network architectures and hyperparameters to improve model accuracy.
2. Feature Engineering: Incorporate additional features like weather conditions and player form to further enhance predictions.
3. Real-time Predictions: Develop a system for real-time score prediction during live matches.

# Contact
Feel free to reach out for questions :

Email: shubham.kendal@email.com
LinkedIn: [Your LinkedIn Profile(https://www.linkedin.com/in/shubhamkendal/)
