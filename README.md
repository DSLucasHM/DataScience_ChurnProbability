# Customer Churn Prediction with Neural Networks

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-green)

A robust neural network-based system for predicting customer churn with automated preprocessing, model training with cross-validation, and evaluation capabilities.

## 👨‍💻 Author

**Lucas Miyazawa** 

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/lucasmiyazawa/) 
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:lucasmiyazawa@icloud.com)

## 📚 About the Project

This customer churn prediction system uses deep learning techniques—specifically neural networks—to detect early signs of customer attrition. By analyzing behavioral patterns and demographic attributes, the model accurately identifies high-risk customers. This enables businesses to implement targeted retention strategies, reduce churn rates, and increase customer lifetime value. The system is designed to integrate seamlessly into existing workflows, providing actionable insights that support data-driven decision-making.

The model implements best practices in machine learning, including:
- Automated feature preprocessing for numerical and categorical data
- Cross-validation to ensure model robustness
- Hyperparameter optimization 
- Threshold optimization to balance precision and recall
- Comprehensive evaluation metrics

## 🌟 Key Features

- **Advanced preprocessing pipeline** for handling mixed data types
- **Neural network architecture** specifically designed for churn prediction
- **Cross-validation** for reliable performance estimation
- **Threshold optimization** to maximize business value
- **Comprehensive metrics** for model evaluation
- **Production-ready code** with save/load capabilities

## 🛠️ Technical Stack

- **Python**: Core programming language
- **TensorFlow/Keras**: Deep learning framework
- **scikit-learn**: For preprocessing and evaluation metrics
- **pandas**: Data manipulation
- **numpy**: Numerical computation
- **joblib**: Model serialization

## 🚀 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Set up virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Using the Jupyter Notebook

The project includes a comprehensive Jupyter notebook that walks you through the entire churn prediction workflow:

1. **Open the notebook**
   ```bash
   jupyter notebook Customer_Churn_Prediction_with_Neural_Networks.ipynb
   ```

2. **Run the cells in sequence** to:
   - Set up your environment
   - Preprocess your customer data
   - Create and train the neural network model
   - Evaluate performance
   - Generate and save predictions

3. **Customize parameters** in the notebook to match your specific dataset and business requirements

## 📂 Project Structure

```
/customer-churn-prediction/
├── main.ipynb                # Main notebook for training and evaluation
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation
├── Dataset/                  # Folder containing raw input data
│   └── customer_data.csv     # CSV file with customer information
├── Models/                   # Folder containing trained models and preprocessing tools
│   ├── model_churn_tf.keras  # Trained churn prediction model (Keras format)
│   └── preprocessor.pkl      # Serialized data preprocessor (Pickle format)
└── predictions/              # Folder containing output predictions
    └── churn_predictions.csv # CSV file with churn prediction results
```

## 📝 Dataset Information

The system expects a dataset with the following structure:
- A unique customer identifier column (default: 'CustomerID')
- A binary target column (default: 'Churn') with values 0 (not churned) and 1 (churned)
- A mix of numerical and categorical features describing customer characteristics and behavior

Common features include:
- Demographics (age, gender)
- Account information (tenure, subscription type, contract length)
- Behavior metrics (usage frequency, support calls, payment delays)
- Financial data (total spend)

## 🔍 Model Architecture

The default neural network architecture consists of:
- Input layer matching preprocessed feature dimensions
- Dense hidden layer with tanh activation and L2 regularization
- Dropout layer for preventing overfitting
- Single output neuron with sigmoid activation for binary classification

This architecture was selected after experimentation for optimal performance on churn prediction tasks.



## 🚧 Future Improvements

- [ ] Implement hyperparameter tuning with Bayesian optimization
- [ ] Add feature importance analysis
- [ ] Develop a simple web interface for model usage
- [ ] Implement model monitoring capabilities
- [ ] Add model interpretability features
- [ ] Support for imbalanced datasets with class weighting

Feel free to improve the prompt, customize the process, and reach out if you have any questions or suggestions.


Thank You!

