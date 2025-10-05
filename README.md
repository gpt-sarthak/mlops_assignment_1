MLOps Assignment 1: House Price Prediction
This project implements a machine learning workflow to predict house prices using the Boston Housing dataset. It includes scripts to train and evaluate two regression models: a Decision Tree Regressor and a Kernel Ridge Regressor.

📋 Prerequisites
Before you begin, ensure you have Conda (or Miniconda) installed on your system to manage the environment.

⚙️ Installation
To set up the project and install the necessary dependencies, follow these steps in your terminal.

Clone the Repository:

Bash

git clone <your-repository-url>
cd <repository-directory>
Create and Activate the Conda Environment:
This command creates a new environment named mlops_a1 with Python 3.9.

Bash

conda create --name mlops_a1 python=3.9 -y
conda activate mlops_a1
Install Required Packages:
Install all the necessary libraries from the requirements.txt file.

Bash

pip install -r requirements.txt
▶️ How to Run the Code
You can run the training scripts for each model directly from your terminal.

To run the Decision Tree Regressor:

Bash

python train.py
To run the Kernel Ridge Regressor:

Bash

python train2.py
Each script will load the data, train the respective model, evaluate it on the test set, and print the final Mean Squared Error (MSE) score to the console.
