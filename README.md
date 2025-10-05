<br>

# MLOps Assignment 1: House Price Prediction

-----

> This project implements a machine learning workflow to predict house prices using the Boston Housing dataset. It includes scripts to train and evaluate two regression models: a Decision Tree Regressor and a Kernel Ridge Regressor.

-----

## Prerequisites

Before you begin, ensure you have **Conda** (or Miniconda) installed on your system to manage the environment.

-----

## Installation and Setup

To set up the project and install the necessary dependencies, follow these steps in your terminal.

1.  **Clone the Repository**

    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create and Activate the Conda Environment** This command creates a new environment named `mlops_a1` with Python 3.9.

    ```bash
    conda create --name mlops_a1 python=3.9 -y
    conda activate mlops_a1
    ```

3.  **Install Required Packages** This command installs all the necessary libraries from the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

-----

## How to Run the Code

You can run the training scripts for each model directly from your terminal. Each script will load the data, train the model, evaluate it on the test set, and print the final **Mean Squared Error (MSE)** score to the console.

  * **To run the Decision Tree Regressor:**

    ```bash
    python train.py
    ```

  * **To run the Kernel Ridge Regressor:**

    ```bash
    python train2.py
    ```
