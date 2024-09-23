# Ames Horizon

<!-- **Hazelnut: Nutty Precision for your SQL Queries** -->

<!-- [![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/yourusername/hazelnut/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) -->

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [License](#license)

<!-- - [Project Demo](#project-demo) -->
<!-- - [API Usage](#api-usage) -->
<!-- - [Guidelines](#guidelines) -->

## Introduction

This project aims to predict housing prices in Ames, Iowa using a machine learning regression model. The dataset includes detailed features on residential properties, allowing for accurate price predictions. The project incorporates extensive data preprocessing, feature engineering, exploratory data analysis (EDA), outlier detection, and multicollinearity analysis. A complete training pipeline is built using ZenML, applying Strategy and Template design patterns to enhance modularity and scalability. Python and Scikit-learn are used for core model development.

<!-- ## Project Demo
Here is a demonstration of Hazelnut in action:

[![Hazelnut Demo](https://img.youtube.com/vi/5KhLWRgA0XA/0.jpg)](https://www.youtube.com/watch?v=5KhLWRgA0XA) -->

## Features
- **Regression Model**: Predict housing prices based on multiple property attributes.
- **Data Preprocessing**: Handles missing values, feature encoding, and scaling.
Feature Engineering: Generates meaningful features from the dataset.
- **EDA and Visualization**: In-depth exploration of key features and correlations.
Outlier Detection: Identifies and handles outliers for more accurate predictions.
- **Multicollinearity Analysis**: Detects correlated features to prevent model instability.
- **Training Pipeline**: Built using ZenML for automated and reusable workflows.

## Installation

To get started with Ames Horizon, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Aditi-Asati/Ames-Price-Prediction.git
    cd Ames-Price-Prediction
    ```

<!-- 3. **Set Up Environment Variables**:
    Create a `.env` file in the root directory and add your database credentials.
    ```plaintext
    DB_HOST=your_database_host
    DB_USER=your_database_user
    DB_PASS=your_database_password
    DB_NAME=your_database_name
    ``` -->



## License

Ames Horizon is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for more details.


<!-- 
### running the api

execute

```
python -m src.api.api
```

from project root -->