# Feature Liminality Analysis

This Jupyter Notebook (`FeatureLiminalityIPYNB.ipynb`) explores various methodologies for understanding and quantifying "Feature Importance" and "Feature Certainty" (or "Liminality") in machine learning contexts. It demonstrates three distinct approaches: a custom Genetic Algorithm for feature selection, analysis of input features using Mutual Information and average magnitude with a CNN model, and SHAP-based interpretability for a dense neural network.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Key Concepts](#key-concepts)
3.  [Methodologies Explored](#methodologies-explored)
    * [Genetic Algorithm for Feature Selection](#genetic-algorithm-for-feature-selection)
    * [CNN-Based Input Feature Analysis](#cnn-based-input-feature-analysis)
    * [SHAP-Based Feature Analysis](#shap-based-feature-analysis)
4.  [Visualizations](#visualizations)
5.  [Requirements](#requirements)
6.  [How to Run](#how-to-run)

## Project Overview

The goal of this project is to provide a comparative view of how different techniques identify important features and how consistently those features exhibit significance across iterations or different interpretability methods. This concept of consistency or reliability in feature selection is termed "Feature Certainty" or "Liminality."

## Key Concepts

* **Feature Importance:** A score indicating how much a particular feature contributes to a model's predictions or the target variable.
* **Feature Certainty (Liminality):** A measure of the consistency or robustness of a feature's importance. It can be interpreted as how often a feature is selected, its average magnitude, or the consistency of its SHAP values across different samples or model runs.

## Methodologies Explored

The notebook is structured into three main sections, each demonstrating a different approach to feature analysis:

### Genetic Algorithm for Feature Selection

This section implements a custom `GeneticAlgorithmModel` class.
* It performs binary feature selection using a genetic algorithm, where each "chromosome" represents a set of selected features.
* **Fitness Function:** Based on the sum of Mutual Information scores for the selected features.
* **Tracking:** It tracks the "feature importance" (Mutual Information) and "feature certainty" (normalized frequency of selection across generations and population).
* **Active Data Consideration (ADC):** The model also tracks the number of active features over generations, which can be seen as an "Active Data Consideration" percentage.

### CNN-Based Input Feature Analysis

This section trains a simple 1D Convolutional Neural Network (CNN) on a synthetic dataset and analyzes the input features.
* **Feature Importance:** Calculated using Mutual Information between the input features and the target variable.
* **Feature Certainty:** Derived as the average absolute value (magnitude) of each feature across the dataset. This metric helps understand the general presence or "activation" level of a feature.

### SHAP-Based Feature Analysis

This section trains a dense neural network and uses SHAP (SHapley Additive exPlanations) to interpret its predictions.
* **SHAP Values:** KernelExplainer is used to compute SHAP values, which represent the contribution of each feature to the prediction for individual instances.
* **Feature Importance:** Calculated as the mean absolute SHAP value for each feature across a set of test samples.
* **Feature Certainty:** Determined by the fraction of non-zero SHAP values for each feature across the samples. This indicates how consistently a feature contributes to predictions (positively or negatively) for different instances.

## Visualizations

Each section generates plots to visualize the relationship between "Feature Importance" and "Feature Certainty." Common plots include:
* Bar plots showing feature importance.
* Scatter plots overlaying feature certainty.
* Combined scatter plots where feature certainty is represented by color, allowing for a visual assessment of features that are both important and consistently impactful.
* SHAP summary plots for a global view of feature contributions.

## Requirements

To run this notebook, you will need the following Python libraries:

* `numpy`
* `matplotlib`
* `scikit-learn`
* `tensorflow`
* `shap`

You can install them using pip:

```bash
pip install numpy matplotlib scikit-learn tensorflow shap
```
## How to Run

1.  **Clone the repository** (if applicable) or download the `FeatureLiminalityIPYNB.ipynb` file.
2.  **Ensure you have Jupyter Notebook or JupyterLab installed.** If not, you can install it via pip:

    ```bash
    pip install notebook  # or pip install jupyterlab
    ```

3.  **Navigate to the directory** containing the notebook in your terminal.
4.  **Launch Jupyter Notebook/Lab:**

    ```bash
    jupyter notebook # or jupyter lab
    ```

5.  **Open `FeatureLiminalityIPYNB.ipynb`** from the Jupyter interface.
6.  **Run all cells** to execute the code and generate the plots.
