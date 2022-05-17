
# Link Prediction Challenge - Kaggle 2022
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/) [![PythonVersion](https://camo.githubusercontent.com/fcb8bcdc6921dd3533a1ed259cebefdacbc27f2148eab6af024f6d6458d5ec1f/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f707974686f6e2d332e36253230253743253230332e37253230253743253230332e38253230253743253230332e392d626c7565)](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)
 ## Overview
 
The challenge of predicting the presence of a link between two nodes in a network is known as link prediction. Here we will solve the problem of predicting if a research publication will cite another research paper. For that, we have access to a citation network that includes hundreds of thousands of research publications, as well as their abstracts and author lists.

The pipeline used to solve this problem is identical to that used to solve any classification problem; the goal is to learn the parameters of a classifier using edge information, and then use the classifier to predict whether two nodes are related by an edge or not. Our goal in this project is to transform the different types of data, i.e. abstracts, authors and citation graph to create a feature matrix that we can feed to the classifier that will tackle the link prediction problem. Our model performance will be evaluated with the log loss metric.

This model was created for the following [Kaggle competition](https://www.kaggle.com/c/altegrad-2021/leaderboard) for the 2021/2022 Advanced learning for text and graph data course. It is ranked **TOP 1** both on the public and private learderboard.

## Team
The team **OverTen** is composed by **Xavier Jiménez**, **Jean Quentin** and **Sacha Revol**.

# Submission
Best submission and results on the validation dataset can be reproduced using the `best_submission.ipynb` file.

# Preprocessing
File `Preprocessing.ipynb` handles preprocessing for abstracts, authors and graph data.

# Feature matrix creation & evaluation
File `ALTEGRAD_project_v2.ipynb` handles the different steps for matrix creation and evaluation (i.e. LR, RF, XGBoost, LGBM, CatBoost)
File `nn-classifier.ipynb` implements the MLP classifier.

# Author Graphs
Files `weighted_co_authors_graph.py`, `utils.py` and `citation_graph.py` handle authors Graph creation

# Embeddings
Files `*_embedding.py/ipynb` handle abstract and graph node embeddings.

# Hyperparameter Optimization
Files `*_optimization.ipynb` find best hyperparameters for a given model using HyperOpt package.



