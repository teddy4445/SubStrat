# SubStrat: Faster AutoML with Measure-Preserving Data Subsets
Automated machine learning (AutoML) frameworks have become important  tools in the data scientists' arsenal, as they dramatically reduce the manual work devoted to the construction of ML pipelines.
Such frameworks intelligently search among millions of possible configurations of feature engineering steps, model selection and hyper-parameters tuning options, to finally output an optimal pipeline in terms of predictive accuracy. 

However, when the dataset is large, each individual configuration takes longer to execute, therefore the overall AutoML running times become increasingly high.
In this work we present SubStrat, an AutoML optimization strategy that tackles the data size, rather than configuration space. 
It wraps existing AutoML tools, and instead of execute them on directly on the entire dataset, SubStrat uses a genetic-based algorithm to find a small yet representative \textit{data subset} which preserves a characteristic of the original one. It then employs the AutoML tool on the small subset, and finally, it refines the resulted pipeline by executing a restricted, much shorter, AutoML process on the large dataset.

The project aims to answer an instance of the following general task:
Given a matrix (D := (R, C)) and a fitness function (F) such that D has |R| = M rows and |C| = N columns, 
and F: D -> |R. In addition, given 0 < m << M and 0 < n << N, the sizes of the subsets of the rows and columns of the original matrix. 
We wish to find subsets in sizes (m, n) such that:
S = min_{r, c} F(r, c). The resulted matrix 'S' is defined to be the summary of the matrix D.

We used the outcome of this task as part of time and resource optimization process in an autoML context. Namely, we find a data subset of the original dataset, compute autoML using the small matrix and fine-tune the resulted model's hyperparameters using the autoML tool on the entire dataset. 

### Table of Contents
1. [Usage](#usage)
2. [Data](#data)
3. [Experiments](#experiments)
4. [Algorithm](#algorithm)
3. [Files structure](#files)
4. [Dependencies](#dependancies)

<a name="usage"/>

## Usage 

1. Clone the repo
2. Install the '**requirements.txt**' file (pip install requirements.txt)
3. Put the relevant data (columns and file format are shown in the bottom of this file).
4. run the project from the **main.py** file (python main.py or python3 main.py)

* The "/experiments" contains a set of experiments to analyze the algorithms and some included in the manuscript.

<a name="data"/>

## Data 
At the current state, we are using 10 datasets that can be found in the "/big_data" folder.
The idea is to examine the summary performance on numerical (categorical data treated as numerical) datasets.

The data is from Kaggle and ICU. We tried to gather data from different fields, sizes, and distrebutions, links to the datasets are provided below:

1. Dataset #1 - https://www.kaggle.com/chronicenigma/airline-passenger-satisfaction-classification?scriptVersionId=59340276
2. Dataset #2 - https://www.kaggle.com/lazebnik4445/general-signal-processing
3. Dataset #3 - https://www.kaggle.com/ratnadeepgawade/carinsurancedata
4. Dataset #4 - https://www.kaggle.com/sahistapatel96/mushroom-classification
5. Dataset #5 - https://www.kaggle.com/lazebnik4445/air-quality
6. Dataset #6 - https://www.kaggle.com/kadirduran/bike-demand-visualization
7. Dataset #7 - https://www.kaggle.com/lazebnik4445/lead-generation-form
8. Dataset #8 - https://archive.ics.uci.edu/ml/datasets/Myocardial+infarction+complications
9. Dataset #9 - https://www.kaggle.com/lazebnik4445/heart-disease
10. Dataset #10 - https://archive.ics.uci.edu/ml/datasets/Poker+Hand

<a name="experiments"/>

## experiments
Given an input dataset and a target feature, we first directly employ an AutoML tool and obtain its output ML pipeline configuration.
Recall that the AutoML tools (we use the popular AutoSklearn and TPOT frameworks, as described below) apply sophisticated algorithms to prune non-promising ML pipeline configurations, and finally output the pipeline with the highest predictive accuracy on the specified target feature. 
We record both the running time and the accuracy of the resulted model, which serve as our primary baseline, denoted Full-AutoML. We then examine whether our subset based strategy can indeed reduce AutoML running times, and still generate ML pipelines as accurate as Full-AutoML. 
To generate the data subsets, we used \algo{} as well as 10 other baselines. For each instance, we compute the relative running time and accuracy w.r.t Full-AutoML. We report the following metrics: \textit{time-reduction}, which indicates how much time was saved. We used 10 popular datasets from Kaggle and UCI Machine Learning Repository, as shown above. We tried to represent a wide range of domains and datasets sizes.  

<a name="algorithm"/>

## Algorithm 
The algorithms can be found in the '/summary_algorithms' folder. 

<a name="dependancies"/>

## Dependencies 
- Python               3.7.1
- numpy                1.20.2
- matplotlib           3.4.0
- pandas               1.2.3
- scikit-learn         0.24.1
- seaborn              0.11.1
- opencv-python        4.5.3
- TPOT                 0.11.7
- slots                0.4.0
- scipy                latest
- auto-sklearn         latest

These can be found in the **requirements.txt** and easily installed using the "pip install requirements.txt" command in your terminal. 

## Optimal data subset's size per algorithm
We computed a grid search as shown in figure 4 in the manuscript for each one of the algorithms regarding the optimal data subset size. The results are as follows:
#### SubStract: sqrt(N) and 0.25M
#### IG-KM: sqrt(N) and 0.25M
#### MAB: sqrt(N) and 0.25M
#### IG-RAND: sqrt(N) and 0.25M
#### MC-100K: sqrt(N) and 0.25M
#### MC-100: 0.01 and 0.25M
#### KM: sqrt(N) and 0.5M
After obtaining these hyperparameters, the other (spesific) hyperparameters, if any, of each algorithm is found on 5 option grid-search approach such that the values to check picked manually. 
