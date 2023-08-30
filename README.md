# SubStrat: A Subset-Based Optimization Strategy for Faster AutoML

When the dataset is large, AutoML running times become increasingly high. 
We introduce SubStrat, an AutoML optimization strategy that tackles the data size, rather than configuration space. 
It wraps existing AutoML tools such as AutoSklearn, TPOT and H2O, and instead of execute them on directly on the entire dataset, SubStrat uses a genetic-based algorithm to find a small yet representative data subset which preserves a characteristic of the original one. It then employs the AutoML tool on the small subset, and finally, it refines the resulted pipeline by executing a restricted, much shorter, AutoML process on the large dataset.

SubStrat is based on the following paperes:

- Teddy Lazebnik, Amit Somech, and Abraham Itzhak Weinberg. [SubStrat: A
Subset-Based Optimization Strategy for Faster AutoML](https://www.vldb.org/pvldb/vol16/p772-somech.pdf). PVLDB, 16(4): 772 -
780, 2022. doi:10.14778/3574245.3574261

- Teddy Lazebnik, and Amit Somech. [Demonstrating SubStrat: A Subset-Based Strategy for Faster AutoML on Large Datasets.](https://dl.acm.org/doi/abs/10.1145/3511808.3557160) Proceedings of the 31st ACM International Conference on Information & Knowledge Management. 2022.

## New [SubStrat-Automl](https://github.com/analysis-bots/SubStrat)  python package
We released SubStrat-automl as a Python Package. Install it via
```
pip install substrat-automl
```
For more instruction and examples, please refer to the new [SubStrat github repository](https://github.com/analysis-bots/SubStrat) 

</br>
</br>
</br>


### Experiments reproducability - Table of contents
1. [Usage (paper code)](#usage)
2. [Data](#data)
3. [Experiments details](#experiments)
4. [Algorithm](#algorithm)
3. [Files structure](#files)
4. [Dependencies](#dependancies)

<a name="usage"/>

## Usage (paper code)

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
