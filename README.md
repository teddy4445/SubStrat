# Stability Feature Selection Dataset Summary

In this project, we obtain a "good" summary of a dataset based on the approach of stable feature selection.
We define a summary to be a "good" summary if it is both stable and optimize some condition we would like to preserve in a dataset.

The project aims to answer an instance of the following general task:
Given a matrix (D := (R, C)) and a fitness function (F) such that D has |R| = M rows and |C| = N columns, 
and F: D -> |R. In addition, given 0 < m << M and 0 < n << N, the sizes of the subsets of the rows and columns of the original matrix. 
We wish to find subsets in sizes (m, n) such that:
S = min_{r, c} F(r, c). The resulted matrix 'S' is defined to be the summary of the matrix D.

### Table of Contents
1. [Usage](#usage)
2. [Data](#data)
3. [Algorithm](#algorithm)
4. [Files structure](#files)
5. [Dependencies](#dependancies)

<a name="usage"/>

## Usage 

1. Clone the repo
2. Install the '**requirements.txt**' file (pip install requirements.txt)
3. Put the relevant data (columns and file format are shown in the bottom of this file).
4. run the project from the **main.py** file (python main.py or python3 main.py)

<a name="data"/>

## Data 
At the current state, we are using five datasets that can be found in the "/data" folder:
- **dataset_1_birds_sings.csv**: birds singing dataset from Kaggle.
- **dataset_2_headech_prodrom.csv**: clinical headache prodrom experiment from an clinical company.
- **dataset_3_liver-disorders.csv**: liver disorders clinical data from Kaggle.
- **dataset_4_mfeat-morphological.csv**: not sure about this one.
- **dataset_5_page-blocks.csv**: A page blocks analysis dataset from Kaggle.

The idea is to examine the summary performance on numerical (categorical data treated as numerical) datasets.

<a name="algorithm"/>

## Algorithm 
### Greedy algorithm
One can treat the dataset summary task as a double feature selection (FS) problem.
Namely, selecting (n) out of (N) features from (D) and (m) out of (M) features from (D.transpose()), respectively. 
Assuming unknown connection between the rows and columns of matrix (D), we can run a greedy algorithm on the each of of them in a turn.
Using the best result from the last iteration, in the following. This way, receiving an (local) optimal result. 

<a name="files"/>

## Files Structure
- **main.py**: Manage the running of the simulation with analysis of results for the paper and IO operations. This is used as a first attempt on the algorithm.
- **summary_process_score_functions.py**: A static class with methods for scoring dataset's summary.
- **movie_from_images_maker.py**: This class responsible for making videos from sequences of images.
- **greedy_summary_algorithm.py**: This class is a wrapper over the greedy summary algorithm.
- **analysis_converge_process.py**: This class analyze and plot the converge process of a summary algorithm.
- **summary_wellness_scores.py**: A static class with methods for evaluating the wellness of a summary.
- **multi_score_multi_ds_experiment.py**: This class generates a summary table of a summary's algorithm performance over multiple score functions, datasets, and summary sizes.
- **table.py**: A simple table class, allowing to populate pandas' dataframe object in an easy way for this programmer.
- **converge_report.py**: A class responsible to store and provide analysis on a summary's algorithm converge process.

<a name="dependancies"/>

## Dependencies 
- Python               3.7.1
- numpy                1.20.2
- matplotlib           3.4.0
- pandas               1.2.3
- scikit-learn         0.24.1
- seaborn              0.11.1
- opencv-python        4.5.3
- scipy                latest

These can be found in the **requirements.txt** and easily installed using the "pip install requirements.txt" command in your terminal. 
