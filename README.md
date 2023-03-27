# SubStrat Architecture
## What the use case SubStrat solve:
    
    Reduce the time of AutoML computation time while retaining the output model performance.
    The strategy of SubStrat is reducing the size of the dataset for faster AutoML computation time.

## Requierments:
1. model accuracy needs to be similar to regular runs of AutoML for the same Dataset.
   * Less than 5% decrease in accuracy.
   * The user should be able to change the accuracy for faster results. 
2. By default implement the reduction size of the database using the algorithm shown in the article.
    * The user should be able to change the implementation of the reduction size of the data.
3. Support of multi AutoML platforms.
   * Easy interfaces for adding new platforms.
   * At the first one platform each time, maybe later multi platform at any time.
4. Easy to use
    * the flow is:
      * data reuction size(substrat by default)
      * Fit
      * Fine-tune
   * It should be easy for the user to run all the 3 steps, mainly because the user probably has lots of steps for his workflow, and adding another step should be easy.
5. Support any size of data, even data that can’t be loaded for the RAM at once.
    * At start SubStrat will only support only that can be loaded in on time to the RAM, 
    * In the future we will support bigger databases.

6. enable to compare?

<br>

### basic exaple of the architecure
```python

class BasicSummaryAlgorithm(object):
        def __init__(data: pd.DataFrame, target_column: str, desired_row_size=10, desired_col_size=5, accuracy=0.95, timeout = 30):
            pass
        
        def _evaluation_function(self, data: pd.DataFrame,  subset_data:pd.DataFrame) -> float:
            # We already have some implemented function the the user can use.
            raise NotImplemented
        
        def run(self) -> pd.DataFrame:
            raise NotImplemented
    
class SubStrat(object):
    def __init__(dataset_x, dataset_y, autoMLClassifier, default_algo:BasicSummaryAlgorithm=GreedySummary):
        pass

    def run_summary(summery_algorithm: BasicSummaryAlgorithm=GreedySummary) -> pd.DataFrame:
        return summery_algorithm.run()

    def run_model(self, algo:BasicSummaryAlgorithm=None):
        raise NotImplemented
    
    def compare?
```