# ai-dev


No numpy, no pandas, no sklearn. Only Hardcore. 

Own AI/ML algorithms implementation from scratch and its testing.

The reason I do that: to learn internals of ml algorithms.

There are:
- [Regression](https://github.com/kotsky/ai-dev#regression): linear, non-linear, multi-variable.
- [Logistic Regression](https://github.com/kotsky/ai-dev#logistic-regression)
- [K-Nearest Neighbors](https://github.com/kotsky/ai-dev#k-nearest-neighbors)
- [K-Mean](https://github.com/kotsky/ai-dev#k-mean)
- Principal Component Analysis

To support data manipulation, I developed data_reader.py package instead of using pandas libs.
Data is stored in table-like data-structure in the main memory. 
It's nice to work with data sets which fit memory.

# Regression
1. Regression implementation.
2. About used data.
3. Results

### Regression implementation
Regression was implemented with:
- optimization algorithm: gradient descent
- learning rate
- regularization (penalty) L2 (Lasso)
- for various number of iterations
- logs writing for further debugging / plot

### About data
Fuel Consumption vs CO2 EMISSIONS in /data
### Results
Trained model shows as little as 3% error of prediction.

Workflow can be found here 

https://kotsky.github.io/projects/ai_from_scratch/regression_workflow.html
# Logistic regression
1. Logistic regression implementation.
2. About used data.
3. Results

### Logistic regression implementation
With:
- optimization algorithm: gradient descent
- learning rate
- regularization (penalty) L2 (Lasso)
- for various number of iterations
- logs writing for further debugging / plot
- adjustable logistic coefficient for prediction threshold
- evaluation with confusion matrix, precision and recall

### About data
Loan data in /data
### Results
Trained model has 71% accuracy that given people from a test data set will or not return loan taken before.

f1 score showed the best logistic threshold at 0.27, which we used to find out the best precision 74% and recall 95%.

Workflow can be found here 

https://kotsky.github.io/projects/ai_from_scratch/logistic_regression_workflow.html

78% accuracy was achieved by using standard libs (sklearn). 
Its workflow can be found here 

https://github.com/kotsky/ai-studies/blob/main/Projects/Project%20Loan/Loan%20Model.ipynb

# K-Nearest Neighbors
1. KNN implementation.
2. About used data.
3. Results

### KNN implementation
Normal KNN algorithm with optimized the nearest points storing/comparing 
based on a special k-size max heap data structure.

### About data
Loan data in /data. This is done to compare results with LR model.

### Results
We can say that k = 4 roughly is the best for our model, 
which gives 72% accuracy, 74% precision and 94% recall.

Its workflow can be found here: 

https://kotsky.github.io/projects/ai_from_scratch/knn_workflow.html

# K-Mean
1. K-Mean implementation.
2. About used data.
3. Results

### K-Mean implementation
Normal K-Mean algorithm based on distance calculation 
between centroids and training points. 

### About data
It's all about customers of some store. So we can 
try to identify clusters of these customers.

### Results
With a proper visualization and cost function analyse,
we find our that the best K is 5 for customers' features Income vs 
Years of Employed. 

Moreover, we could predict what cluster is the most 
suitable for a new customer.

Its workflow can be found here:

https://kotsky.github.io/projects/ai_from_scratch/kmean_workflow.html