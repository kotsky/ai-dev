# ai-dev


No numpy, no pandas, no sklearn. Only Hardcore. 

Own AI/ML algorithms implementation from scratch and its testing.

The reason I do that: to learn internals of ml algorithms.

There are:
- [regression](https://github.com/kotsky/ai-dev/blob/main/regression_workflow.ipynb): linear, non-linear, multi-variable.
- (later) classification: KNN? SVM?

To support data manipulation, I developed data_reader.py package instead of using pandas libs.

# Regression
1. Regression implementation.
2. About used data.
3. Results

## Regression implementation
Regression was implemented with:
- optimization algorithm: gradient descent
- learning rate
- regularization (penalty) L2 (Lasso)
- for various number of iterations
- logs writing for further debugging / plot

## About data
Fuel Consumption vs CO2 EMISSIONS in /data
## Results
Trained model shows as little as 3% error of prediction.

Workflow can be found here 

https://github.com/kotsky/ai-dev/blob/main/regression_workflow.ipynb