# AI: K-Mean

## Clusterization problem with K-Mean: what kind of customers do we have?

In this project I use my own designed K-Mean and DataTable packages to get clusters of our customers based on their attributes.

I'll walk you through this process of model definition, fine-tuning and evaluation. You might want to jump to the interesting section immediately.


1. [Section 1](#id_1)- preparation
2. [Section 2](#id_2)- model building
3. [Section 3](#id_3)- K-Mean first try
4. [Section 4](#id_4)- K-Mean full up
4. [Section 5](#id_5)- results


# Introduction
I had developed my own Pandas-like simplified package data_reader.py to fetch different kind of data faster than Pandas can do and contains main features for K-Mean AI like data preparation for training/testing, split data, adding new features, creating combined one, ploting and many others. 

In addition, to enhance my AI regression model's knowledge, I designed a kmean.py package, which is pre-configured to initialize 20 times different starting positions of centroids, 5 times iteration for each training and returning the best centroids based on min cost function which was achieved after model training. 

All these features and techniques I would like to show in this notebook.

For additional package usage, refer to doxy in its src code.

For that session, I'm going to use a /data/Cust_Segmentation.csv file, which contains table-like structure data of our customers.

# Section 1<a id='id_1'></a> - preparation


```python
# upload packages
import data_reader as dr
import clusterization.kmean as km
import matplotlib.pyplot as plt
import random
```


```python
# to have similar random results
random.seed(301)
```


```python
# data uploading and feature enablement
main_table = dr.DataTable("data/Cust_Segmentation.csv")
```


```python
# returns all features from our data
main_table.head
```




    ['Customer Id',
     'Age',
     'Edu',
     'Years Employed',
     'Income',
     'Card Debt',
     'Other Debt',
     'Defaulted',
     'Address',
     'DebtIncomeRatio']



Let's pick few features up for further analysis and K-Mean testing. I'm interested in attitude between Income vs Years Employed.


```python
plt.title("Income vs Years Employed")
main_table.plot("Income", "Years Employed")
```


    
![png](output_7_0.png)
    


Cool, let's apply K-Mean with 2 clusters. But before, let's scale our data to range -1...+1 for better training experience, assuming we calculate distance between points in same scale.


```python
 main_table.max_scaling()
```

    Column Customer Id was scaled
    Column Age was scaled
    Column Edu was scaled
    Column Years Employed was scaled
    Column Income was scaled
    Column Card Debt was scaled
    Column Other Debt was scaled
    Column Defaulted was scaled
    Column Address was scaled
    Column DebtIncomeRatio was scaled



```python
# new plot
plt.title("Scaled Income vs Years Employed")
main_table.plot("Income", "Years Employed")
```


    
![png](output_10_0.png)
    



```python
main_table.activate_features(["Income", "Years Employed"])
main_table.select_target("Customer Id")
```

    Feature Income was added
    Feature Years Employed was added
    Target Customer Id was added


# Section 2<a id='id_2'></a> - model building


```python
# split data
main_table.split_data(0.7, shuffle=True)
```

    Shuffle was done
    Data was split as follows: 0.7 training set and 0.30000000000000004 testing set



```python
# here we do training.testing data generation from our data set
min_max_info = main_table.get_min_max_features()
training_data = main_table.get_training_data()
test_data = main_table.get_testing_data()
training_data = training_data[0]
test_data = test_data[0]

# set labels for further data visualization
labels = main_table.get_labels()[0]
```


```python
# model initialization
model_kmean = km.KMean()
```


```python
# let's do K-Mean algo only 3 times for different randomly init centroids
model_kmean.NUMBER_OF_CENTROIDS_INITIALIZATION = 3
model_kmean.set_labels(labels)

# next command allows to visualizate every learning iteration
# during method .fit()
model_kmean.set_monitor(True)

# set 5 learning iterations
model_kmean.epoch = 5

# set 3 number of centroids
model_kmean.number_of_centroids = 3

# what data will be used for the learning?
model_kmean.set_training_data(training_data)

# additional in case of centroid random initialization in range min-max
# (not in the use right now)
model_kmean.set_min_max_ranges(min_max_info)

# we want to pick points up randomly from 
# our data set and make them as our initial centroids.
# to do that, set centroid_mode to None
model_kmean.centroid_mode = None 
```

# Section 3<a id='id_3'></a> - first try

Let's find our the best three centroids for our data. We are going to do 3 times random centroid initialization and to have 5 learning iterations for each.


```python
model_kmean.fit()
```


    
![png](output_19_0.png)
    



    
![png](output_19_1.png)
    



    
![png](output_19_2.png)
    



    
![png](output_19_3.png)
    



    
![png](output_19_4.png)
    



    
![png](output_19_5.png)
    



    
![png](output_19_6.png)
    



    
![png](output_19_7.png)
    



    
![png](output_19_8.png)
    



    
![png](output_19_9.png)
    



    
![png](output_19_10.png)
    



    
![png](output_19_11.png)
    



    
![png](output_19_12.png)
    



    
![png](output_19_13.png)
    



    
![png](output_19_14.png)
    


There is logs' saving during learning process. It stores the best logs which happened during all training session based on cost function (means every new centroid initialization gives different final cost function, so the model saves the training process and its centroids for the lowest achieved value of the cost function). Let's analyse them.


```python
cost_functions = model_kmean.cost_functions
axis_x = [x for x in range(1, model_kmean.epoch+1)]
plt.plot(axis_x, cost_functions)
plt.title("Cost function vs learning iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost function")
```




    Text(0, 0.5, 'Cost function')




    
![png](output_21_1.png)
    



```python
# Best centroids:
model_kmean.centroids
```




    [[0.05828239984056782, 0.06169536066651156],
     [0.09649010587881882, 0.25685919395944873],
     [0.1919339918674164, 0.5648802402971185]]



# Section 4<a id='id_4'></a> - full up

In this section I want to find out which K is the best to minimize cost function of the model for these 2 features: Income vs Years Employed.


```python
def store_model(k_mean_model):
    return [k_mean_model.centroids, model_kmean.cost_functions]

model_kmean.NUMBER_OF_CENTROIDS_INITIALIZATION = 5  # let's increase number of random tries
min_k = 2
max_k = 10  # how many k we want to explore
results_by_k = []  # to store resutls

for k in range(min_k, max_k+1):
    model_kmean.number_of_centroids = k
    model_kmean.fit()
    results_by_k.append(store_model(model_kmean))
    model_kmean.cost_functions = None  # reset best model results to store a new one
    # model_kmean.set_monitor(False)  # to disable visualization each step
```


    
![png](output_25_0.png)
    



    
![png](output_25_1.png)
    



    
![png](output_25_2.png)
    



    
![png](output_25_3.png)
    



    
![png](output_25_4.png)
    



    
![png](output_25_5.png)
    



    
![png](output_25_6.png)
    



    
![png](output_25_7.png)
    



    
![png](output_25_8.png)
    



    
![png](output_25_9.png)
    



    
![png](output_25_10.png)
    



    
![png](output_25_11.png)
    



    
![png](output_25_12.png)
    



    
![png](output_25_13.png)
    



    
![png](output_25_14.png)
    



    
![png](output_25_15.png)
    



    
![png](output_25_16.png)
    



    
![png](output_25_17.png)
    



    
![png](output_25_18.png)
    



    
![png](output_25_19.png)
    



    
![png](output_25_20.png)
    



    
![png](output_25_21.png)
    



    
![png](output_25_22.png)
    



    
![png](output_25_23.png)
    



    
![png](output_25_24.png)
    



    
![png](output_25_25.png)
    



    
![png](output_25_26.png)
    



    
![png](output_25_27.png)
    



    
![png](output_25_28.png)
    



    
![png](output_25_29.png)
    



    
![png](output_25_30.png)
    



    
![png](output_25_31.png)
    



    
![png](output_25_32.png)
    



    
![png](output_25_33.png)
    



    
![png](output_25_34.png)
    



    
![png](output_25_35.png)
    



    
![png](output_25_36.png)
    



    
![png](output_25_37.png)
    



    
![png](output_25_38.png)
    



    
![png](output_25_39.png)
    



    
![png](output_25_40.png)
    



    
![png](output_25_41.png)
    



    
![png](output_25_42.png)
    



    
![png](output_25_43.png)
    



    
![png](output_25_44.png)
    



    
![png](output_25_45.png)
    



    
![png](output_25_46.png)
    



    
![png](output_25_47.png)
    



    
![png](output_25_48.png)
    



    
![png](output_25_49.png)
    



    
![png](output_25_50.png)
    



    
![png](output_25_51.png)
    



    
![png](output_25_52.png)
    



    
![png](output_25_53.png)
    



    
![png](output_25_54.png)
    



    
![png](output_25_55.png)
    



    
![png](output_25_56.png)
    



    
![png](output_25_57.png)
    



    
![png](output_25_58.png)
    



    
![png](output_25_59.png)
    



    
![png](output_25_60.png)
    



    
![png](output_25_61.png)
    



    
![png](output_25_62.png)
    



    
![png](output_25_63.png)
    



    
![png](output_25_64.png)
    


    Centroid 3 doesn't have any point



    
![png](output_25_66.png)
    


    There is no point assigned to this 3 centroid



    
![png](output_25_68.png)
    



    
![png](output_25_69.png)
    



    
![png](output_25_70.png)
    



    
![png](output_25_71.png)
    



    
![png](output_25_72.png)
    



    
![png](output_25_73.png)
    



    
![png](output_25_74.png)
    



    
![png](output_25_75.png)
    



    
![png](output_25_76.png)
    



    
![png](output_25_77.png)
    



    
![png](output_25_78.png)
    



    
![png](output_25_79.png)
    



    
![png](output_25_80.png)
    



    
![png](output_25_81.png)
    



    
![png](output_25_82.png)
    



    
![png](output_25_83.png)
    



    
![png](output_25_84.png)
    



    
![png](output_25_85.png)
    



    
![png](output_25_86.png)
    



    
![png](output_25_87.png)
    



    
![png](output_25_88.png)
    



    
![png](output_25_89.png)
    



    
![png](output_25_90.png)
    



    
![png](output_25_91.png)
    



    
![png](output_25_92.png)
    



    
![png](output_25_93.png)
    



    
![png](output_25_94.png)
    



    
![png](output_25_95.png)
    



    
![png](output_25_96.png)
    



    
![png](output_25_97.png)
    



    
![png](output_25_98.png)
    



    
![png](output_25_99.png)
    



    
![png](output_25_100.png)
    



    
![png](output_25_101.png)
    


    Centroid 5 doesn't have any point



    
![png](output_25_103.png)
    


    There is no point assigned to this 5 centroid



    
![png](output_25_105.png)
    



    
![png](output_25_106.png)
    



    
![png](output_25_107.png)
    



    
![png](output_25_108.png)
    



    
![png](output_25_109.png)
    



    
![png](output_25_110.png)
    



    
![png](output_25_111.png)
    



    
![png](output_25_112.png)
    



    
![png](output_25_113.png)
    



    
![png](output_25_114.png)
    



    
![png](output_25_115.png)
    



    
![png](output_25_116.png)
    



    
![png](output_25_117.png)
    



    
![png](output_25_118.png)
    



    
![png](output_25_119.png)
    



    
![png](output_25_120.png)
    



    
![png](output_25_121.png)
    



    
![png](output_25_122.png)
    



    
![png](output_25_123.png)
    



    
![png](output_25_124.png)
    



    
![png](output_25_125.png)
    



    
![png](output_25_126.png)
    



    
![png](output_25_127.png)
    



    
![png](output_25_128.png)
    



    
![png](output_25_129.png)
    



    
![png](output_25_130.png)
    



    
![png](output_25_131.png)
    



    
![png](output_25_132.png)
    



    
![png](output_25_133.png)
    



    
![png](output_25_134.png)
    



    
![png](output_25_135.png)
    



    
![png](output_25_136.png)
    



    
![png](output_25_137.png)
    



    
![png](output_25_138.png)
    



    
![png](output_25_139.png)
    



    
![png](output_25_140.png)
    



    
![png](output_25_141.png)
    



    
![png](output_25_142.png)
    



    
![png](output_25_143.png)
    



    
![png](output_25_144.png)
    



    
![png](output_25_145.png)
    



    
![png](output_25_146.png)
    



    
![png](output_25_147.png)
    



    
![png](output_25_148.png)
    



    
![png](output_25_149.png)
    



    
![png](output_25_150.png)
    



    
![png](output_25_151.png)
    



    
![png](output_25_152.png)
    



    
![png](output_25_153.png)
    



    
![png](output_25_154.png)
    



    
![png](output_25_155.png)
    



    
![png](output_25_156.png)
    



    
![png](output_25_157.png)
    



    
![png](output_25_158.png)
    



    
![png](output_25_159.png)
    



    
![png](output_25_160.png)
    



    
![png](output_25_161.png)
    



    
![png](output_25_162.png)
    



    
![png](output_25_163.png)
    



    
![png](output_25_164.png)
    



    
![png](output_25_165.png)
    



    
![png](output_25_166.png)
    



    
![png](output_25_167.png)
    



    
![png](output_25_168.png)
    



    
![png](output_25_169.png)
    



    
![png](output_25_170.png)
    



    
![png](output_25_171.png)
    



    
![png](output_25_172.png)
    



    
![png](output_25_173.png)
    



    
![png](output_25_174.png)
    



    
![png](output_25_175.png)
    



    
![png](output_25_176.png)
    



    
![png](output_25_177.png)
    



    
![png](output_25_178.png)
    



    
![png](output_25_179.png)
    



    
![png](output_25_180.png)
    



    
![png](output_25_181.png)
    



    
![png](output_25_182.png)
    



    
![png](output_25_183.png)
    



    
![png](output_25_184.png)
    



    
![png](output_25_185.png)
    



    
![png](output_25_186.png)
    



    
![png](output_25_187.png)
    



    
![png](output_25_188.png)
    



    
![png](output_25_189.png)
    



    
![png](output_25_190.png)
    



    
![png](output_25_191.png)
    



    
![png](output_25_192.png)
    



    
![png](output_25_193.png)
    



    
![png](output_25_194.png)
    



    
![png](output_25_195.png)
    



    
![png](output_25_196.png)
    



    
![png](output_25_197.png)
    



    
![png](output_25_198.png)
    



    
![png](output_25_199.png)
    



    
![png](output_25_200.png)
    



    
![png](output_25_201.png)
    



    
![png](output_25_202.png)
    



    
![png](output_25_203.png)
    



    
![png](output_25_204.png)
    



    
![png](output_25_205.png)
    



    
![png](output_25_206.png)
    



    
![png](output_25_207.png)
    



    
![png](output_25_208.png)
    



    
![png](output_25_209.png)
    



    
![png](output_25_210.png)
    



    
![png](output_25_211.png)
    



    
![png](output_25_212.png)
    



    
![png](output_25_213.png)
    



    
![png](output_25_214.png)
    



    
![png](output_25_215.png)
    



    
![png](output_25_216.png)
    



    
![png](output_25_217.png)
    



    
![png](output_25_218.png)
    


    Centroid 6 doesn't have any point



    
![png](output_25_220.png)
    


    There is no point assigned to this 6 centroid



    
![png](output_25_222.png)
    



    
![png](output_25_223.png)
    



    
![png](output_25_224.png)
    



    
![png](output_25_225.png)
    



    
![png](output_25_226.png)
    



    
![png](output_25_227.png)
    



    
![png](output_25_228.png)
    



    
![png](output_25_229.png)
    



    
![png](output_25_230.png)
    


# Section 5<a id='id_5'></a> - results

Let's find our what we just trained!


```python
axis_x = [x+min_k for x in range(max_k - min_k + 1)]

centroids_by_k = []
cost_functions_by_k = []

for k_results in results_by_k:
    centroids_by_k.append(k_results[0])
    
    # return the min cost function for particular k
    cost_functions_by_k.append((k_results[1][-1]))
    
plt.plot(axis_x, cost_functions_by_k)
plt.title("Cost function vs K")
plt.xlabel("K")
plt.ylabel("Cost function")
```




    Text(0, 0.5, 'Cost function')




    
![png](output_28_1.png)
    


Cost functions for each K was taken as a cost function of the best trained model for particular number of centroids (K) of the last iteration (in our case - 5th iteration). Based on the picture above we can see that K = [2, 3, 5] is good. The best K based on the lowest cost functions is:


```python
min_cost_function = float("inf")
k_best = None
for idx in range(len(cost_functions_by_k)):
    k = idx + min_k
    if min_cost_function > cost_functions_by_k[idx]:
        min_cost_function = cost_functions_by_k[idx]
        k_best = k
print("Best K is", k_best)
```

    Best K is 5


Let's set centroids of K = 5 to our model and try to predict which cluster a new customer is belong to.


```python
customer_data = test_data[20]
customer_data  # income, years of employed
```




    [0.06, 0.15]




```python
best_centroids = centroids_by_k[k_best - min_k]  # this is our best centroids from our K = 5
best_centroids
```




    [[0.19454494774105843, 0.5726270955486146],
     [0.09118599771283756, 0.28134241729899934],
     [0.053608498534171364, 0.08063670915819787],
     [0.12748539892757962, 0.15752650013355549],
     [0.06080041797283175, 0.004928941724051094]]




```python
# set best learned centroids to our model
model_kmean.set_best_centroids(best_centroids)
```


```python
model_kmean.predict(customer_data)
```




    3



The customer with scaled data [0.06 Income, 0.15 Years of Employed] belongs to a cluster 4 (3 + 1).
