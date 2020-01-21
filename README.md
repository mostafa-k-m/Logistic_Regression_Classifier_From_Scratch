# Logistic Regression Model
## Importing the model
Our model is a class packaged in the **LogisticRegression.py** file for convenience purposes.


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from LogisticRegression import LogisticRegression
%matplotlib inline
plt.rcParams["figure.figsize"] = (20,10)
```

---

# Experiments to help understand the model
In these experiments we will use two-feature binary class datasets and try to find the optimal wieights for our classifier by manipulating the no. of iterations and the learning rate.

## First Experiment: Fitting the Model to a Reduced Iris Dataset
We Will start with a simple experiment. We are using a subset of the iris dataset `iris-reduced.csv`, we are using two classes and two feautres. The two classes are highly separated, this allows us to understabd more about the how the model works.

Provided the learning rate is $\eta$ and the number of iterations is $\tau$

### High # of iterations, low learning rate
- $\tau$ = 150,000
- $\eta$ = 1e-5


```python
Data=np.loadtxt('iris-reduced.csv')
X=Data[:,1:3]
y=Data[:,4]
classifier = LogisticRegression(X,y)
f = classifier.train(150000, 1e-5)
```

    @ no of iterations = 150000 and learning rate = 1e-05 
     final W = [[-0.07708747]
     [-0.41811067]
     [ 0.56088464]] 
     final cost = [0.34484796]



![png](output_5_1.png)


**We notice that, despite the high number of iterations, the learning gradient is not steep enough. this suggest that our learning rate was too low.**

### Reducing # of iterations and increasing the learning rate
- $\tau$ = 50,000
- $\eta$ = 1e-3


```python
classifier = LogisticRegression(X,y)
f = classifier.train(50000, 1e-3)
```

    @ no of iterations = 50000 and learning rate = 0.001 
     final W = [[-0.46054026]
     [-2.23480103]
     [ 2.66065312]] 
     final cost = [0.01812753]



![png](output_7_1.png)


**We notice that with this particular dataset, due to how widely separated the classes are, we can use very high learning rates and very low number of iterations.**

### Trying a low # of itertions and a very high learning rate
- $\tau$ = 1000
- $\eta$ = 0.5


```python
classifier = LogisticRegression(X,y)
f = classifier.train(1000, 0.5, draw_history = False)
```

    @ no of iterations = 1000 and learning rate = 0.5 
     final W = [[-0.77434097]
     [-3.62275983]
     [ 4.33845848]] 
     final cost = [0.00230502]



![png](output_9_1.png)


---
$\pagebreak$

## Second Experiment: Fitting the to Model animesh-agrwal's Student Exams Dataset
We will use a dataset from [animesh-agrwal's github](https://github.com/animesh-agarwal/Machine-Learning/blob/master/LogisticRegression/data/marks.txt) originally used in a course on logistic regression he made, the file attaches is `animesh-agarwal.csv`. the dataset is not completely linearly separable, meaning our decision boundary will end up misclassifying some points regardless of how much we optimize.

### A too high learning rate
- $\tau$ = 10,000
- $\eta$ = .01


```python
dataxy = np.loadtxt('animesh-agarwal.csv', dtype=float ,delimiter=',')
X = dataxy[:,(0,1)]
y = dataxy[:,2]
classifier = LogisticRegression(X,y)
f = classifier.train(10000, .01)
```

    @ no of iterations = 10000 and learning rate = 0.01 
     final W = [[-7.65900397]
     [ 0.41024768]
     [-0.05324509]] 
     final cost = [4.60514431]



![png](output_12_1.png)


**The error curve above shows that the learning rate is way too high, we will modify our parameters accordingly.**

### Effects of the Learning Rate and No. of Iterations on accuracy
#### Learning Rate vs. Accuracy
We begin by trying differen learning rates at 100,000 iterations. We will plot the result on a semilog scale to find the optimal learning rate.


```python
dict = {}
learning_rates = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
for i in learning_rates:
    classifier = LogisticRegression(X,y)
    f = classifier.train(100000, i, results = False)
    y_predict = classifier.predict(X)
    Accuracy = 100*np.sum(y == y_predict)/len(y)
    dict[i] = Accuracy
f, ax = plt.subplots()
ax.set(xscale='log',title='Learning Rate vs Accuracy', xlabel='Learning Rate', ylabel='Accuracy')
sns.lineplot(list(dict.keys()),list(dict.values()), label = "learning rate vs accuracy, @ 100,000 iterations", ax = ax)
ax.legend(loc='lower center', bbox_to_anchor=(.5, -.3), ncol=1)
```




    <matplotlib.legend.Legend at 0x7fe933ecbf50>




![png](output_14_1.png)


#### No. of Iterations vs. Accuracy
we will take the best learning rate from above and try it with different numbers of itration, increasing iterations with every experiment.


```python
dict = {}
for i in range(10000,200001, 50000):
    classifier = LogisticRegression(X,y)
    f = classifier.train(i, 1e-3, results = False)
    y_predict = classifier.predict(X)
    Accuracy = 100*np.sum(y == y_predict)/len(y)
    dict[i] = Accuracy
f, ax = plt.subplots()
ax.set(title='No. of iterations vs Accuracy', xlabel='No. of iterations', ylabel='Accuracy')
sns.lineplot(list(dict.keys()),list(dict.values()), label = "No. of iterations, @ 1e-3 learning rate", ax = ax)
ax.legend(loc='lower center', bbox_to_anchor=(.5, -.3), ncol=1)
```




    <matplotlib.legend.Legend at 0x7fe93359f410>




![png](output_16_1.png)


### Measuring discrimination with near optimal parameters
Based on the above, we will try to look at the decision line at:
- $\tau$ = 100,000 and $\eta$ = .001
- $\tau$ = 250,000 and $\eta$ = .001


```python
classifier = LogisticRegression(X,y)
f = classifier.train(100000, .001)

y_predict = classifier.predict(X)
Accuracy = 100*np.sum(y == y_predict)/len(y)
print("\n")
print(f"Points on the correct side of the decision boundary = {Accuracy}%")
```

    @ no of iterations = 100000 and learning rate = 0.001 
     final W = [[-4.81180027]
     [ 0.04528064]
     [ 0.03819149]] 
     final cost = [0.38737536]
    
    
    Points on the correct side of the decision boundary = 91.0%



![png](output_18_1.png)


- $\tau$ = 250,000
- $\eta$ = 0.001

**We think these parameters are close to optimal. The decision boundary classifies 92% of the points correctly, that is fair given that the classes are not linearly separable.**


```python
classifier = LogisticRegression(X,y)
f = classifier.train(250000, .001, draw_history = False)

y_predict = classifier.predict(X)
Accuracy = 100*np.sum(y == y_predict)/len(y)
print("\n")
print(f"Points on the correct side of the decision boundary = {Accuracy}%")
```

    @ no of iterations = 250000 and learning rate = 0.001 
     final W = [[-8.42279005]
     [ 0.07308283]
     [ 0.06668396]] 
     final cost = [0.29757435]
    
    
    Points on the correct side of the decision boundary = 92.0%



![png](output_21_1.png)


---
$\pagebreak$

# Testing the model prediction capabilities
In this section we will use larger and more complex binary class dataset and divide them into training and test data to test our predictions.

One of the restrictions of this model is that the labels have to be (0,1) encoded, for that we will use the `LabelEncoder` function from scikit learn.

We will also use `train_test_split` function from scikit. 


```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
```

We will also define a function to draw a confusion matrix.


```python
import scikitplot as skplt
def confusion(y_test, y_predict):
    fig, ax = plt.subplots()

    skplt.metrics.plot_confusion_matrix(
        y_test, 
        y_predict,
        ax=ax)

    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)
    plt.show() 
```

---

## First experiment B.D. Ripley's Synthetized Dataset
We will use the [prnn_synth](https://www.openml.org/d/464) dataset, a synthetized data from `Pattern Recognition and Neural Networks' by B.D. Ripley. Cambridge University Press (1996)  ISBN  0-521-46086-7.` the file attached is `prnn_synth.csv`

This dataset contains 250 instances, 2 features and 2 classes. the classes are not completely separable.

### Importing the data


```python
df = pd.read_csv('prnn_synth.csv')
df.dropna(how="all", inplace=True)

class_in_strings = lambda x: 'Class ' + str(x)
df['yc']= df["yc"].apply(class_in_strings)

df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>xs</th>
      <th>ys</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>250.000000</td>
      <td>250.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.072758</td>
      <td>0.504362</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.489496</td>
      <td>0.254823</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.246525</td>
      <td>-0.191313</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.509234</td>
      <td>0.323365</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.041834</td>
      <td>0.489827</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.369964</td>
      <td>0.704390</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.861296</td>
      <td>1.093178</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pairplot(data = df, hue = "yc")
```




    <seaborn.axisgrid.PairGrid at 0x7fe923363ad0>




![png](output_30_1.png)



```python
X = np.asarray(df[["xs", "ys"]])
y = np.asarray(df["yc"])

enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y)
```

### Experimenting With Parametes
#### Learning Rate vs Accuracy


```python
dict = {}
learning_rates = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
for i in learning_rates:
    classifier = LogisticRegression(X,y)
    f = classifier.train(50000, i, results = False)
    y_predict = classifier.predict(X)
    Accuracy = 100*np.sum(y == y_predict)/len(y)
    dict[i] = Accuracy
f, ax = plt.subplots()
ax.set(xscale='log',title='Learning Rate vs Accuracy', xlabel='Learning Rate', ylabel='Accuracy')
sns.lineplot(list(dict.keys()),list(dict.values()), label = "learning rate vs accuracy, @ 50,000 iterations", ax = ax)
ax.legend(loc='lower center', bbox_to_anchor=(.5, -.3), ncol=1)
```




    <matplotlib.legend.Legend at 0x7fe9232cc710>




![png](output_33_1.png)


#### No. of Iterations vs Accuracy


```python
dict = {}
for i in range(10000,100001, 10000):
    classifier = LogisticRegression(X,y)
    f = classifier.train(i, 1e-2, results = False)
    y_predict = classifier.predict(X)
    Accuracy = 100*np.sum(y == y_predict)/len(y)
    dict[i] = Accuracy
f, ax = plt.subplots()
ax.set(title='No. of iterations vs Accuracy', xlabel='No. of iterations', ylabel='Accuracy')
sns.lineplot(list(dict.keys()),list(dict.values()), label = "No. of iterations, @ 1e-2 learning rate", ax = ax)
ax.legend(loc='lower center', bbox_to_anchor=(.5, -.3), ncol=1)
```




    <matplotlib.legend.Legend at 0x7fe9336fff90>




![png](output_35_1.png)


### Testing the Model on the Best Parameters we Found
We will split our data into training and test data using 80/20 split.


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

classifier = LogisticRegression(X_train,y_train)
f = classifier.train(20000, 0.01, draw_history = False)
```

    @ no of iterations = 20000 and learning rate = 0.01 
     final W = [[-2.95402125]
     [ 1.29535646]
     [ 6.06386095]] 
     final cost = [0.36986368]



![png](output_37_1.png)


### Classifying our test data


```python
y_predict = classifier.predict(X_test)
Accuracy = 100*np.sum(y_test == y_predict)/len(y_test)
print(f"Accuracy of predicted labels = {Accuracy}%")
```

    Accuracy of predicted labels = 88.0%



```python
confusion(y_test, y_predict)
```


![png](output_40_0.png)


### Effect of the Order of the Samples on the Final Accuracy
We will use the parameter `random_state` in scikit's function `train_test_split` to generate randome training and test samples from our data, we will test all of them with the best parameters from above to see if they are going to affect the accuacy


```python
dict = {}
for i in range(1,51, 5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    classifier = LogisticRegression(X_train,y_train)
    f = classifier.train(20000, 0.01, results = False)
    y_predict = classifier.predict(X_test)
    Accuracy = 100*np.sum(y_test == y_predict)/len(y_test)
    dict[i] = Accuracy
f, ax = plt.subplots()
ax.set(title='Order of Samples vs. Accuracy', xlabel='Random State', ylabel='Accuracy')
sns.lineplot(list(dict.keys()),list(dict.values()), label = "Accuracy @ 20,000 iterations, eta = 0.01", ax = ax)
ax.legend(loc='lower center', bbox_to_anchor=(.5, -.3), ncol=1)
```




    <matplotlib.legend.Legend at 0x7fe9337d4550>




![png](output_42_1.png)


---
$\pagebreak$

## Second Experiment: The Full Iris Dataset
We will start again with the [Iris](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data) dataset, this time using the full feature set (4 features) but restricting it to two classes. can be found in the `iris.data` file attached: 

### Information about the dataset, quoted from the source
"This is perhaps the best known database to be found in the pattern
       recognition literature.  Fisher's paper is a classic in the field
       and is referenced frequently to this day.  (See Duda & Hart, for
       example.)  The data set contains 3 classes of 50 instances each,
       where each class refers to a type of iris plant.  One class is
       linearly separable from the other 2; the latter are NOT linearly
       separable from each other."

**We are going to use two classes and according to the description they will be linearly separable:**
- 0 = 'Setosa'
- 1 = 'Versicolor'

### Importing the data


```python
feature_dict = {i:label for i,label in zip(
                range(4),
                  ('sepal length in cm',
                  'sepal width in cm',
                  'petal length in cm',
                  'petal width in cm', ))}

df = pd.read_csv('iris.data')
df.columns = [l for i,l in sorted(feature_dict.items())] + ['class label']
df.dropna(how="all", inplace=True)

df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length in cm</th>
      <th>sepal width in cm</th>
      <th>petal length in cm</th>
      <th>petal width in cm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>149.000000</td>
      <td>149.000000</td>
      <td>149.000000</td>
      <td>149.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.848322</td>
      <td>3.051007</td>
      <td>3.774497</td>
      <td>1.205369</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.828594</td>
      <td>0.433499</td>
      <td>1.759651</td>
      <td>0.761292</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.300000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.100000</td>
      <td>2.800000</td>
      <td>1.600000</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.800000</td>
      <td>3.000000</td>
      <td>4.400000</td>
      <td>1.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.400000</td>
      <td>3.300000</td>
      <td>5.100000</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.900000</td>
      <td>4.400000</td>
      <td>6.900000</td>
      <td>2.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pairplot(data = df, hue = "class label")
```




    <seaborn.axisgrid.PairGrid at 0x7fe922ad8550>




![png](output_46_1.png)


### Taking only Iris-setosa and Iris-versicolor instances
These are the two classes with the most separation.

#### Training the model
We will split our data into training and test data using 80/20 split.


```python
X = df[['sepal length in cm',
        'sepal width in cm',
        'petal length in cm',
        'petal width in cm']]
y = df["class label"]


enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y)

X = np.asarray(X)
y = np.asarray(y)
filter = (y==0)|(y==1)
X = X[filter,:]
y = y[filter]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

classifier = LogisticRegression(X_train,y_train)
f = classifier.train(1000, 0.01, two_d = False)
```

    @ no of iterations = 1000 and learning rate = 0.01 
     final W = [[-0.20402627]
     [-0.29053776]
     [-1.04934456]
     [ 1.59764918]
     [ 0.68252448]] 
     final cost = [0.06774462]



![png](output_49_1.png)


#### Classifying our test data


```python
y_predict = classifier.predict(X_test)
Accuracy = 100*np.sum(y_test == y_predict)/len(y_test)
print(f"Accuracy of predicted labels = {Accuracy}%")
```

    Accuracy of predicted labels = 100.0%



```python
confusion(y_test, y_predict)
```


![png](output_52_0.png)


### Taking only Iris-versicolor and Iris-verginica instances
These are the two classes with the least separation.

#### Importing the data


```python
X = df[['sepal length in cm',
        'sepal width in cm',
        'petal length in cm',
        'petal width in cm']]
y = df["class label"]


X = np.asarray(X)
y = np.asarray(y)
filter = (y=="Iris-versicolor")|(y=="Iris-virginica")
X = X[filter,:]
y = y[filter]
enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y)
```

#### Experimenting With Parametes
##### Learning Rate vs. Accuracy


```python
dict = {}
learning_rates = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
for i in learning_rates:
    classifier = LogisticRegression(X,y)
    f = classifier.train(10000, i, results = False)
    y_predict = classifier.predict(X)
    Accuracy = 100*np.sum(y == y_predict)/len(y)
    dict[i] = Accuracy
f, ax = plt.subplots()
ax.set(xscale='log',title='Learning Rate vs Accuracy', xlabel='Learning Rate', ylabel='Accuracy')
sns.lineplot(list(dict.keys()),list(dict.values()), label = "learning rate vs accuracy, @ 10,000 iterations", ax = ax)
ax.legend(loc='lower center', bbox_to_anchor=(.5, -.3), ncol=1)
```




    <matplotlib.legend.Legend at 0x7fe932c941d0>




![png](output_56_1.png)


##### No. of Iterations vs Accuracy


```python
dict = {}
for i in range(1000,25001, 1000):
    classifier = LogisticRegression(X,y)
    f = classifier.train(i, 1e-3, results = False)
    y_predict = classifier.predict(X)
    Accuracy = 100*np.sum(y == y_predict)/len(y)
    dict[i] = Accuracy
f, ax = plt.subplots()
ax.set(title='No. of iterations vs Accuracy', xlabel='No. of iterations', ylabel='Accuracy')
sns.lineplot(list(dict.keys()),list(dict.values()), label = "No. of iterations, @ 1e-3 learning rate", ax = ax)
ax.legend(loc='lower center', bbox_to_anchor=(.5, -.3), ncol=1)
```




    <matplotlib.legend.Legend at 0x7fe932a3edd0>




![png](output_58_1.png)


#### Testing the Model on the Best Parameters we Found


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

classifier = LogisticRegression(X_train,y_train)
f = classifier.train(10000, 1e-2, two_d = False)
```

    @ no of iterations = 10000 and learning rate = 0.01 
     final W = [[-1.46646965]
     [-2.40292358]
     [-2.16742733]
     [ 3.43459288]
     [ 3.5655377 ]] 
     final cost = [0.16311965]



![png](output_60_1.png)


#### Classifying our test data


```python
y_predict = classifier.predict(X_test)
Accuracy = 100*np.sum(y_test == y_predict)/len(y_test)
print(f"Accuracy of predicted labels = {Accuracy}%")
```

    Accuracy of predicted labels = 95.0%



```python
confusion(y_test, y_predict)
```


![png](output_63_0.png)


#### Effect of the Order of the Samples on the Final Accuracy
We will use the parameter `random_state` in scikit's function `train_test_split` to generate randome training and test samples from our data, we will test all of them with the best parameters from above to see if they are going to affect the accuacy


```python
dict = {}
for i in range(1,51, 5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    classifier = LogisticRegression(X_train,y_train)
    f = classifier.train(10000, 1e-2, results = False)
    y_predict = classifier.predict(X_test)
    Accuracy = 100*np.sum(y_test == y_predict)/len(y_test)
    dict[i] = Accuracy
f, ax = plt.subplots()
ax.set(title='Order of Samples vs. Accuracy', xlabel='Random State', ylabel='Accuracy')
sns.lineplot(list(dict.keys()),list(dict.values()), label = "Accuracy @ 10,000 iterations, eta = 0.01", ax = ax)
ax.legend(loc='lower center', bbox_to_anchor=(.5, -.3), ncol=1)
```




    <matplotlib.legend.Legend at 0x7fe933706f50>




![png](output_65_1.png)


---
$\pagebreak$

## Third Experiment: Banknote Authentication Dataset
We will use the [banknote authentication Data Set](https://archive.ics.uci.edu/ml/datasets/banknote+authentication) which has 4 features, two classes and 1372 instances. It can be found in the `data_banknote_authentication.csv` file attached.

### Information about the dataset, quoted from the source
"Data were extracted from images that were taken from genuine and forged banknote-like specimens. For digitization, an industrial camera usually used for print inspection was used. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tool were used to extract features from images."

### Importing the data


```python
df = pd.read_csv('data_banknote_authentication.csv')
feature_dict = {i:label for i,label in zip(
                range(4),
                  ('variance of Wavelet Transformed image ',
                    'skewness of Wavelet Transformed image',
                    'curtosis of Wavelet Transformed image',
                    'entropy of image', ))}

df.columns = [l for i,l in sorted(feature_dict.items())] + ['class label']
df.dropna(how="all", inplace=True)

class_in_strings = lambda x: 'Class ' + str(x)
df['class label']= df["class label"].apply(class_in_strings)

df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variance of Wavelet Transformed image</th>
      <th>skewness of Wavelet Transformed image</th>
      <th>curtosis of Wavelet Transformed image</th>
      <th>entropy of image</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1371.000000</td>
      <td>1371.000000</td>
      <td>1371.000000</td>
      <td>1371.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.431410</td>
      <td>1.917434</td>
      <td>1.400694</td>
      <td>-1.192200</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.842494</td>
      <td>5.868359</td>
      <td>4.310105</td>
      <td>2.101683</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-7.042100</td>
      <td>-13.773100</td>
      <td>-5.286100</td>
      <td>-8.548200</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-1.774700</td>
      <td>-1.711300</td>
      <td>-1.553350</td>
      <td>-2.417000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.495710</td>
      <td>2.313400</td>
      <td>0.616630</td>
      <td>-0.586650</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.814650</td>
      <td>6.813100</td>
      <td>3.181600</td>
      <td>0.394810</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.824800</td>
      <td>12.951600</td>
      <td>17.927400</td>
      <td>2.449500</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pairplot(data = df, hue = "class label")
```




    <seaborn.axisgrid.PairGrid at 0x7fe93294a6d0>




![png](output_69_1.png)



```python
X = df[['variance of Wavelet Transformed image ',
        'skewness of Wavelet Transformed image',
        'curtosis of Wavelet Transformed image',
        'entropy of image']]
y = df["class label"]

X = np.asarray(X)
y = np.asarray(y)

enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y)
```

#### Experimenting With Parametes
##### Learning Rate vs. Accuracy


```python
dict = {}
learning_rates = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
for i in learning_rates:
    classifier = LogisticRegression(X,y)
    f = classifier.train(2000, i, results = False)
    y_predict = classifier.predict(X)
    Accuracy = 100*np.sum(y == y_predict)/len(y)
    dict[i] = Accuracy
f, ax = plt.subplots()
ax.set(xscale='log',title='Learning Rate vs Accuracy', xlabel='Learning Rate', ylabel='Accuracy')
sns.lineplot(list(dict.keys()),list(dict.values()), label = "learning rate vs accuracy, @ 2,000 iterations", ax = ax)
ax.legend(loc='lower center', bbox_to_anchor=(.5, -.3), ncol=1)
```




    <matplotlib.legend.Legend at 0x7fe922c96150>




![png](output_72_1.png)


##### No. of Iterations vs Accuracy


```python
dict = {}
for i in range(100,3001, 100):
    classifier = LogisticRegression(X,y)
    f = classifier.train(i, 1e-3, results = False)
    y_predict = classifier.predict(X)
    Accuracy = 100*np.sum(y == y_predict)/len(y)
    dict[i] = Accuracy
f, ax = plt.subplots()
ax.set(title='No. of iterations vs Accuracy', xlabel='No. of iterations', ylabel='Accuracy')
sns.lineplot(list(dict.keys()),list(dict.values()), label = "No. of iterations, @ 1e-3 learning rate", ax = ax)
ax.legend(loc='lower center', bbox_to_anchor=(.5, -.3), ncol=1)
```




    <matplotlib.legend.Legend at 0x7fe932c7c990>




![png](output_74_1.png)


### Training the model
We will split our data into training and test data using 80/20 split.


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

classifier = LogisticRegression(X_train,y_train)
f = classifier.train(3000, 0.1, two_d = False)
```

    @ no of iterations = 3000 and learning rate = 0.1 
     final W = [[ 3.0352988 ]
     [-2.82396881]
     [-1.63429646]
     [-1.95560593]
     [-0.18816712]] 
     final cost = [0.02675094]



![png](output_76_1.png)


### Classifying our test data


```python
y_predict = classifier.predict(X_test)
Accuracy = 100*np.sum(y_test == y_predict)/len(y_test)
print(f"Accuracy of predicted labels = {Accuracy}%")
```

    Accuracy of predicted labels = 99.27272727272727%



```python
confusion(y_test, y_predict)
```


![png](output_79_0.png)


### Effect of the Order of the Samples on the Final Accuracy
We will use the parameter `random_state` in scikit's function `train_test_split` to generate randome training and test samples from our data, we will test all of them with the best parameters from above to see if they are going to affect the accuacy


```python
dict = {}
for i in range(1,51, 5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    classifier = LogisticRegression(X_train,y_train)
    f = classifier.train(3000, 0.1, results = False)
    y_predict = classifier.predict(X_test)
    Accuracy = 100*np.sum(y_test == y_predict)/len(y_test)
    dict[i] = Accuracy
f, ax = plt.subplots()
ax.set(title='Order of Samples vs. Accuracy', xlabel='Random State', ylabel='Accuracy')
sns.lineplot(list(dict.keys()),list(dict.values()), label = "Accuracy @ 3,000 iterations, eta = 0.1", ax = ax)
ax.legend(loc='lower center', bbox_to_anchor=(.5, -.3), ncol=1)
```




    <matplotlib.legend.Legend at 0x7fe922a1e410>




![png](output_81_1.png)

