MLCompare
=========

A python script to compare different suppervised machine learning algorithms on a given dataset. At the moment only support classification algorithms.

Note that this is meant to give you a quick insight on how some algorithms perform on a data set. For a faster execution time, set the number of threads in the .py file.

Example
-------

> #######Input:
> data = datasets.load_iris()

> MLCompare(data.data, data.target)

The output will be the results of the algorithms. As an example:

> RESULTS:

>
> Logistic Classifier:      Test: 0.79     CV: 0.73     Train Time: 0.001s    Score Time: 0.001s     Parameters: {'penalty': 'l2', 'C': 1.0, 'fit_intercept': True}

>
>
>
> MultinomialNB:      Test: 0.96     CV: 0.87     Train Time: 0.002s     Score Time: 0.000s     Parameters: {'alpha': 0.0001, 'fit_prior': False}

>
>
>
> SVM:      Test: 0.90     CV: 1.00     Train Time: 0.001s     Score Time: 0.000s     Parameters: {'kernel': 'poly', 'C': 0.0001, 'gamma': 10.0, 'degree': 2}

>
>
>
> GaussianNB:      Test: 0.94     CV: 1.00     Train Time: 0.001s     Score Time: 0.001s     Parameters: {}

>
>
>
> kNN:      Test: 0.93     CV: 0.73     Train Time: 0.001s     Score Time: 0.002s     Parameters: {'n_neighbors': 10, 'weights': 'distance', 'algorithm': 'auto'}

>
>
>
> Decision Tree:      Test: 0.96     CV: 0.93     Train Time: 0.001s     Score Time: 0.000s     Parameters: {'max_features': None, 'min_samples_split': 4, 'criterion': 'gini', 'max_depth': None, 'min_samples_leaf': 1}

>
>
>
> AdaBoost:      Test: 0.96     CV: 0.93     Train Time: 0.001s     Score Time: 0.001s     Parameters: {'n_estimators': 2, 'base_estimator': DecisionTreeClassifier(compute_importances=None, criterion='gini',>
            max_depth=None, max_features=None, min_density=None,>
            min_samples_leaf=1, min_samples_split=2, random_state=None,>
            splitter='best'), 'learning_rate': 0.0001, 'algorithm': 'SAMME.R'}

>
>
>
> Random Forest:      Test: 0.96     CV: 0.80     Train Time: 0.150s     Score Time: 0.120s     Parameters: {'n_jobs': 4, 'bootstrap': True, 'min_samples_leaf': 2, 'n_estimators': 10, 'min_samples_split': 2, 'criterion': 'gini', 'max_features': None, 'max_depth': 2}

>
>
>
> Evaluation Ended.
