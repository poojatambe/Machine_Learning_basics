from sklearn import datasets
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier


data=datasets.load_iris()
# print(data.data)
# print(data.target)

# Grid search cv
def grid_search(X, Y, params, estimator, cv):
    gs= GridSearchCV(estimator, param_grid=params, cv=cv)
    gs = gs.fit(X, Y)
    print("Grid Search score:", gs.best_score_)
    print("Grid Search best parameters:", gs.best_params_)


# Ramdomized search cv
def random_search(X, Y, params, estimator, cv, random_state):
    rs= RandomizedSearchCV(estimator, params, cv=cv, random_state= random_state)
    rs = rs.fit(X, Y)
    print("Random search score:", rs.best_score_)
    print("Random search best parameters:", rs.best_params_)


params={ 'n_neighbors' : [3, 5, 7, 9, 11],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}

grid_search(data.data, data.target, params, KNeighborsClassifier(), 3)
random_search(data.data, data.target, params,KNeighborsClassifier(), 3, 42)

