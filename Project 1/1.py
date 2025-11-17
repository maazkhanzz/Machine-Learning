from sklearn import datasets 
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV


### PART 1 ###
# Scikit-Learn provides many popular datasets. The breast cancer wisconsin dataset is one of them. 
# Writing code that fetches the breast cancer wisconsin dataset. 
# https://scikit-learn.org/stable/datasets/toy_dataset.html

X, y = datasets.load_breast_cancer(return_X_y=True) 

print("There are", X.shape[0], "instances described by", X.shape[1], "features.")

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.4, stratify = y , random_state = 42) 
 
clf = tree.DecisionTreeClassifier(criterion = 'entropy' , min_samples_split = 6, random_state = 42)
clf.fit(X_train, y_train)

predC = clf.predict(X_test)

print('The accuracy of the classifier is', accuracy_score(y_test, predC))

_ = tree.plot_tree(clf, filled=True, fontsize=12)
plt.show()

### PART 2.1 ###
# Visualize the training and test accuracies as a function of the maximum depth of the decision tree

trainAccuracy = []  
testAccuracy = [] 
depthOptions = range(1, 16)  
for depth in depthOptions: 

    cltree = tree.DecisionTreeClassifier(criterion = 'entropy', min_samples_split = 6, max_depth = depth, random_state = 42)
    cltree = cltree.fit(X_train, y_train)
    y_predTrain = cltree.predict(X_train)
    y_predTest = cltree.predict(X_test)
    trainAccuracy.append(accuracy_score(y_train, y_predTrain))
    testAccuracy.append(accuracy_score(y_test,y_predTest))

plt.plot(depthOptions, trainAccuracy, 'o-', depthOptions, testAccuracy, 's-')
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Tree Depth')
plt.ylabel('Classifier Accuracy')


### PART 2.2 ###
# Use sklearn's GridSearchCV function to perform an exhaustive search to find the best tree depth and the minimum number of samples to split a node
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

parameters = {'max_depth': range(1, 16), 'min_samples_split': [2, 4, 6, 8, 10, 12, 14]}
clf = GridSearchCV(estimator=tree.DecisionTreeClassifier(criterion='entropy', random_state=42), param_grid=parameters, cv=10, scoring='accuracy',n_jobs=-1) 
clf.fit(X_train, y_train) #(4 points)
tree_model = clf.best_estimator_ #(4 points)

print("The maximum depth of the tree sis", clf.best_params_['max_depth'], 
      'and the minimum number of samples required to split a node is', clf.best_params_['min_samples_split'])

_ = tree.plot_tree(tree_model, filled=True, fontsize = 12)




