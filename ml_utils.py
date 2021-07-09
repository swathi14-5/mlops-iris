from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# define a Gaussain NB classifier
clf = GaussianNB()
clf1= DecisionTreeClassifier(random_state=0,max_depth=5)

# define the class encodings and reverse encodings
classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def load_model():
    # load the dataset from the official sklearn datasets
    X, y = datasets.load_iris(return_X_y=True)

    # do the test-train split and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    clf1.fit(X_train, y_train)

    # calculate the print the accuracy score
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Model trained with accuracy acc: {round(acc, 3)}")
    
    # calculate the print the accuracy score

    acc1 = accuracy_score(y_test, clf1.predict(X_test))
    print(f"Model trained with accuracy acc1: {round(acc1, 3)}")
    #best accuracy
    if acc>acc1:
        return("acc is best",acc)
    else:
        return("acc1 is best",acc1)


# function to predict the flower using the model
def predict(query_data):
    x = list(query_data.dict().values())
    prediction = clf.predict([x])[0]
    prediction1 = clf.predict([x])[0]
    print(f"Model prediction: {classes[prediction]}")
    print(f"Model prediction: {classes[prediction1]}")

    #predicitng the best model
    if prediction>prediction1:
        return(classes[prediction])
    else:
        return(classes[prediction1])

# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.flower_class] for d in data]

    # fit the classifier again based on the new data obtained
    clf.fit(X, y)
    #to get the better accuracy trainig for the new data.
    clf1.fit(X, y)
