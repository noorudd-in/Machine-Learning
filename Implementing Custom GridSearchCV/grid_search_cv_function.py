#This function will returns 60% randomly sample data ranging from 0 to length of training data without duplicates.
def select_data_without_duplicates(x_train):
    return random.sample(range(0, len(x_train)), int(0.6*len(x_train)))

def GridSearchCV(x_train, y_train, classifier, parameter, folds):
    trainscores = []
    testscores = []
    for k in tqdm(parameter['n_neighbors']):
        training_fold_scores = []
        cv_fold_scores = []
        for j in range(0, cv_folds):
            #Spliting Data into train and test
            training_data = select_data_without_duplicates(x_train) #60% of the x_train data
            cv_data = list(set(list(range(1, len(x_train)))) - set(training_data)) #Remaining data: 100% - 60%
            #Evaluating X_train, Y_train, X_test, Y_test data from new split
            X_train = x_train[training_data]
            X_cv = x_train[cv_data]
            Y_train = y_train[training_data]
            Y_cv = y_train[cv_data]
            #Applying KNN Algorithm and fitting model
            classifier.n_neighbors = k
            classifier.fit(X_train, Y_train)
            #Predicting accuracy for X_test data and appending value to testing_fold_scores
            Y_cv_predict = classifier.predict(X_cv)
            cv_fold_scores.append(accuracy_score(Y_cv, Y_cv_predict))
            #Predicting accuracy for X_test data and appending value to training_fold_scores
            Y_train_predict = classifier.predict(X_train)
            training_fold_scores.append(accuracy_score(Y_train, Y_train_predict))
            
        #For each fold i.e folds=0,1,2 we have predicted accuracy for x_train and x_test. Since we have three values,
        #We will take means of three accuracies and appending them to trainscores and testscores.
        trainscores.append(np.mean(np.array(training_fold_scores)))
        testscores.append(np.mean(np.array(cv_fold_scores)))
    return trainscores,testscores
