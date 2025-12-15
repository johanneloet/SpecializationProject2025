import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn import svm
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import os
import joblib

" -- This file is from Roya, I have just change from SVR to SVC --"

random_state = 343

def plot_confusion_matrix(y_test, y_test_fit, CV_suffix = "", class_names=[]):
    save_path="./plots_use/SVM_tests"
    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    cm = confusion_matrix(y_test, y_test_fit, labels=class_names, normalize='true')

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation='vertical',cmap=plt.cm.Blues, values_format=".2f")
    for text in disp.ax_.texts:
        text.set_fontsize(8)
    plt.tight_layout()
    #plt.show()

    save_file = os.path.join(save_path, f"{CV_suffix}_confusion_matrix.png")
    plt.savefig(save_file, dpi=300)
    plt.close()


def run_SVC(X_train, y_train, X_test, y_test, class_names=[], CV_suffix = "", opt = None, time_window = None):

    """  SVC   """
    save_base_path="./models"
    # Create a directory for the model
    os.makedirs(save_base_path, exist_ok=True)

    from sklearn.utils import shuffle

    #Shuffle training samples before anything else
    X_train, y_train = shuffle(X_train, y_train, random_state=random_state)

    if opt == True:
        print('SVC hyper parameter tuning')

        # Define the model (without specifying kernel yet)
        model = svm.SVC()

        # Define hyperparameters to search, including different kernels
        param_grid = {
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Kernel coefficient
            'kernel': ['linear', 'rbf'],
            'C': [0.01, 0.1, 1, 5],
            #'epsilon': [0.2, 0.5, 1.0], # this is for regression only
            # 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Try different kernels
            # 'C': [0.1, 1, 10, 100],  # Regularization parameter
            # 'epsilon': [0.1, 0.2, 0.5],  # Epsilon in the epsilon-SVR
        }
        
        # Perform Grid Search with Cross-Validation
        n_splits = 3

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=StratifiedKFold(n_splits), n_jobs=-1)
        
        # Fit the model with the best parameters found
        grid_search.fit(X_train, y_train)  #, groups=train_groups
        
        # Get the best parameters and model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        # Print the best parameters
        # print(f"Best hyperparameters: {best_params}")

        #  View cross-validation fold results
        cv_results = grid_search.cv_results_
        # print("Fold-wise test results:")
        for i in range(3):  # Assuming 3-fold cross-validation
            fold_score = cv_results[f'split{i}_test_score'][grid_search.best_index_]
            print(f"Fold {i+1} test score: {fold_score}")

        # Evaluate the model on test data
        Overall = best_model.score(X_test, y_test)
        # print(f"Test R^2 score: {Overall}")

        model = best_model
        # Fit the model using the training data
        # model.fit(X_train, y_train)

        # Save the model to a file
        model_filename = os.path.join(save_base_path, f"SVC_{CV_suffix}_{time_window}.joblib")
        joblib.dump(model, model_filename)
        


    y_test_fit      = model.predict(X_test)
    accuracy_test   = model.score(X_test, y_test)
    f1_test         = f1_score(y_test, y_test_fit, average='macro')
    precision_test  = precision_score(y_test, y_test_fit, average='macro')
    recall_test     = recall_score(y_test, y_test_fit, average='macro')

    y_train_fit     = model.predict(X_train)
    accuracy_train  = model.score(X_train, y_train)
    f1_train         = f1_score(y_train, y_train_fit, average='macro')
    precision_train  = precision_score(y_train, y_train_fit, average='macro')
    recall_train     = recall_score(y_train, y_train_fit, average='macro')
        
    test_results    = (y_test_fit, accuracy_test, f1_test, precision_test, recall_test)
    train_results   = (y_train_fit, accuracy_train, f1_train, precision_train, recall_train)
        
    # Call the plot function with the necessary parameters
    Title = f"SVC_{CV_suffix}_{time_window}"
    #plot_confusion_matrix(y_test, y_test_fit, CV_suffix=Title, class_names=class_names)

    return test_results, train_results


def run_SVC_with_feature_tuning(train_df_per_space, test_df_per_space, class_names=[], CV_suffix = "", opt = None, time_window = None, label_mapping=None,
                                feature_spaces =  ["baseline",
                                                "expanded+baseline",
                                                # "expanded_only",
                                                # "time_only",
                                                # "time_only+exp_FSR",
                                                # "freq_only",
                                                # "freq_only+exp_FSR",
                                                "FSR_only",
                                                "arm_only+FSR",
                                                "back_only+FSR",
                                                "IMU_only"]):
    
    save_base_path="./models"
    # Create a directory for the model
    os.makedirs(save_base_path, exist_ok=True)
    space_and_performance_summaries = {}
    for space in feature_spaces:
        train_df = train_df_per_space[space]
        test_df = test_df_per_space[space]
        
        # map labels..
        if label_mapping is not None:
            print('mapping labels')
            y_train = train_df["label"].map(label_mapping)
            y_test = test_df["label"].map(label_mapping)
            labels = ["hands_up", "push_pull", "squat_lift", "stand_sit", "walking"]
        else:    
            labels = ["hand_up_back", "hands_forward", "hands_up", "push", "pull", "squatting", "lifting", "sitting", "standing", "walking"]
            y_train = train_df["label"]
            y_test = test_df["label"]
        # Separate features and labels
        X_train = train_df.drop(columns=["label"])
        #groups = train_df["test_id"]
        #y_train = y_train["label"]

        X_test = test_df.drop(columns=["label"]) # test_id column is not added bact to test df, therefore only drop label
        #y_test = y_test["label"]

            
        # Scale data (SD of 1 and mean of 0)
        scaler = StandardScaler().set_output(transform="pandas")
        # Fit scaler on training data
        scaler.fit(X_train)
        # Transform both train and test set with the scaler
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print(f"Scaling complete. Feature space: {space}")
        
        # Apply pca
        pca = PCA(n_components=0.95)
        pca_fit = pca.fit(X_train_scaled)
        pca_components = pca.n_components_
        print (f"pca components: {pca_components}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum()}")

        X_train_pca = pca_fit.transform(X_train_scaled)
        X_test_pca = pca_fit.transform(X_test_scaled)
        
        # le = LabelEncoder()
        # y_train_encoded = le.fit_transform(y_train)
        # y_test_encoded = le.transform(y_test)
        
        # below is originally implemented SVC function - credits Maria and Roya
        
        """  SVC   """
        from sklearn.utils import shuffle
        #Shuffle training samples before anything else
        X_train_pca, y_train = shuffle(X_train_pca, y_train, random_state=random_state)

        if opt == True:
            print('SVC hyper parameter tuning')
            # Define the model (without specifying kernel yet)
            model = svm.SVC()

            # Define hyperparameters to search, including different kernels
            param_grid = {
                'gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Kernel coefficient
                'kernel': ['linear', 'rbf'],
                'C': [0.01, 0.1, 1, 5], # keep C relatively small to avoid overfitting
                }
            
            # Perform Grid Search with Cross-Validation
            n_splits = 3
            print('Beginning gridsearch')
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=StratifiedKFold(n_splits), n_jobs=-1, scoring='f1_macro')
            
            print("Fitting gridsearch")
            # Fit the model with the best parameters found
            grid_search.fit(X_train_pca, y_train) 
            
            # Get the best parameters and model
            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_
            best_f1 = grid_search.best_score_
            
            y_test_fit      = best_model.predict(X_test_pca)
            accuracy_test   = best_model.score(X_test_pca, y_test)
            f1_test         = f1_score(y_test, y_test_fit, average='macro')
            precision_test  = precision_score(y_test, y_test_fit, average='macro')
            recall_test     = recall_score(y_test, y_test_fit, average='macro')

            y_train_fit     = best_model.predict(X_train_pca)
            accuracy_train  = best_model.score(X_train_pca, y_train)
            f1_train         = f1_score(y_train, y_train_fit, average='macro')
            precision_train  = precision_score(y_train, y_train_fit, average='macro')
            recall_train     = recall_score(y_train, y_train_fit, average='macro')
                
            test_results    = {
                'y_test_fit' : y_test_fit, 
                'accuracy' : accuracy_test, 
                'f1' :  f1_test, 
                'precision' : precision_test,
                'recall' : recall_test,
                'hyperparameters' : best_params}
            
            train_results   = (y_train_fit, accuracy_train, f1_train, precision_train, recall_train)
            
            space_and_performance_summaries[space] = test_results

            #Print the best parameters
            #print(f"Best hyperparameters: {best_params}")

            #  View cross-validation fold results
            cv_results = grid_search.cv_results_
            # print("Fold-wise test results:")
            for i in range(3):  # Assuming 3-fold cross-validation
                fold_score = cv_results[f'split{i}_test_score'][grid_search.best_index_]
                print(f"Fold {i+1} test score: {fold_score}")

            # Evaluate the model on test data
            #Overall = best_model.score(X_test_pca, y_test_encoded)
            # print(f"Test R^2 score: {Overall}")

            
            # Fit the model using the training data
            # model.fit(X_train, y_train)

            # Save the model to a file
            """
            model_filename = os.path.join(save_base_path, f"SVC_{CV_suffix}_{time_window}.joblib")
            joblib.dump(model, model_filename)
            
            if best_f1 > current_best_f1:
                current_best_f1 = best_f1
                best_space = space
                model_final = best_model
                X_train_best_space = X_train_pca
                X_test_best_space = X_test_pca
                y_train_best_space = y_train
                y_test_best_space = y_test
            
            space_and_f1_summaries[space] = {"f1" : best_f1, 
                                             "parameters" : best_params}
            
            
    y_test_fit      = model_final.predict(X_test_best_space)
    accuracy_test   = model_final.score(X_test_best_space, y_test_best_space)
    f1_test         = f1_score(y_test_best_space, y_test_fit, average='macro')
    precision_test  = precision_score(y_test_best_space, y_test_fit, average='macro')
    recall_test     = recall_score(y_test_best_space, y_test_fit, average='macro')

    y_train_fit     = model_final.predict(X_train_best_space)
    accuracy_train  = model_final.score(X_train_best_space, y_train_best_space)
    f1_train         = f1_score(y_train_best_space, y_train_fit, average='macro')
    precision_train  = precision_score(y_train_best_space, y_train_fit, average='macro')
    recall_train     = recall_score(y_train_best_space, y_train_fit, average='macro')
        
    test_results    = (y_test_fit, accuracy_test, f1_test, precision_test, recall_test)
    train_results   = (y_train_fit, accuracy_train, f1_train, precision_train, recall_train)
    """
    
    # y_test_fit_labels = le.inverse_transform(y_test_fit)
    # y_test_labels = le.inverse_transform(y_test_best_space)

    # y_train_fit_labels = le.inverse_transform(y_train_fit)
    # y_train_labels = le.inverse_transform(y_train_best_space)
    
        
    # Call the plot function with the necessary parameters
    Title = f"SVC_{CV_suffix}_{time_window}"
    #plot_confusion_matrix(y_test, y_test_fit, CV_suffix=Title, class_names=class_names)
    
    return space_and_performance_summaries, y_test # assume y_test is the same for every space. It should be as it is only labels
        
        
        
        
    


