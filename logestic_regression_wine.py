import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold

import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

# for ordinal regression
class WineQualityClassifier1:
    def skmultinomial(self):

                # load wine data and create binary classification column
        wine = pd.read_csv("winequality_dataset/winequality-red.csv", sep=";")
        # Create a new DataFrame not containing the five random data from wine_data
        # Separate five data points as raw data
        five_data = wine.sample(5).copy()
        five_data.reset_index(drop=True, inplace=True)


        wine_data = wine[~wine.index.isin(five_data.index)].copy()
        wine_data.reset_index(drop=True, inplace=True)

        bins = [0, 6, 7, 10]
        labels = ["low", "medium", "high"]
        wine_data["quality_cat"] = pd.cut(wine_data["quality"], bins=bins, labels=labels)
        five_data["quality_cat"] = pd.cut(five_data["quality"], bins=bins, labels=labels)
        X = wine_data.drop(["quality", "quality_cat"], axis=1)
        target_ordinal = wine_data["quality_cat"]




        raw_input = five_data.drop(["quality", "quality_cat"], axis=1)
        five_data['quality_cat'] = pd.Categorical(five_data['quality_cat'])
        raw_output = five_data['quality_cat']

        # standardize the Data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        raw_input_scaled = scaler.fit_transform(raw_input)
        # split the data
        ordinal_train, ordinal_test, targetordinal_train, targetordinal_test = train_test_split(X_scaled, target_ordinal,
                                                                                                    test_size=0.5,
                                                                                                    random_state=100)

        # fit to the model

        ordinal_probit = OrderedModel(endog=targetordinal_train, exog=ordinal_train, link='probit', ordered=True)
        ordinal_probit = ordinal_probit.fit(method='bfgs', maxiter=1000)

        ordinal_logit = OrderedModel(endog=targetordinal_train, exog=ordinal_train, link='logit', ordered=True)
        ordinal_logit = ordinal_logit.fit(method='bfgs', maxiter=1000)


        # predict the model
        predicted_logit = ordinal_probit.predict(exog=ordinal_test)
        pred_choice_logit = predicted_logit.argmax(axis=1)

        predicted_probit = ordinal_logit.predict(exog=ordinal_test)
        pred_choice_probit = predicted_probit.argmax(axis=1)


        # True labels
        true_labels = np.asarray(targetordinal_test.cat.codes)

        # Model Evaluation: Calculating accuracy and other classification metrics
        test_accuracy_logit = accuracy_score(true_labels, pred_choice_logit)
        test_accuracy_probit = accuracy_score(true_labels, pred_choice_probit)
        print("-" * 50)
        print("Test Accuracy of probit ordinal regression: {:.2f}%".format(test_accuracy_logit * 100))
        print("Test Accuracy of logit ordinal regression: {:.2f}%".format(test_accuracy_probit * 100))

        # five data test
        predicted_logit1 = ordinal_logit.predict(exog=raw_input_scaled)
        pred_choice_logit1 = predicted_logit1.argmax(axis=1)

        predicted_probit1 = ordinal_probit.predict(exog=raw_input_scaled)
        pred_choice_probit1 =  predicted_probit1.argmax(axis=1)


        # Print the converted labels
        pred_labels_probit = [labels[label] for label in pred_choice_logit1]
        pred_labels_logit = [labels[label] for label in pred_choice_probit1]


        # Create a DataFrame to store the actual and predicted labels
        df = pd.DataFrame({'Actual Labels': raw_output, 'Predicted Labels':pred_labels_probit})

        # Print the DataFrame
        print("-" * 50)
        print(df)
        print("-" * 50)
        test_accuracy_logit1 = accuracy_score(pred_labels_logit, raw_output)
        test_accuracy_probit1 = accuracy_score(pred_labels_probit, raw_output)
        print("Test Accuracy of probit ordinal regression for five raw data: {:.2f}%".format(test_accuracy_logit1 * 100))
        print("Test Accuracy of logit ordinal regression for five raw data: {:.2f}%".format(test_accuracy_probit1 * 100))

        print("-" * 50)
        print("Classification Report for  probit and logit model for five raw data:")
        print(classification_report(raw_output, pred_labels_probit, zero_division=1))





# for multinomial logestic regression using sklearn library
class WineQualityClassifier:
    def __init__(self):
        self.wine_data = None
        self.five_data = None
        self.X = None
        self.y = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.target_train = None
        self.target_test = None
        self.l1_model = None
        self.l2_model = None
        self.elastic_net_model = None
        self.k = None
        self.kf = None
        self.scores = None

    def ordinal_regression():
        print()  

    def load_data(self, file_path):
        wine = pd.read_csv(file_path, sep=";")
        self.five_data = wine.sample(5).copy()
        self.five_data.reset_index(drop=True, inplace=True)
        self.wine_data = wine[~wine.index.isin(self.five_data.index)].copy()
        self.wine_data.reset_index(drop=True, inplace=True)

    def preprocess_data(self):
        bins = [0, 4, 6, 7, 10]
        labels = ["D", "C", "B", "A"]
        self.wine_data["quality_cat"] = pd.cut(self.wine_data["quality"], bins=bins, labels=labels)
        self.five_data["quality_cat"] = pd.cut(self.five_data["quality"], bins=bins, labels=labels)
        self.X = self.wine_data.drop(["quality", "quality_cat"], axis=1)
        self.y = self.wine_data["quality_cat"]

        self.X_five= self.five_data.drop(["quality", "quality_cat"], axis=1)
        self.fiveY = self.five_data["quality_cat"]

        self.scaler = MinMaxScaler()
        total_sulfur_dioxide = self.X["total sulfur dioxide"].values.reshape(-1, 1)
        total_sulfur_dioxide_normalized = self.scaler.fit_transform(total_sulfur_dioxide)
        self.X["total sulfur dioxide"] = total_sulfur_dioxide_normalized

    def split_data(self):
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.X_five = self.scaler.transform(self.five_data.drop(["quality", "quality_cat"], axis=1))

        self.X_train, self.X_test, self.target_train, self.target_test = train_test_split(self.X_scaled, self.y, test_size=0.2, random_state=100)

    def train_models(self):
        self.l1_model = LogisticRegression(penalty="l1", multi_class="multinomial", solver="saga", max_iter=1000)
        self.l1_model.fit(self.X_train, self.target_train)

        self.l2_model = LogisticRegression(penalty="l2", multi_class="multinomial", solver="newton-cg", max_iter=1000)
        self.l2_model.fit(self.X_train, self.target_train)

        self.elastic_net_model = LogisticRegression(penalty="elasticnet", multi_class="multinomial", solver="saga", l1_ratio=0.2, max_iter=1000)
        self.elastic_net_model.fit(self.X_train, self.target_train)

    def perform_cross_validation(self):
        self.k = 5
        self.kf = KFold(n_splits=self.k, shuffle=True, random_state=42)
        self.scores = cross_val_score(self.l1_model, self.X_scaled, self.y, cv=self.kf)

        # make predictions on test set and evaluate accuracy
        l1_preds = self.l1_model.predict(self.X_test)
        l2_preds = self.l2_model.predict(self.X_test)
        elastic_net_preds = self.elastic_net_model.predict(self.X_test)

        l1_acc = accuracy_score(self.target_test , l1_preds)
        l2_acc = accuracy_score(self.target_test , l2_preds)
        elastic_net_acc = accuracy_score(self.target_test, elastic_net_preds)

        #  Calculate F1 score for L1 model
        l1_f1 = f1_score(self.target_test, l1_preds, average='macro')
        print("F1 score for L1 model: {:.2%}".format(l1_f1))

        # Calculate F1 score for L2 model
        l2_f1 = f1_score(self.target_test, l2_preds, average='macro')
        print("F1 score for L2 model: {:.2%}".format(l2_f1))

        # Calculate F1 score for Elastic-Net model
        elastic_net_f1 = f1_score(self.target_test, elastic_net_preds, average='macro')
        print("F1 score for Elastic-Net model: {:.2%}".format(elastic_net_f1))
        print("-" * 50)
        print("Accuracy for L1 model: {:.2f}%".format(l1_acc*100))
        print("Accuracy for L2 model: {:.2f}%".format(l2_acc*100))
        print("Accuracy for elastic-net model: {:.2f}%".format(elastic_net_acc*100))

        # Make predictions for five raw data
        predictions1 = self.l2_model.predict(self.X_five)
        predictions2 = self.l1_model.predict(self.X_five)
        predictions3 = self.elastic_net_model.predict(self.X_five)
        print("-" * 50)
        # Generate classification report for each model's predictions
        print("Classification Report for L1 model for five raw data:")
        print(classification_report(self.fiveY, predictions1, zero_division=1))
        print("-" * 50)
        print("Classification Report for L2 model for five raw data:")
        print(classification_report(self.fiveY, predictions2, zero_division=1))
        print("-" * 50)
        print("Classification Report for Elastic-Net model for five raw data:")
        print(classification_report(self.fiveY, predictions3, zero_division=1))

#  ordinal regression 's experiment pf  wine's quality without categorization 
class WineQualitywithoutCategorization:
    def statsOrdinal(self):

                # load wine data and create binary classification column
        wine = pd.read_csv("winequality_dataset/winequality-red.csv", sep=";")
        # Create a new DataFrame not containing the five random data from wine_data
        # Separate five data points as raw data
        five_data = wine.sample(5).copy()
        five_data.reset_index(drop=True, inplace=True)


        wine_data = wine[~wine.index.isin(five_data.index)].copy()
        wine_data.reset_index(drop=True, inplace=True)

       
        X = wine_data.drop(["quality"], axis=1)
        wine_data['quality']  = pd.Categorical(wine_data['quality'], categories=[3, 4, 5, 6, 7, 8], ordered=True)

        target_ordinal = wine_data["quality"]
    

        raw_input = five_data.drop(["quality"], axis=1)
        # data transformation for five raw data 
        five_data['quality']  = pd.Categorical(five_data['quality'])
        raw_output = five_data['quality']
        

        # standardize the Data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        raw_input_scaled = scaler.fit_transform(raw_input)
        # split the data
        ordinal_train, ordinal_test, targetordinal_train, targetordinal_test = train_test_split(X_scaled, target_ordinal,
                                                                                                    test_size=0.5,
                                                                                                    random_state=100)

        # fit to the model

        ordinal_probit = OrderedModel(endog=targetordinal_train, exog=ordinal_train, link='probit', ordered=True)
        ordinal_probit = ordinal_probit.fit(method='bfgs', maxiter=1000)

        ordinal_logit = OrderedModel(endog=targetordinal_train, exog=ordinal_train, link='logit', ordered=True)
        ordinal_logit = ordinal_logit.fit(method='bfgs', maxiter=1000)


        # predict the model
        predicted_logit = ordinal_probit.predict(exog=ordinal_test)
        pred_choice_logit = predicted_logit.argmax(axis=1)

        predicted_probit = ordinal_logit.predict(exog=ordinal_test)
        pred_choice_probit = predicted_probit.argmax(axis=1)


        # # True labels
        true_labels = np.asarray(targetordinal_test.cat.codes)

        # Model Evaluation: Calculating accuracy and other classification metrics
        test_accuracy_logit = accuracy_score( true_labels , pred_choice_logit)
        test_accuracy_probit = accuracy_score( true_labels , pred_choice_probit)
        print("-" * 50)
        print("Test Accuracy of probit ordinal regressionm without categorization : {:.2f}%".format(test_accuracy_logit * 100))
        print("Test Accuracy of logit ordinal regression without categorization: {:.2f}%".format(test_accuracy_probit * 100))

        # five data test
        predicted_logit1 = ordinal_logit.predict(exog=raw_input_scaled)
        pred_choice_logit1 = predicted_logit1.argmax(axis=1)

        predicted_probit1 = ordinal_probit.predict(exog=raw_input_scaled)
        pred_choice_probit1 =  predicted_probit1.argmax(axis=1)


        # Map ordinal labels to quality labels
        predicted_quality = pd.Series(pred_choice_probit1).map({0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8}).values

        quality_categorical = pd.Categorical(five_data['quality'], ordered=True)
        true_labels = quality_categorical.codes



        # Create a DataFrame to store the actual and predicted labels
        df = pd.DataFrame({'Actual Labels': raw_output, 'Predicted Labels':  predicted_quality })

        # Print the DataFrame
        print("-" * 50)
        print(df)
        print("-" * 50)
        test_accuracy_logit1 = accuracy_score(raw_output,  predicted_quality)
        test_accuracy_probit1 = accuracy_score(raw_output,  predicted_quality)
        print("Test Accuracy of probit ordinal regression  without categorization for five raw data: {:.2f}%".format(test_accuracy_logit1 * 100))
        print("Test Accuracy of logit ordinal regression  without categorization for five raw data: {:.2f}%".format(test_accuracy_probit1 * 100))

        print("-" * 50)
        print("Classification Report for  probit and logit model without categorization for five raw data:")
        print(classification_report(raw_output,  predicted_quality, zero_division=1))




if __name__ == '__main__':
    # Create an instance of the WineQualityClassifier class
    wine_classifier = WineQualityClassifier()

    # Load the wine data
    wine_classifier.load_data("winequality_dataset/winequality-red.csv")

    # Preprocess the data
    wine_classifier.preprocess_data()

    # Split the data into train and test sets
    wine_classifier.split_data()

    # Train the models
    wine_classifier.train_models()

    # Perform cross-validation
    wine_classifier.perform_cross_validation()
    print("-" * 50)
    # Print the cross-validation scores
    print("Cross-validation scores for L1 regularization: ", wine_classifier.scores)

    wine_classifier1 = WineQualityClassifier1()

    wine_classifier1.skmultinomial()

    qualityWithoutCategorization = WineQualitywithoutCategorization()

    qualityWithoutCategorization.statsOrdinal()





