import sys
import os
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from src.exception import CustomError
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "preprocessor.pkl"
    )  # location where the preprocessor object will be stored after transformation


# class for changing the data type of the features based on the domain knowledge
class TypeCaster(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.int_to_str_features: list[str] = ["MSSubClass", "OverallCond"]

    def fit(self, X: pd.DataFrame, y=None):
        self._is_fitted = True
        return self

    def transform(self, X):
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted yet.")
        X = X.copy()
        for feature in X.columns:
            if feature in self.int_to_str_features:
                X[feature] = X[feature].astype(str)
        logging.info(
            f"TypeCaster: Transformed {self.int_to_str_features} features to string as they represent category rather than numerical imporatance. "
        )
        return X


class DomainImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.none_features = [
            "poolqc",
            "miscfeature",
            "alley",
            "fence",
            "masvnrtype",
            "fireplacequ",
            "garagetype",
            "garagefinish",
            "garagequal",
            "garagecond",
            "bsmtexposure",
            "bsmtfintype2",
            "bsmtqual",
            "bsmtcond",
            "bsmtfintype1",
        ]
        self.zero_features = ["masvnrarea", "garageyrblt"]
        self.lot_frontage_median: dict[str, float] = (
            {}
        )  # store the median values of 'LotFrontage' for each 'Neighborhood'
        self.electrical_mode: str = ""  # store the mode value of 'Electrical' column
        self.global_lot_frontage_median: float = (
            0.0  # store the global median value of 'LotFrontage' in case there are neighborhoods in test data which are not present in training data
        )

    # this function is for learning the parameters quired for imputation from the training data and storing those parameters in the instance variables of the class which will be used later for imputation in both training and test data
    def fit(self, X: pd.DataFrame, y=None):
        if "LotFrontage" in X.columns and "neighborhood" in X.columns:
            for neighborhood in X["neighborhood"].unique():
                median_value = X.loc[
                    X["neighborhood"] == neighborhood, "lotFrontage"
                ].median()  # extracting the median value of 'LotFrontage' for each 'Neighborhood'
                logging.info(
                    f"DomainImputer: Calculated median value of 'LotFrontage' for neighborhood '{neighborhood}' is {median_value}."
                )
                self.lot_frontage_median[neighborhood] = (
                    median_value  # storing the median values in a dictionary with neighborhood as key and median value as value
                )
                self.global_lot_frontage_median = np.mean(
                    list(self.lot_frontage_median.values())
                )  # calculating the global median value of 'LotFrontage' by taking the mean of all the neighborhood median values
        if "electrical" in X.columns:
            logging.info(
                f"DomainImputer: Calculated mode value of 'Electrical' column is {X['electrical'].mode()[0]}."
            )
            self.electrical_mode = X["electrical"].mode()[
                0
            ]  # extracting the mode value of 'Electrical' column and storing it in a variable
        return self

    def transform(self, X):
        X = X.copy()
        for feature in X.columns:
            if feature in self.none_features:
                logging.info(
                    f"DomainImputer: Filling missing values of '{feature}' with 'None' as it represents absence of the feature."
                )
                X[feature] = X[feature].fillna(
                    "None"
                )  # filling the missing values with 'None' for the features in none_features list
            elif feature in self.zero_features:
                logging.info(
                    f"DomainImputer: Filling missing values of '{feature}' with 0 as it represents absence of the feature."
                )
                X[feature] = X[feature].fillna(
                    0
                )  # filling the missing values with 0 for the features in zero_features list
            elif feature == "lotFrontage":
                logging.info(
                    f"DomainImputer: Filling missing values of 'LotFrontage' based on the median value for each neighborhood."
                )
                for neighborhood, median_value in self.lot_frontage_median.items():
                    if neighborhood in X["neighborhood"].unique():
                        X.loc[X["neighborhood"] == neighborhood, "lotFrontage"] = X.loc[
                            X["neighborhood"] == neighborhood, "lotFrontage"
                        ].fillna(
                            median_value
                        )  # filling the missing values of 'LotFrontage' with the corresponding median value based on the neighborhood
                    else:
                        logging.info(
                            f"DomainImputer: Neighborhood '{neighborhood}' not found in the data. Filling missing values of 'LotFrontage' with global median value {self.global_lot_frontage_median}."
                        )
                        X["lotFrontage"] = X["lotFrontage"].fillna(
                            self.global_lot_frontage_median
                        )  # filling the missing values of 'LotFrontage' with global median value in case there are neighborhoods in test data which are not present in training data
            elif feature == "electrical":
                logging.info(
                    f"DomainImputer: Filling missing values of 'Electrical' column with mode value '{self.electrical_mode}' as it is the most common value in the column."
                )
                X[feature] = X[feature].fillna(
                    self.electrical_mode
                )  # filling the missing values of 'Electrical' column with the mode value
            return X


# class for dropping the constant numerical features from the data based on the variance threshold
class DropConstantNumerical(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cols_to_drop: list[str] = (
            []
        )  # store the list of constant numerical features which will be identified during fitting and dropped during transformation
        self.threshold: float = (
            0.01  # threshold for identifying constant features based on variance
        )

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        target = "saleprice"
        numerical = [
            feature
            for feature in X.select_dtypes(include=np.number).columns
            if feature != target
        ]
        vt = VarianceThreshold(self.threshold)
        vt.fit(
            X[numerical]
        )  # checking the constant features of numerical features based on threshold 0.01
        variance_result = (
            vt.get_support()
        )  # getting the result of checking the passed data with the threshold
        self.cols_to_drop = [
            col for col, s in zip(X[numerical], variance_result) if not s
        ]  # storing those numerical features whose variance is below 0.01, which means those features are constant and do not have any importance in the prediction of the target variable, in a list which will be used later for dropping those features from the data
        return self

    def transform(self, X):
        X = X.copy()
        logging.info(
            f"DropConstantNumerical: Dropping constant numerical features {self.cols_to_drop} as their variance is below the threshold of {self.threshold}."
        )
        X = X.drop(
            columns=[col for col in self.cols_to_drop if col in X.columns]
        )  # dropping the constant numerical features from the data
        return X


# class for dropping the constant categorical features from the data based on the mutual information with the target variable
class DropConstantCategorical(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cols_to_drop: list[str] = (
            []
        )  # store the list of constant categorical features which will be identified during fitting and dropped during transformation
        self.mi_threshold: float = (
            0.01  # threshold for identifying constant features based on frequency
        )

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        ordinal_categories = {
            "lotshape": ["IR3", "IR2", "IR1", "Reg"],
            "exterqual": ["Po", "Fa", "TA", "Gd", "Ex"],
            "extercond": ["Po", "Fa", "TA", "Gd", "Ex"],
            "bsmtqual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
            "bsmtcond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
            "bsmtexposure": ["None", "No", "Mn", "Av", "Gd"],
            "bsmtfintype1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
            "bsmtfintype2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
            "heatingqc": ["Po", "Fa", "TA", "Gd", "Ex"],
            "kitchenqual": ["Po", "Fa", "TA", "Gd", "Ex"],
            "functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
            "fireplacequ": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
            "garagefinish": ["None", "Unf", "RFn", "Fin"],
            "garagequal": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
            "garagecond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
            "paveddrive": ["N", "P", "Y"],
            "poolqc": ["None", "Fa", "TA", "Gd", "Ex"],
            "fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"],
            "electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
        }
        categorical = [
            feature
            for feature in X.select_dtypes(include="object").columns
            if feature != "saleprice"
        ]

        quasi_constant_cat_features = [
            feature
            for feature in categorical
            if X[feature].value_counts(normalize=True).iloc[0] > 0.95
        ]  # if the most frequently occuring category of this current feature is seen in almost 99% of training data, then this feature is considered as constant categorical feature
        mi_storage = {}
        # instead of directly dropping the near constant categorical features, we must check the statistical relation of these near constant features with the target variable then only we drop those features which has very weak relation with target variable
        for feature in quasi_constant_cat_features:
            if feature in ordinal_categories:
                mapping = {
                    k: i for i, k in enumerate(ordinal_categories[feature])
                }  # creating the dictionary which shows the category as key and its corresponding index as value, here index represents the position of the category in the given order of current nominal categorical feature
                x = X[feature].map(mapping).to_frame(name=feature)
            else:
                x = (
                    X[feature].astype("category").cat.codes.to_frame(name=feature)
                )  # converting the current categorical features value into cat codes, which is done only for nominal categorical features
            mi = mutual_info_regression(
                x, y.loc[x.index]
            )  # calculating the statistical relationship between categories and the target variable, and here we are using .loc[x.index] for matching the corresponding training data and output variable
            mi_storage[feature] = mi[
                0
            ]  # storing the feature as key and the mi value as the value in the dict
        self.cols_to_drop = [
            feature
            for feature, mi_value in mi_storage.items()
            if mi_value < self.mi_threshold
        ]  # we drop those categorical features that have very low statistical relationship with the target variables
        return self

    def transform(self, X):
        X = X.copy()
        logging.info(
            f"DropConstantCategorical: Dropping constant categorical features {self.cols_to_drop} as their maximum category frequency is above the threshold of {self.threshold}."
        )
        X = X.drop(
            columns=[col for col in self.cols_to_drop if col in X.columns]
        )  # dropping the constant categorical features from the data
        return X


class NumericFeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cols_to_drop: list[str] = []
        self.corr_target_threshold: float = (
            0.01  # threshold for identifying constant features based on correlation with target variable
        )
        self.corr_feature_threshold: float = (
            0.75  # threshold for identifying constant features based on correlation with other features
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = X.copy()
        target = "saleprice"
        imp_num_features = []
        numerical = [
            feature
            for feature in X.select_dtypes(include=np.number).columns
            if feature != target
        ]
        for feature in numerical:
            corr = X[feature].corr(
                y.loc[X.index]
            )  # finding the correlation of all the numeric features with the target variable
        if abs(corr) > self.corr_target_threshold:
            imp_num_features.append(feature)
        imp_corr_matrix = X[
            imp_num_features
        ].corr()  # finding the correlation of each important numerical features with eachother
        for i in range(len(imp_corr_matrix.columns)):
            for j in range(i):
                r = imp_corr_matrix.iloc[
                    i, j
                ]  # extracting the correlation value between ith and jth features
                ci = imp_corr_matrix.columns[i]  # extracting the ith feature
                cj = imp_corr_matrix.columns[j]  # extracting the jth feature
                if (
                    ci != cj
                    and abs(r) > self.corr_feature_threshold
                    and ci != target
                    and cj != target
                ):  # if the correlation value between this pair is high, then we remove that feature which has weaker correlation with the target variable
                    corr_i_target = X[ci].corr(
                        y.loc[X.index]
                    )  # finding the correlation of the ith index feature with the target variable
                    corr_j_target = X[cj].corr(
                        y.loc[X.index]
                    )  # finding the correlation of the jth index feature with the target variable
                    if ci in imp_num_features and abs(corr_i_target) < abs(
                        corr_j_target
                    ):
                        imp_num_features.remove(
                            ci
                        )  # removing the weaker feature or the feature which has weaker correlation with the target variable
                    elif cj in imp_num_features:
                        imp_num_features.remove(cj)
        self.cols_to_drop = [feature for feature in numerical if feature not in imp_num_features]  # storing the list of numerical features which are not important for the prediction of target variable and can be dropped from the data
        return self

    def transform(self, X):
        X = X.copy()
        logging.info(
            f"NumericFeatureSelection: Dropping numerical features {self.cols_to_drop} as their correlation with target variable is below the threshold of {self.corr_target_threshold} or they have high correlation with other features above the threshold of {self.corr_feature_threshold}."
        )
        X = X.drop(columns =[feature for feature in self.cols_to_drop if feature in X.columns] )
        return X
