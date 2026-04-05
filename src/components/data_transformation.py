import sys
import os
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from src.exception import CustomError
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from typing import Optional


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "preprocessor.pkl"
    )  # location where the preprocessor object will be stored after transformation


#CLASSES FOR PREPIPELINE CREATION
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

#class for filling the null values based on domain knowledge
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

class CreateNewFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.date_features = ['yearbuilt','yrsold','garageyrblt','yearremodadd']
    def fit(self, X: pd.DataFrame, y=None):
        return self
    def transform(self, X):
        binary_features = {
            'has_pool': 'poolarea',
            'has_garage': 'garagearea',
            'has_fireplace': 'fireplaces',
            'has_basement': 'totalbsmtsf',
            'has_2nd_floor': '2nXlrsf',
            'has_masonry': 'masvnrarea'
        }
        if all(c in X.columns for c in ['yearbuilt','yrsold']):
         X['house_age'] = X['yrsold'] - X['yearbuilt']  #creating the age of house
        if all(c in X.columns for c in ['garageyrblt','yrsold']): 
         X['garage_age'] = np.where(X['garageyrblt']!=0,X['yrsold'] - X['garageyrblt'],-1) #creating garage_age, if the garageyrblt is 0, then it means the house didn't had any garage, so we are filling garage age wth -1 in such case
        if all(c in X.columns for c in ['yearremodadd','yrsold']):
         X['remodeled_age'] = X['yrsold'] - X['yearremodadd'] #calculating the age of house since it was remodeled
        for new_feature, original_feature in binary_features.items():
            if original_feature in X.columns:
                X[new_feature] = (X[original_feature] > 0).astype(int) 
               

        cols_to_drop = [feature for feature in self.date_features if feature in X.columns]
        X = X.drop(columns=cols_to_drop)
        return X







#CLASSES FOR FEATURE SELECTION
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

    def fit(self, X: pd.DataFrame, y:pd.Series):
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

#dropping the features which have high multicollinearity with other features and very weak correlation with target variable, because they are the redundant features
class MulticollinearityDropper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cols_to_drop: list[str] = []
        self.target_relation_threshold: float = 0.1
    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = X.copy()
        target = "saleprice"
        numerical = [feature for feature in X.select_dtypes(include=np.number).columns if feature != target]
        variables = X[numerical].dropna()  #we must drop the null values before checking the variance inflation factor
        vif = pd.DataFrame({
            'features':variables.columns,
            'vif_value':[variance_inflation_factor(variables.values,i) for i in range(variables.shape[1])],
            'corr_with_target':X[numerical].corrwith(y.loc[X.index])  #finding the correlation with the target variable based on their corresponding target 'y'
        }).sort_values('vif_value',ascending=False)
        self.cols_to_drop = vif[(vif['vif_value'] > 10) & (abs(vif['corr_with_target'])<self.target_relation_threshold)]['features'].tolist()  #those features which have higher vif value than 10 and very low correlation value of less than 0.1 with the target variable, we drop them 
        return self
    def transform(self, X):
        X = X.copy()
        logging.info(
            f"MulticollinearityDropper: Dropping features {self.cols_to_drop} due to high multicollinearity with other features and low correlation with target variable."
        )
        X = X.drop(columns=[feature for feature in self.cols_to_drop if feature in X.columns])
        return X
   

#dropping the categorical features which have weak statistical relationship with the target variable based on ANOVA test    
class DropWeakCategorical(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cols_to_drop: list[str] = []
    def fit(self, X: pd.DataFrame, y: pd.Series):
        target = "saleprice"
        categorical = [feature for feature in X.select_dtypes(exclude=np.number).columns if feature != target]
        anova_report = []
        for feature in categorical:
            groups = [y.loc[group.index].values for _,group in X.groupby(feature)]  #extracting the values of the saleprice based on different categories of current feature, based on the index of the category
            f_stats,p_value = stats.f_oneway(*groups)  #anova test of different saleprice values based on each categories of current feature
            anova_report.append({
                'feature':feature,
                'f_stats':f_stats,
                'p_value':p_value
            })
        total_result_cat = pd.DataFrame(anova_report).sort_values('p_value')
        self.cols_to_drop= total_result_cat[(total_result_cat['p_value']>0.05) | (total_result_cat['p_value'].isna())]['feature'].tolist()
        return self
    def transform(self, X):
        X = X.copy()
        logging.info(
            f"DropWeakCategorical: Dropping features {self.cols_to_drop} due to weak statistical relationship with target variable based on ANOVA test."
        )
        X = X.drop(columns=[feature for feature in self.cols_to_drop if feature in X.columns])
        return X
    

#classes for numerical and categorical encoding and scaling
class DataTransformer(BaseEstimator, TransformerMixin):
    nominal_categories = [
    'mssubclass', 'mszoning', 'alley', 'landcontour', 'lotconfig',
    'neighborhood', 'condition1', 'condition2', 'bldgtype', 'housestyle',
    'roofstyle', 'roofmatl', 'exterior1st', 'exterior2nd', 'masvnrtype',
    'foundation', 'heating', 'centralair', 'garagetype', 'miscfeature',
     'saletype', 'salecondition'
]
    ordinal_mapping = {
    'lotshape':      ['IR3', 'IR2', 'IR1', 'Reg'],
    'exterqual':     ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'extercond':     ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'bsmtqual':      ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'bsmtcond':      ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'bsmtexposure':  ['None', 'No', 'Mn', 'Av', 'Gd'],
    'bsmtfintype1':  ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'bsmtfintype2':  ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'heatingqc':     ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'kitchenqual':   ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'functional':    ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
    'fireplacequ':   ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'garagefinish':  ['None', 'Unf', 'RFn', 'Fin'],
    'garagequal':    ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'garagecond':    ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'paveddrive':    ['N', 'P', 'Y'],
    'poolqc':        ['None', 'Fa', 'TA', 'Gd', 'Ex'],
    'fence':         ['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'],
    'electrical':    ['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr'],
}

    def __init__(self):
        self.numerical_features:list[str] =[]
        self.categorical_features:list[str] = []
        self.ordinal_features:list[str] = []
        self.nominal_features:list[str] = []
        self._is_fitted:bool = False
        self.pre_pipeline:Optional[Pipeline] = None
        self.feature_selection_pipeline: Optional[Pipeline] = None
        self.preprocessor: Optional[ColumnTransformer] = None  
    
    #this function is for updating the numerical, categorical, ordinal and nominal features 
    def remaining_features(self, X: pd.DataFrame) -> list[str]:
        self.categorical_features = [feature for feature in X.select_dtypes(exclude=np.number).columns]
        self.numerical_features = [feature for feature in X.select_dtypes(include=np.number).columns if feature != 'saleprice']
        self.ordinal_features = [feature for feature in self.categorical_features if feature in self.ordinal_mapping.keys()]
        self.nominal_features = [feature for feature in self.categorical_features if feature in self.nominal_categories]


    #this function is for building the pre pipeline of typecasting, domain aware imputation and feature creation based on domain knowledge, which will be applied before feature selection in the data transformation process
    def build_prepipeline(self):
        prepipeline_steps = [
            ("type_caster", TypeCaster()),
            ("domain_imputer", DomainImputer()),
            ("feature_creator", CreateNewFeatures())
        ]
        self.pre_pipeline = Pipeline(prepipeline_steps)
        return self.pre_pipeline
    
    def build_feature_selection_pipeline(self):
        feature_selection_steps = [
            ("drop_constant_numerical", DropConstantNumerical()),
            ("drop_constant_categorical", DropConstantCategorical()),
            ("numeric_feature_selection", NumericFeatureSelection()),
            ("multicollinearity_dropper", MulticollinearityDropper()), 
            ("drop_weak_categorical", DropWeakCategorical())
        ]
        self.feature_selection_pipeline = Pipeline(feature_selection_steps)
        return self.feature_selection_pipeline

    def numerical_pipeline(self):
        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),  
            ("scaler", StandardScaler())
        ])
        return numerical_pipeline

    