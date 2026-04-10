import sys
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin  #for custom transformers for custom pipeline
from sklearn.pipeline import Pipeline
from src.exception import CustomError
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Optional
from src.utils import save_object, load_object

 

@dataclass
class DataTransformationConfig:
    #path for storing the trained prepipeline
    prepipeline_obj_file_path: str = os.path.join(
        "artifacts", "prepipeline.pkl"
    )
    #path for storing the trained feature selection pipeline
    feature_selection_obj_file_path: str = os.path.join(
        "artifacts", "feature_selection.pkl"
    )
    #path for storing the trained preprocessor object
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "preprocessor.pkl"
    )  # location where the preprocessor object will be stored after transformation


#CLASSES FOR PREPIPELINE CREATION
# class for changing the data type of the features based on the domain knowledge
class TypeCaster(BaseEstimator, TransformerMixin):
    def __init__(self):
       
        self.int_to_str_features: list[str] = ["mssubclass", "overallcond"]

    def fit(self, X: pd.DataFrame, y=None):
        self.is_fitted_ = True  #we must assign self.is_fitted_ = True in the fit method of custom transformers , otherwise sklearn will raise an error when we  try to use this transformer in the pipeline, because sklearn checks for the presence of self.is_fitted_ attribute in the custom transformer to check whether the transformer is fitted or not
        return self

    def transform(self, X):
        X = X.copy()
        for feature in X.columns:
            if feature in self.int_to_str_features:
                X[feature] = X[feature].astype(str)
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
        )  # store the median values of 'lotfrontage' based on 'neighborhood' category
        self.electrical_mode: str = ""  # store the mode value of 'Electrical' column
        self.global_lot_frontage_median: float = (
            0.0  # store the global median value of 'lotfrontage' in case there are neighborhoods in test data which are not present in training data
        )

    # this function is for learning the parameters quired for imputation from the training data and storing those parameters in the instance variables of the class which will be used later for imputation in both training and test data
    def fit(self, X: pd.DataFrame, y=None):
        self.is_fitted_ = True
        if "lotfrontage" in X.columns and "neighborhood" in X.columns:
            for neighborhood in X["neighborhood"].unique():  #looping through each unique data in the neighbourhood column
                median_value = X.loc[
                    X["neighborhood"] == neighborhood, "lotfrontage"
                ].median()  # extracting the median value of 'lotfrontage' of each category of the feature 'neighborhood'
                self.lot_frontage_median[neighborhood] = (
                    median_value  # storing the median values in a dictionary with neighborhood as key and median value as value
                )
            self.global_lot_frontage_median = np.median(
                    list(self.lot_frontage_median.values())
                )  # calculating the global median value of 'LotFrontage' by taking the mean of all the neighborhood median values
        if "electrical" in X.columns:
            self.electrical_mode = X["electrical"].mode()[
                0
            ]  # extracting the mode value of 'Electrical' column and storing it in a variable
        return self

    def transform(self, X):
        X = X.copy()
        #only if the current features of X are in the list of features which we have used for imputation
        for feature in self.zero_features:
            if feature in X.columns: X[feature] = X[feature].fillna(0)
        for feature in self.none_features:
            if feature in X.columns: X[feature] = X[feature].fillna("None")
        if "lotfrontage" in X.columns and "neighborhood" in X.columns:
            neighborhood_medians = X["neighborhood"].map(self.lot_frontage_median)  #getting the median value of 'lotfrontage' based on the category of 'neighbourhood'
            X["lotfrontage"] = X["lotfrontage"].fillna(neighborhood_medians).fillna(self.global_lot_frontage_median)

        if "electrical" in X.columns:
            X["electrical"] = X["electrical"].fillna(self.electrical_mode)
        return X

class CreateNewFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.date_features = ['yearbuilt','yrsold','garageyrblt','yearremodadd']
    def fit(self, X: pd.DataFrame, y=None):
        self.is_fitted_ = True
        return self
    def transform(self, X):
        import numpy as np
        X = X.copy()
        binary_features = {
            'has_pool': 'poolarea',
            'has_garage': 'garagearea',
            'has_fireplace': 'fireplaces',
            'has_basement': 'totalbsmtsf',
            'has_2nd_floor': '2ndflrsf',
            'has_masonry': 'masvnrarea'
        }
        #creation of age features
        #if the current X has the necessary features, then only we make the new features based on those required features such as 'yearbuilt','yearsold',.......    
        if all(c in X.columns for c in ['yearbuilt','yrsold']):
         X['house_age'] = X['yrsold'] - X['yearbuilt']  #creating the age of house
        if all(c in X.columns for c in ['garageyrblt','yrsold']): 
         X['garage_age'] = np.where(X['garageyrblt']!=0,X['yrsold'] - X['garageyrblt'],-1) #creating garage_age, if the garageyrblt is 0, then it means the house didn't had any garage, so we are filling garage age wth -1 in such case
        if all(c in X.columns for c in ['yearremodadd','yrsold']):
         X['remodeled_age'] = X['yrsold'] - X['yearremodadd'] #calculating the age of house since it was remodeled
        
        #creation of has features, showing whether the house has that feature or not
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
        from sklearn.feature_selection import VarianceThreshold
        self.is_fitted_ = True
        X = X.copy()
        target = "saleprice"
        numerical = [
            feature
            for feature in X.select_dtypes(include=np.number).columns
            if feature != target
        ]
        X[numerical] = X[numerical].fillna(X[numerical].median())  #filling the null values of numerical features with median before checking the variance threshold, because if there are null values in the data, then the variance threshold will not work properly
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
        from sklearn.feature_selection import  mutual_info_regression
        self.is_fitted_ = True
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
            for feature in X.select_dtypes(exclude=np.number).columns
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
                x = X[feature].map(mapping).to_frame(name=feature)  #converting into dataframe
            else:
                x = (
                    X[feature].astype("category").cat.codes.to_frame(name=feature)
                )  # converting the current categorical features value into cat codes, which is done only for nominal categorical features
            mi = mutual_info_regression(
                x, y.loc[x.index]
            )  # calculating the statistical relationship between categories and the target variable, and here we are using .loc[x.index] for matching the corresponding training data and output variable
            mi_storage[feature] = mi[
                0
            ]  # storing the feature as key and its mi value as the value in the dict
        self.cols_to_drop = [
            feature
            for feature, mi_value in mi_storage.items()
            if mi_value < self.mi_threshold
        ]  # we drop those nearly constant categorical features that have very low statistical relationship with the target variables
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(
            columns=[col for col in self.cols_to_drop if col in X.columns]
        )  # dropping the constant categorical features from the data
        return X


class NumericFeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cols_to_drop: list[str] = []
        self.corr_target_threshold: float = (
            0.1  # threshold for identifying constant features based on correlation with target variable
        )
        self.corr_feature_threshold: float = (
            0.75  # threshold for identifying constant features based on correlation with other features
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.is_fitted_ = True
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
        
        removed_cols = set()
        for i in range(len(imp_corr_matrix.columns)):
            for j in range(i):
                
                r = imp_corr_matrix.iloc[
                    i, j
                ]  # extracting the correlation value between ith and jth features
                ci = imp_corr_matrix.columns[i]  # extracting the ith feature
                cj = imp_corr_matrix.columns[j]  # extracting the jth feature
                
                if ci in removed_cols or cj in removed_cols:
                    continue  # if any of the two features is already removed, then we skip the correlation check for that pair of features


                if (
                    ci != cj
                    and abs(r) > self.corr_feature_threshold
                    and ci != target
                    and cj != target
                ):  # if the correlation value between this pair is high, then we remove that feature which has weaker correlation with the target variable, between two of them
                    corr_i_target = X[ci].corr(
                        y.loc[X.index]
                    )  # finding the correlation of the ith index feature with the target variable
                    corr_j_target = X[cj].corr(
                        y.loc[X.index]
                    )  # finding the correlation of the jth index feature with the target variable
                    if abs(corr_i_target) < abs(
                        corr_j_target
                    ):
                        removed_cols.add(
                            ci
                        )  # removing the weaker feature or the feature which has weaker correlation with the target variable
                    else:
                        removed_cols.add(
                            cj
                        )
        self.cols_to_drop = [feature for feature in numerical if feature not in imp_num_features or feature in removed_cols]  # storing the list of numerical features which are not important for the prediction of target variable and can be dropped from the data
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns =[feature for feature in self.cols_to_drop if feature in X.columns] )
        return X

#dropping the features which have high multicollinearity with other features and very weak correlation with target variable, because they are the redundant features
class MulticollinearityDropper(BaseEstimator, TransformerMixin):  #this applies only on numerical features
    def __init__(self):
        self.cols_to_drop: list[str] = []
        self.target_relation_threshold: float = 0.1
    def fit(self, X: pd.DataFrame, y: pd.Series):
        import numpy as np
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        self.is_fitted_ = True
        X = X.copy()
        target = "saleprice"
        numerical = [feature for feature in X.select_dtypes(include=np.number).columns if feature != target]
        variables = X[numerical].dropna()  #we must drop the null values before checking the variance inflation factor
        y = y.loc[variables.index]  #matching the corresponding target variable values with the non null values of numerical features
        vif = pd.DataFrame({
            'features':variables.columns,
            'vif_value':[variance_inflation_factor(variables.values,i) for i in range(variables.shape[1])],
            'corr_with_target':variables.corrwith(y)  #finding the correlation with the target variable based on their corresponding target 'y'
        }).sort_values('vif_value',ascending=False)
        self.cols_to_drop = vif[(vif['vif_value'] > 10) & (abs(vif['corr_with_target'])<self.target_relation_threshold)]['features'].tolist()  #those features which have higher vif value than 10 and very low correlation value of less than 0.1 with the target variable, we drop them 
        return self
    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=[feature for feature in self.cols_to_drop if feature in X.columns])
        return X
   

#dropping the categorical features which have weak statistical relationship with the target variable based on ANOVA test    
class DropWeakCategorical(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cols_to_drop: list[str] = []
    def fit(self, X: pd.DataFrame, y: pd.Series):
        from scipy import stats
        self.is_fitted_ = True
        target = "saleprice"
        categorical = [feature for feature in X.select_dtypes(exclude=np.number).columns if feature != target]
        anova_report = []
    
        for feature in categorical:
            groups = [y.loc[group.index].values for _,group in X.groupby(feature) if len(group)>0]  #extracting the values of the saleprice based on different categories of current feature, based on the index of the category
            if len(groups) <2:
                self.cols_to_drop.append(feature)  #if there is only one category in the current feature, then we can directly drop that feature as it doesn't have any statistical relationship with the target variable   
                continue  #skipping the current feature, as we cannot calculate the ANOVA test for the feature having only one category  
            f_stats,p_value = stats.f_oneway(*groups)  #anova test of different saleprice values based on each categories of current feature
            anova_report.append({
                'feature':feature,
                'f_stats':f_stats,
                'p_value':p_value
            })
        total_result_cat = pd.DataFrame(anova_report).sort_values('p_value')
        weak_anova_cols = total_result_cat[(total_result_cat['p_value']>0.05) | (total_result_cat['p_value'].isna())]['feature'].tolist()
        self.cols_to_drop += weak_anova_cols  #total cols to drop after finding weak ANOVA and single categorical feature
        return self
    def transform(self, X):
        X = X.copy()
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
        self.config = DataTransformationConfig()
        self.numerical_features:list[str] =[]
        self.categorical_features:list[str] = []
        self.ordinal_features:list[str] = []
        self.nominal_features:list[str] = []
        self._is_fitted_:bool = False
        self.pre_pipeline:Optional[Pipeline] = None  
        self.feature_selection_pipeline: Optional[Pipeline] = None 
        self.preprocessor: Optional[ColumnTransformer] = None  
    
    #this function is for updating the numerical, categorical, ordinal and nominal features after applying feature selection, feature engineering and so on
    def update_features_collection(self, X: pd.DataFrame) -> None:
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
        return Pipeline(prepipeline_steps)
        
        
    
    def build_feature_selection_pipeline(self):
        feature_selection_steps = [
            ("drop_constant_numerical", DropConstantNumerical()),
            ("drop_constant_categorical", DropConstantCategorical()),
            ("numeric_feature_selection", NumericFeatureSelection()),
            ("multicollinearity_dropper", MulticollinearityDropper()), 
            ("drop_weak_categorical", DropWeakCategorical())
        ]
        return Pipeline(feature_selection_steps)
       

    def numerical_pipeline(self):
        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),  
            ("scaler", StandardScaler())
        ])
        return numerical_pipeline

    def ordinal_pipeline(self):
        ordinal_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),  
            ("ordinal_encoder", OrdinalEncoder(categories=[self.ordinal_mapping[feature] for feature in self.ordinal_features], handle_unknown="use_encoded_value", unknown_value=-1))
        ])
        return ordinal_pipeline
    
    def nominal_pipeline(self):
        nominal_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),  
            ("onehot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        return nominal_pipeline
    def build_preprocessor(self):
        return ColumnTransformer([
            ("numerical_pipeline", self.numerical_pipeline(), self.numerical_features),
            ("ordinal_pipeline", self.ordinal_pipeline(), self.ordinal_features),
            ("nominal_pipeline", self.nominal_pipeline(), self.nominal_features)
        ])
        
   
    def fit(self, X: pd.DataFrame, y: pd.Series):  #for fitting and transforming the training data
        if 'id' in X.columns:
            X = X.drop(columns=['id'])  #dropping the 'id' column as it doesn't have any importance in the prediction of target variable and it is just a unique identifier for each row in the data
        y = np.log1p(y)  #taking the log of target variable to make it more normally distributed, as the distribution of saleprice is right skewed and taking log will make it more normal which will help the model to learn better
        self.pre_pipeline= self.build_prepipeline()   #built the prepipeline
        
        #AFTER EVERY TRAINING OF PIPELINE , WE UPDATE THE NUMERICAL,CATEGORICAL FEATURES AND SAVE THE TRAINED PIPELINE OBJECT
        X_preprocessed = self.pre_pipeline.fit_transform(X)   #applying the pre pipeline of typecasting, domain aware imputation and feature creation based on domain knowledge
        save_object(self.config.prepipeline_obj_file_path, self.pre_pipeline)  
        self.update_features_collection(X_preprocessed)  #updating the numerical, categorical, ordinal and nominal features based on the changes in the data after applying the pre pipeline
        
        self.feature_selection_pipeline =  self.build_feature_selection_pipeline()   #built the feature selection pipeline
        X_selected = self.feature_selection_pipeline.fit_transform(X_preprocessed, y) #applying the feature selection pipeline of dropping constant numerical features, dropping constant categorical features, selecting important numerical features based on correlation with target variable and correlation with other features, dropping the features which have high multicollinearity with other features and very weak correlation with target variable, and dropping the categorical features which have weak statistical relationship with the target variable based on ANOVA test
        save_object(self.config.feature_selection_obj_file_path, self.feature_selection_pipeline)
        self.update_features_collection(X_selected)  #updating the numerical, categorical, ordinal and nominal features based on the changes in the data after applying the feature selection pipeline
        
        self.preprocessor = self.build_preprocessor()  #building the preprocessor for encoding and scaling based on the updated numerical, ordinal and nominal features after feature selection
        self.preprocessor.fit(X_selected)  #can use either fit or fit_transform for the preprocessor method
        save_object(self.config.preprocessor_obj_file_path, self.preprocessor)
  
        self._is_fitted_ = True
        return self
    def transform(self, X: pd.DataFrame,y:Optional[pd.Series]=None):  #for transforming the test data based on the parameters learned from the training data
        import numpy as np
        if not self._is_fitted_:
            raise RuntimeError("DataTransformer not fitted yet.")
        if 'id' in X.columns:
            X = X.drop(columns=['id'])
        pre_pipeline = load_object(self.config.prepipeline_obj_file_path)  #loading the trained pre pipeline object      
        X = pre_pipeline.transform(X)  #applying the pre pipeline of typecasting, domain aware imputation and feature creation based on domain knowledge
        feature_selection_pipeline = load_object(self.config.feature_selection_obj_file_path)  #loading the trained feature selection pipeline object
        X = feature_selection_pipeline.transform(X) #applying the feature selection pipeline of dropping constant numerical features, dropping constant categorical features, selecting important numerical features based on correlation with target variable and correlation with other features, dropping the features which have high multicollinearity with other features and very weak correlation with target variable, and dropping the categorical features which have weak statistical relationship with the target variable based on ANOVA test
        preprocessor = load_object(self.config.preprocessor_obj_file_path)  #loading the trained preprocessor object for encoding and scaling based on the updated numerical, ordinal and nominal features after feature selection
        X_encoded = preprocessor.transform(X)
        if y is not None:
            y_logged = np.log1p(y).values  #taking the log of target variable to make it more normally distributed, and converting them into numpy array
            return X_encoded, y_logged  #combining the transformed features and target variable and returning them together
        return X_encoded
    


if __name__ == "__main__":
    try:
        df_train = pd.read_csv(os.path.join('root_data','train_data.csv'))
        df_test = pd.read_csv(os.path.join('root_data','test_data.csv'))
        dt = DataTransformer()
        df_train.columns = df_train.columns.str.lower()
        df_test.columns = df_test.columns.str.lower()
        X_train = df_train.drop(columns=['saleprice'])
        y_train = df_train['saleprice']
        dt.fit(X_train, y_train)  #fitting the data transformer on the training data to learn the parameters required for transformation and feature selection
        X_train_encoded, y_train_logged = dt.transform(X_train, y_train)
        X_test_encoded = dt.transform(df_test) #as the target saleprice is not present in the test data, so need for dropping the unavailable feature
    except Exception as e:
        raise CustomError(e, sys)    
