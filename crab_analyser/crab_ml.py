import logging

import category_encoders as ce
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import linear_model
from sklearn.compose import TransformedTargetRegressor, make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, RobustScaler

logger = logging.getLogger('crabdata')


class CrabAgePredictor:
    def __init__(self, crab_data):
        """
        constructor
        :param crab_data:
        """
        self.crab_data = crab_data
        self.columns = ["length", "diameter", "height", "weight", "shucked_weight", "viscera_weight", "shell_weight",
                        "age", "sex"]
        self.continuous_var_columns = ["length", "diameter", "height", "weight", "shucked_weight", "viscera_weight",
                                       "shell_weight", "age"]

    def reverse_ohe(self, row):
        """
        reverse codes onehotencoding required for OLS

        :param row:
        :return:
        """
        if row["sex_F"] == 1:
            return 'F'
        if row["sex_M"] == 1:
            return 'M'
        if row["sex_I"] == 1:
            return 'I'
        # this should never happen
        return 'na'

    def pre_process_data(self):
        """
        Imputes missing values and removes outliers more than 3 std away
        :return:
        """
        logger.debug("Preprocessing called")
        self.crab_data.fillna(self.crab_data.mean(), inplace=True)
        raw_df_cont = self.crab_data[self.continuous_var_columns]
        x = raw_df_cont[~(np.abs(stats.zscore(raw_df_cont)) < 3).all(axis=1)]
        return self.crab_data.drop(x.index).copy().reset_index(drop=True)

    def ols_prediction(self):
        """
        uses linear regression after standardising to normal dist
        prints out accuracy metrics and then saves the design matrix with y and predicted y as a csv file
        also creates another column to calculate relative percentage difference between y and predicted y
        :return:
        """
        logger.info("running Linear Regression model")
        crab_df_woo = self.pre_process_data()
        transformer = QuantileTransformer(output_distribution='normal')
        # since I observed that the data was skewed, I decided to transform the continuous variables to normal dist
        reg = linear_model.LinearRegression()
        t_reg = TransformedTargetRegressor(regressor=reg, transformer=transformer)
        ohe = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True, drop_invariant=True)
        crab_df_woo_enc = ohe.fit_transform(crab_df_woo)
        X = crab_df_woo_enc.drop("age", axis=1)
        y = crab_df_woo_enc[["age"]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
        t_reg.fit(X_train, y_train)
        s = t_reg.score(X_test, y_test)
        logger.info("R-squared from Linear Regression is: {0}".format(s))
        y_pred = t_reg.predict(X)
        mse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        logger.debug("Linear Regression MAE: {0}".format(mae))
        logger.debug("Linear Regression RMSE: {0}".format(mse))
        logger.debug("Linear Regression R-squared: {0}".format(s))

        crab_df = X.copy()
        crab_df["age"] = pd.Series(y.values.ravel())
        crab_df["age_ols"] = pd.Series(y_pred.ravel())
        crab_df['sex'] = crab_df.apply(lambda row: self.reverse_ohe(row), axis=1)
        crab_df.drop(["sex_I", "sex_M", "sex_F"], axis=1, inplace=True)
        crab_df["percentage_difference"] = np.abs(
            np.divide((crab_df["age"] - crab_df["age_ols"]), crab_df["age"]) * 100)
        crab_df.to_csv("crab_predit_ols.csv", index=False)
        logger.info("Crab data with predicted variables saved: {0}".format("crab_predit_ols.csv"))
        logger.info("Linear Regression execution finished")

    def rf_prediction(self):
        """
        uses ensemble (Random Forest) method to predict crab age
        :return:
        """
        logger.info("running Random Forest model")
        X = self.crab_data.drop("age", axis=1)
        y = self.crab_data[["age"]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
        #
        numerical_features = X_train.dtypes == 'float'
        categorical_features = ~numerical_features
        # I used pipelining so that the predicted values were automatically transformed/scaled back
        preprocess = make_column_transformer(
            (RobustScaler(), numerical_features),
            (OneHotEncoder(sparse=False), categorical_features)
        )
        forest = RandomForestRegressor(n_estimators=5000, max_depth=20, min_samples_leaf=2, min_samples_split=4,
                                       random_state=100)
        f_reg = Pipeline(steps=[('preprocess', preprocess), ('model', forest)])
        f_reg_ttr = TransformedTargetRegressor(regressor=f_reg)
        f_reg_ttr.fit(X_train, y_train)
        s = f_reg_ttr.score(X_test, y_test)
        logger.info("R-squared from Random Forest is: {0}".format(s))
        y_pred = f_reg_ttr.predict(X)
        mse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        logger.debug("RandomForest MAE: {0}".format(mae))
        logger.debug("RandomForest RMSE: {0}".format(mse))
        logger.debug("RandomForest R-squared: {0}".format(s))
        # recreate the original dataset
        crab_df = X.copy()
        crab_df["age"] = pd.Series(y.values.ravel())
        crab_df["age_forest"] = pd.Series(y_pred.ravel())
        crab_df["percentage_difference"] = np.abs(
            np.divide((crab_df["age"] - crab_df["age_forest"]), crab_df["age"]) * 100)
        crab_df.to_csv("crab_predit_forest.csv", index=False)
        logger.info("Crab data with predicted variables saved: {0}".format("crab_predit_forest.csv"))
        logger.info("Random Forest execution finished")

    def run(self):
        """
        main function for the class
        :return:
        """
        logger.info("run called")
        self.ols_prediction()
        self.rf_prediction()
        logger.info("machine learning process finished")
