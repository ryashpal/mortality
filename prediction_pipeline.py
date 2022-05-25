import logging
import sys

log = logging.getLogger("")
log.setLevel(logging.INFO)
format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
log.addHandler(ch)

def readData(target):

    import pandas as pd
    import psycopg2

    # information used to create a database connection
    sqluser = 'postgres'
    dbname = 'mimic4'
    hostname = 'localhost'
    port_number = 5434

    # Connect to postgres with a copy of the MIMIC-III database
    con = psycopg2.connect(dbname=dbname, user=sqluser, host=hostname, port=port_number, password='mysecretpassword')

    dataQuery = """select * from sepsis_micro.data_matrix_qc_1;"""
    dataDf = pd.read_sql_query(dataQuery, con)

    log.info('Formatting data')

    dropCols = [
        'micro_specimen_id',
        'person_id',
        'seven_day_mortality',
        'fourteen_day_mortality',
        'twentyone_day_mortality',
        'twentyeight_day_mortality',
        'sixty_day_mortality',
        'ninety_day_mortality',
        'onetwenty_day_mortality',
        'Ambulatory Clinic / Center',
        'Ambulatory Surgical Center',
        'Emergency Room - Hospital',
        'Emergency Room and Inpatient Visit',
        'Inpatient Visit',
        'Observation Room',
        'AMBULATORY OBSERVATION',
        'DIRECT EMER.',
        'ELECTIVE',
        'EU OBSERVATION',
        'EW EMER.',
        'OBSERVATION ADMIT',
        'SURGICAL SAME DAY ADMISSION',
        'URGENT',
        'AMBULATORY SURGERY TRANSFER',
        'CLINIC REFERRAL',
        'EMERGENCY ROOM',
        'INFORMATION NOT AVAILABLE',
        'INTERNAL TRANSFER TO OR FROM PSYCH',
        'PACU',
        'PHYSICIAN REFERRAL',
        'PROCEDURE SITE',
        'TRANSFER FROM HOSPITAL',
        'TRANSFER FROM SKILLED NURSING FACILITY',
        'WALK-IN/SELF REFERRAL',
    ]

    X = dataDf.drop(dropCols, axis = 1)
    XVitalsMax = dataDf[['temp_max', 'heartrate_max', 'breath_rate_vent_max', 'breath_rate_spon_max', 'resp_rate_max', 'oxygen_max', 'sysbp_max', 'diabp_max', 'meanbp_max', 'sysbp_ni_max', 'diabp_ni_max', 'meanbp_ni_max', 'gcs_motor_max', 'gcs_verbal_max', 'gcs_eye_max']]
    XVitalsMin = dataDf[['temp_min', 'heartrate_min', 'breath_rate_vent_min', 'breath_rate_spon_min', 'resp_rate_min', 'oxygen_min', 'sysbp_min', 'diabp_min', 'meanbp_min', 'sysbp_ni_min', 'diabp_ni_min', 'meanbp_ni_min', 'gcs_motor_min', 'gcs_verbal_min', 'gcs_eye_min']]
    XVitalsAvg = dataDf[['temp_avg', 'heartrate_avg', 'breath_rate_vent_avg', 'breath_rate_spon_avg', 'resp_rate_avg', 'oxygen_avg', 'sysbp_avg', 'diabp_avg', 'meanbp_avg', 'sysbp_ni_avg', 'diabp_ni_avg', 'meanbp_ni_avg', 'gcs_motor_avg', 'gcs_verbal_avg', 'gcs_eye_avg']]
    XVitalsSd = dataDf[['temp_sd', 'heartrate_sd', 'breath_rate_vent_sd', 'breath_rate_spon_sd', 'resp_rate_sd', 'oxygen_sd', 'sysbp_sd', 'diabp_sd', 'meanbp_sd', 'sysbp_ni_sd', 'diabp_ni_sd', 'meanbp_ni_sd', 'gcs_motor_sd', 'gcs_verbal_sd', 'gcs_eye_sd']]
    XVitalsFirst = dataDf[['temp_first', 'heartrate_first', 'breath_rate_vent_first', 'breath_rate_spon_first', 'resp_rate_first', 'oxygen_first', 'sysbp_first', 'diabp_first', 'meanbp_first', 'sysbp_ni_first', 'diabp_ni_first', 'meanbp_ni_first', 'gcs_motor_first', 'gcs_verbal_first', 'gcs_eye_first']]
    XVitalsLast = dataDf[['temp_last', 'heartrate_last', 'breath_rate_vent_last', 'breath_rate_spon_last', 'resp_rate_last', 'oxygen_last', 'sysbp_last', 'diabp_last', 'meanbp_last', 'sysbp_ni_last', 'diabp_ni_last', 'meanbp_ni_last', 'gcs_motor_last', 'gcs_verbal_last', 'gcs_eye_last']]
    XLabsMax = dataDf[['potassium_max', 'chloride_max', 'glucose_max', 'sodium_max', 'bicarbonate_max', 'hemoglobin_max', 'creatinine_max']]
    XLabsMin = dataDf[['potassium_min', 'chloride_min', 'glucose_min', 'sodium_min', 'bicarbonate_min', 'hemoglobin_min', 'creatinine_min']]
    XLabsAvg = dataDf[['potassium_avg', 'chloride_avg', 'glucose_avg', 'sodium_avg', 'bicarbonate_avg', 'hemoglobin_avg', 'creatinine_avg']]
    XLabsSd = dataDf[['potassium_sd', 'chloride_sd', 'glucose_sd', 'sodium_sd', 'bicarbonate_sd', 'hemoglobin_sd', 'creatinine_sd']]
    XLabsFirst = dataDf[['potassium_first', 'chloride_first', 'glucose_first', 'sodium_first', 'bicarbonate_first', 'hemoglobin_first', 'creatinine_first']]
    XLabsLast = dataDf[['potassium_last', 'chloride_last', 'glucose_last', 'sodium_last', 'bicarbonate_last', 'hemoglobin_last', 'creatinine_last']]
    y = dataDf[target + '_mortality']

    return X, XVitalsMax, XVitalsMin, XVitalsAvg, XVitalsSd, XVitalsFirst, XVitalsLast, XLabsMax, XLabsMin, XLabsAvg, XLabsSd, XLabsFirst, XLabsLast, y


def performSfs(X, y):
    log.info('Performing SFS')

    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.tree import DecisionTreeClassifier

    sfs = SequentialFeatureSelector(estimator = DecisionTreeClassifier(), n_features_to_select=25)
    sfs.fit(X, y)

    XMin = X[sfs.get_feature_names_out()]
    return XMin


def buildMLPModel(X, y, layerSize):

    log.info('Building the model')

    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes = (layerSize, layerSize))
    mlp.fit(X, y)

    log.info('Performing cross-validation')

    from sklearn.model_selection import cross_validate

    mlpScores = cross_validate(mlp, X, y, cv=5, scoring=['accuracy', 'f1', 'roc_auc'])
    return mlpScores

def buildLGBMModel(X, y):
    log.info('Performing Hyperparameter optimisation')

    from lightgbm import LGBMClassifier

    from sklearn.model_selection import GridSearchCV

    parameters={
        'max_depth': [6, 9, 12],
        'scale_pos_weight': [0.1, 0.15, 0.2, 0.25, 0.3],
    }

    clf = GridSearchCV(LGBMClassifier(), parameters)

    import re
    data = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    clf.fit(data, y)

    params = clf.cv_results_['params'][list(clf.cv_results_['rank_test_score']).index(1)]

    log.info('Building the model')

    lgbm = LGBMClassifier()
    lgbm.set_params(**params)

    log.info('Performing cross-validation')

    from sklearn.model_selection import cross_validate

    lgbmScores = cross_validate(lgbm, data, y, cv=5, scoring=['accuracy', 'f1', 'roc_auc'])
    return lgbmScores

def buildLRModel(X, y):
    log.info('Performing Hyperparameter optimisation')

    from sklearn.linear_model import LogisticRegression

    from sklearn.model_selection import GridSearchCV

    parameters={
        'solver': ['newton-cg', 'liblinear'],
        'C': [100, 10, 1.0, 0.1, 0.01],
    }

    clf = GridSearchCV(LogisticRegression(), parameters)
    clf.fit(X, y)

    params = clf.cv_results_['params'][list(clf.cv_results_['rank_test_score']).index(1)]

    log.info('Building the model')

    lr = LogisticRegression()
    lr.set_params(**params)

    log.info('Performing cross-validation')

    from sklearn.model_selection import cross_validate

    lrScores = cross_validate(lr, X, y, cv=5, scoring=['accuracy', 'f1', 'roc_auc'])

    return lrScores


def getOutputProbabilities(XVitalsMax, XVitalsMin, XVitalsAvg, XVitalsSd, XVitalsFirst, XVitalsLast, XLabsMax, XLabsMin, XLabsAvg, XLabsSd, XLabsFirst, XLabsLast, y):
    log.info('Split data to test and train sets')
    from sklearn.model_selection import train_test_split

    XVitalsMaxTrain, XVitalsMaxTest, XVitalsMinTrain, XVitalsMinTest, XVitalsAvgTrain, XVitalsAvgTest, XVitalsSdTrain, XVitalsSdTest, XVitalsFirstTrain, XVitalsFirstTest, XVitalsLastTrain, XVitalsLastTest, XLabsMaxTrain, XLabsMaxTest, XLabsMinTrain, XLabsMinTest,XLabsAvgTrain, XLabsAvgTest, XLabsSdTrain, XLabsSdTest, XLabsFirstTrain, XLabsFirstTest, XLabsLastTrain, XLabsLastTest, yTrain, yTest = train_test_split(
        XVitalsMax,
        XVitalsMin,
        XVitalsAvg,
        XVitalsSd,
        XVitalsFirst,
        XVitalsLast,
        XLabsMax,
        XLabsMin,
        XLabsAvg,
        XLabsSd,
        XLabsFirst,
        XLabsLast,
        y,
        test_size=0.2,
        random_state=42
        )

    log.info('Performing Hyperparameter optimisation for XGBoost')

    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from lightgbm import LGBMClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import GridSearchCV

    xgbParameters={
        'max_depth': [6, 9, 12],
        'scale_pos_weight': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    }

    xgbGrid = GridSearchCV(XGBClassifier(use_label_encoder=False, verbosity=0), xgbParameters)
    xgbGrid.fit(XVitalsMax, y)

    xgbParams = xgbGrid.cv_results_['params'][list(xgbGrid.cv_results_['rank_test_score']).index(1)]

    log.info('Performing Hyperparameter optimisation for Logistic Regression')

    lrParameters={
        'solver': ['newton-cg', 'liblinear'],
        'C': [100, 10, 1.0, 0.1, 0.01],
    }

    lrGrid = GridSearchCV(LogisticRegression(), lrParameters)
    lrGrid.fit(XVitalsMax, y)

    lrParams = lrGrid.cv_results_['params'][list(lrGrid.cv_results_['rank_test_score']).index(1)]

    XDict = {
        'VitalsMax': (XVitalsMaxTrain, XVitalsMaxTest),
        'VitalsMin': (XVitalsMinTrain, XVitalsMinTest),
        'VitalsAvg': (XVitalsAvgTrain, XVitalsAvgTest),
        'VitalsSd': (XVitalsSdTrain, XVitalsSdTest),
        'VitalsFirst': (XVitalsFirstTrain, XVitalsFirstTest),
        'VitalsLast': (XVitalsLastTrain, XVitalsLastTest),
        'LabsMax': (XLabsMaxTrain, XLabsMaxTest),
        'LabsMin': (XLabsMinTrain, XLabsMinTest),
        'LabsAvg': (XLabsAvgTrain, XLabsAvgTest),
        'LabsSd': (XLabsSdTrain, XLabsSdTest),
        'LabsFirst': (XLabsFirstTrain, XLabsFirstTest),
        'LabsLast': (XLabsLastTrain, XLabsLastTest),
    }

    probsDict = {}

    for label, (XTrain, XTest) in XDict.items():

        xgb = XGBClassifier(use_label_encoder=False)
        xgb.set_params(**xgbParams)
        xgb.fit(XTrain, yTrain)

        xgbProbs = [p for _, p in xgb.predict_proba(XTest)]

        probsDict[('XGB', label)] = xgbProbs

        lr = LogisticRegression()
        lr.set_params(**lrParams)
        lr.fit(XTrain, yTrain)

        lrProbs = [p2 for p1, p2 in lr.predict_proba(XTest)]

        probsDict[('LR', label)] = lrProbs

        lgbm = LGBMClassifier()
        lgbm.set_params(**xgbParams)
        lgbm.fit(XTrain, yTrain)

        lgbmProbs = [p2 for p1, p2 in lgbm.predict_proba(XTest)]

        probsDict[('LGBM', label)] = lgbmProbs

        mlp = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes = (200, 200))
        mlp.fit(XTrain, yTrain)

        mlpProbs = [p2 for p1, p2 in mlp.predict_proba(XTest)]

        probsDict[('MLP', label)] = mlpProbs

    import pandas as pd

    Xnew = pd.DataFrame()

    for key, value in probsDict.items():
        Xnew[key[0] + '_' + key[1]] = value

    return Xnew


def buildXGBoostModel(X, y):
    log.info('Performing Hyperparameter optimisation')

    from xgboost import XGBClassifier

    from sklearn.model_selection import GridSearchCV

    parameters={
        'max_depth': [6, 9, 12],
        'scale_pos_weight': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    }

    clf = GridSearchCV(XGBClassifier(use_label_encoder=False), parameters)
    clf.fit(X, y)

    params = clf.cv_results_['params'][list(clf.cv_results_['rank_test_score']).index(1)]

    log.info('Building the model')

    xgb = XGBClassifier(use_label_encoder=False)
    xgb.set_params(**params)

    log.info('Performing cross-validation')

    from sklearn.model_selection import cross_validate

    xgbScores = cross_validate(xgb, X, y, cv=5, scoring=['accuracy', 'f1', 'roc_auc'])

    return xgbScores


def saveCvScores(scores_dict, target, currentDir):

    import pickle
    from pathlib import Path

    labels = []
    accuracy_scores = []
    f1_scores = []
    roc_auc_scores = []

    for label, scores in scores_dict.items():
        labels.append(label)
        for key, value in scores.items():
            if key == 'test_accuracy':
                accuracy_scores.append(value)
            if key == 'test_f1':
                f1_scores.append(value)
            if key == 'test_roc_auc':
                roc_auc_scores.append(value)

    cvScoresPath = Path(currentDir, 'cv_scores_' + target + '.pickle')
    with open(cvScoresPath, 'wb') as fp:
        pickle.dump(scores_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


def runPipeline(target):
    log.info('Reading data')

    X, XVitalsMax, XVitalsMin, XVitalsAvg, XVitalsSd, XVitalsFirst, XVitalsLast, XLabsMax, XLabsMin, XLabsAvg, XLabsSd, XLabsFirst, XLabsLast, y = readData(target)

    import os
    import time
    import pandas as pd
    from pathlib import Path

    timestr = time.strftime("%Y%m%d%H%M%S")

    dataDir = Path('./', 'data')
    currentDir = Path(dataDir, timestr)
    if not os.path.exists(currentDir):
        os.makedirs(currentDir)

    XMinPath = Path(dataDir, 'X_min.csv')

    if XMinPath.exists():
        log.info('Reading SFS data from file')
        XMin = pd.read_csv(XMinPath)
        XMin = XMin.loc[:, ~XMin.columns.str.startswith('Unnamed')]
    else:
        XMin = performSfs(X, y)
        XMin.to_csv(XMinPath)

    from sklearn.model_selection import train_test_split

    XTrain, XTest, XMinTrain, XMinTest, yTrain, yTest = train_test_split(X, XMin, y, test_size=0.2, random_state=42)

    log.info('Building XGBoost model with all the features')

    xgbScores = buildXGBoostModel(XTrain, yTrain)

    log.info('Building XGBoost model with the selected features')

    xgbMinScores = buildXGBoostModel(XMinTrain, yTrain)

    log.info('Building LR model with all the features')

    lrScores = buildLRModel(XTrain, yTrain)

    log.info('Building LR model with the selected features')

    lrMinScores = buildLRModel(XMinTrain, yTrain)

    log.info('Building LGBM model with all the features')

    lgbmScores = buildLGBMModel(XTrain, yTrain)

    log.info('Building LGBM model with the selected features')

    lgbmMinScores = buildLGBMModel(XMinTrain, yTrain)

    log.info('Building MLP model with all the features')

    mlpScores = buildMLPModel(XTrain, yTrain, 200)

    log.info('Building MLP model with the selected features')

    mlpMinScores = buildMLPModel(XMinTrain, yTrain, 50)

    log.info('Get Outputs from first level models')

    dataDir = Path('./', 'data')
    XNewPath = Path(dataDir, 'X_new.csv')

    # if XNewPath.exists():
    #     log.info('Reading output probabilities data from file')
    #     Xnew = pd.read_csv(XNewPath)
    #     Xnew = Xnew.loc[:, ~Xnew.columns.str.startswith('Unnamed')]
    # else:
    log.info('Obtaining output probabilities')
    Xnew = getOutputProbabilities(XVitalsMax, XVitalsMin, XVitalsAvg, XVitalsSd, XVitalsFirst, XVitalsLast, XLabsMax, XLabsMin, XLabsAvg, XLabsSd, XLabsFirst, XLabsLast, y)
        # Xnew.to_csv(XNewPath)

    from sklearn.model_selection import train_test_split

    XNewTrain, XNewTest, yTestTrain, yTestTest = train_test_split(Xnew, yTest, test_size=0.2, random_state=42)

    log.info('Building Ensemble XGBoost model with all the features')

    xgbEnsembleNewScores = buildXGBoostModel(Xnew, yTest)

    log.info('Building Ensemble LR model with all the features')

    lrEnsembleScores = buildLRModel(Xnew, yTest)

    scores_dict = {
        'xgb': xgbScores,
        'lr': lrScores,
        'lgbm': lgbmScores,
        'mlp': mlpScores,
        'xgb_min': xgbMinScores,
        'lr_min': lrMinScores,
        'lgbm_min': lgbmMinScores,
        'mlp_min': mlpMinScores,
        'xgb_ensemble': xgbEnsembleNewScores,
        'lr_ensemble': lrEnsembleScores,
        }

    print("Scores: ", scores_dict)

    log.info('Saving the CV results for all the models')

    saveCvScores(scores_dict = scores_dict, target = target, currentDir = currentDir)

    log.info('Completed !!!')

if __name__ == '__main__':
    runPipeline('seven_day')
    runPipeline('fourteen_day')
    runPipeline('twentyone_day')
    runPipeline('twentyeight_day')
    runPipeline('sixty_day')
    runPipeline('ninety_day')
    runPipeline('onetwenty_day')
