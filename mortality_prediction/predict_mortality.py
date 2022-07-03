import logging

log = logging.getLogger("Pipeline")


def readData(con, schemaName, targetStart, targetEnd):

    import pandas as pd

    dataQuery = """
        select
        dm.*,
        (dm.death_datetime > (co.chart_time + INTERVAL '""" + str(targetStart) + """ DAY')) and (dm.death_datetime < (co.chart_time + INTERVAL '""" + str(targetEnd) + """ DAY')) as target
        from
        """ + schemaName + """.data_matrix_qc dm
        inner join """ + schemaName + """.cohort co
        on dm.micro_specimen_id = co.micro_specimen_id and dm.person_id = co.person_id
        ;
    """
    dataDf = pd.read_sql_query(dataQuery, con)
    dataDf.target.fillna(value=False, inplace=True)

    log.info('Formatting data')

    dropCols = [
        'micro_specimen_id',
        'person_id',
        'visit_duration_hrs',
        'gender',
        'death_datetime',
        'target',
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
    y = dataDf["target"]

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

    from sklearn.metrics import make_scorer

    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes = (layerSize, layerSize))
    mlp.fit(X, y)

    log.info('Performing cross-validation')

    from sklearn.model_selection import cross_validate

    mlpScores = cross_validate(mlp, X, y, cv=5, scoring=['accuracy', 'balanced_accuracy', 'average_precision', 'f1', 'roc_auc'])
    mlpScores['test_mccf1_score'] = cross_validate(mlp, X, y, cv=5, scoring = make_scorer(calculateMccF1, greater_is_better=True))['test_score']
    return mlpScores


def buildLGBMModel(X, y):
    log.info('Performing Hyperparameter optimisation')

    from sklearn.metrics import make_scorer

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

    lgbmScores = cross_validate(lgbm, data, y, cv=5, scoring=['accuracy', 'balanced_accuracy',  'average_precision', 'f1', 'roc_auc'])
    lgbmScores['test_mccf1_score'] = cross_validate(lgbm, data, y, cv=5, scoring = make_scorer(calculateMccF1, greater_is_better=True))['test_score']
    return lgbmScores


def buildLRModel(X, y):
    log.info('Performing Hyperparameter optimisation')

    from sklearn.metrics import make_scorer

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

    lrScores = cross_validate(lr, X, y, cv=5, scoring=['accuracy', 'balanced_accuracy',  'average_precision', 'f1', 'roc_auc'])
    lrScores['test_mccf1_score'] = cross_validate(lr, X, y, cv=5, scoring = make_scorer(calculateMccF1, greater_is_better=True))['test_score']

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

    log.info('Building individual models')

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

    return Xnew, yTest


def buildXGBoostModel(X, y):
    log.info('Performing Hyperparameter optimisation')

    from sklearn.metrics import make_scorer

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

    xgbScores = cross_validate(xgb, X, y, cv=5, scoring=['accuracy', 'balanced_accuracy',  'average_precision', 'f1', 'roc_auc'])
    xgbScores['test_mccf1_score'] = cross_validate(xgb, X, y, cv=5, scoring = make_scorer(calculateMccF1, greater_is_better=True))['test_score']

    return xgbScores


def saveCvScores(scores_dict, schemaName, targetStart, targetEnd):

    import os
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

    dataDir = Path('./', 'data')
    currentDir = Path(dataDir, schemaName)
    if not os.path.exists(currentDir):
        os.makedirs(currentDir)
    cvScoresPath = Path(currentDir, 'cv_scores_ts_' + str(targetStart) + '_te_' + str(targetEnd) + '.pickle')
    with open(cvScoresPath, 'wb') as fp:
        pickle.dump(scores_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


def calculateMccF1(x, y):
    import sys
    import os

    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr

    # import R's "base" package
    mccf1 = importr('mccf1')


    old_stdout = sys.stdout # backup current stdout
    sys.stdout = open(os.devnull, "w")
    p = robjects.FloatVector(x)
    t = robjects.FloatVector(y)
    calculateMccf1 = robjects.r['mccf1']
    summary = robjects.r['summary']
    out = summary(calculateMccf1(t, p), 50)[0][0]
    sys.stdout = old_stdout # reset old stdout
    return out


def runPredictions(con, schemaName, targetStart, targetEnd):
    log.info('Reading data')

    X, XVitalsMax, XVitalsMin, XVitalsAvg, XVitalsSd, XVitalsFirst, XVitalsLast, XLabsMax, XLabsMin, XLabsAvg, XLabsSd, XLabsFirst, XLabsLast, y = readData(con, schemaName, targetStart, targetEnd)

    import os
    import time
    import pandas as pd
    from pathlib import Path

    dataDir = Path('./', 'data')

    XMinPath = Path(dataDir, 'X_min.csv')

    if XMinPath.exists():
        log.info('Reading SFS data from file')
        XMin = pd.read_csv(XMinPath)
        XMin = XMin.loc[:, ~XMin.columns.str.startswith('Unnamed')]
    else:
        XMin = performSfs(X, y)
        XMin.to_csv(XMinPath)

    from sklearn.model_selection import train_test_split

    log.info('Building XGBoost model with all the features')

    xgbScores = buildXGBoostModel(X, y)

    log.info('Building XGBoost model with the selected features')

    xgbMinScores = buildXGBoostModel(XMin, y)

    log.info('Building LR model with all the features')

    lrScores = buildLRModel(X, y)

    log.info('Building LR model with the selected features')

    lrMinScores = buildLRModel(XMin, y)

    log.info('Building LGBM model with all the features')

    lgbmScores = buildLGBMModel(X, y)

    log.info('Building LGBM model with the selected features')

    lgbmMinScores = buildLGBMModel(XMin, y)

    log.info('Building MLP model with all the features')

    mlpScores = buildMLPModel(X, y, 200)

    log.info('Building MLP model with the selected features')

    mlpMinScores = buildMLPModel(XMin, y, 50)

    log.info('Get Outputs from first level models')

    dataDir = Path('./', 'data')

    log.info('Obtaining output probabilities')
    Xnew, yNew = getOutputProbabilities(XVitalsMax, XVitalsMin, XVitalsAvg, XVitalsSd, XVitalsFirst, XVitalsLast, XLabsMax, XLabsMin, XLabsAvg, XLabsSd, XLabsFirst, XLabsLast, y)

    from sklearn.model_selection import train_test_split

    log.info('Building Ensemble XGBoost model with all the features')

    xgbEnsembleNewScores = buildXGBoostModel(Xnew, yNew)

    log.info('Building Ensemble LR model with all the features')

    lrEnsembleScores = buildLRModel(Xnew, yNew)

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

    saveCvScores(scores_dict = scores_dict, schemaName = schemaName, targetStart = targetStart, targetEnd = targetEnd)

    log.info('Completed !!!')


def runPredictionsForAllTargets(
    con,
    vitalsBefore = 48,
    vitalsAfter = 48,
    labsBefore = 72,
    labsAfter = 72,
    targetList = [7, 14, 21, 30, 60, 90, 120, (7, 14), (14, 21), (21, 30), (30, 60), (60, 90), (90, 120)],
    anchor = "micro"
    ):

    schemaName = anchor + "_vb_" + str(vitalsBefore) + "_va_" + str(vitalsAfter) + "_lb_" + str(labsBefore) + "_la_" + str(labsAfter)
    log.info("Schema Name: " + schemaName)
    for target in targetList:
        if type(target) is tuple:
            targetStart = target[0]
            targetEnd = target[1]
        else:
            targetStart = 0
            targetEnd = target
        log.info("Running Predictions for schemaName : " + schemaName + ", targetStart : " + str(targetStart) + ", targetEnd : " + str(targetEnd))
        runPredictions(con, schemaName, targetStart, targetEnd)


# if __name__ == '__main__':
    # runPredictionsForAllTargets(
    #     vitalsBefore = 48,
    #     vitalsAfter = 48,
    #     labsBefore = 72,
    #     labsAfter = 72,
    #     targetList = [7, 14, 21, 30, 60, 90, 120, (7, 14), (14, 21), (21, 30), (30, 60), (60, 90), (90, 120)]
    #     )
    # runPredictionsForAllTargets(
    #     vitalsBefore = 48,
    #     vitalsAfter = 48,
    #     labsBefore = 72,
    #     labsAfter = 72,
    #     targetList = [7]
    #     )
    # runPredictionsForAllTargets(
    #     vitalsBefore = 24,
    #     vitalsAfter = 0,
    #     labsBefore = 24,
    #     labsAfter = 0,
    #     targetList = [7, 14, 21, 30, 60, 90, 120, (7, 14), (14, 21), (21, 30), (30, 60), (60, 90), (90, 120)]
    #     )
    # runPredictionsForAllTargets(
    #     vitalsBefore = 48,
    #     vitalsAfter = 0,
    #     labsBefore = 48,
    #     labsAfter = 0,
    #     targetList = [7, 14, 21, 30, 60, 90, 120, (7, 14), (14, 21), (21, 30), (30, 60), (60, 90), (90, 120)]
    #     )
    # runPredictionsForAllTargets(
    #     vitalsBefore = 72,
    #     vitalsAfter = 0,
    #     labsBefore = 72,
    #     labsAfter = 0,
    #     targetList = [7, 14, 21, 30, 60, 90, 120, (7, 14), (14, 21), (21, 30), (30, 60), (60, 90), (90, 120)]
    #     )
