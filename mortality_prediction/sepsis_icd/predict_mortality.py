import logging

log = logging.getLogger("Pipeline")


def readData(dirPath, targetStart, targetEnd):

    import pandas as pd

    dataDf = pd.read_csv(dirPath + 'data_matrix.csv')

    dataDf.anchor_time = dataDf.anchor_time.apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))
    dataDf.death_datetime = dataDf.death_datetime.apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))

    dataDf['target'] = (dataDf['death_datetime'] > (dataDf['anchor_time'] + pd.Timedelta(days=targetStart))) & (dataDf['death_datetime'] < (dataDf['anchor_time'] + pd.Timedelta(days=targetEnd)))
    dataDf.target.fillna(value=False, inplace=True)

    log.info('Formatting data')

    dropCols = [
        'person_id',
        'age',
        'gender',
        'ethnicity_WHITE',
        'ethnicity_BLACK',
        'ethnicity_UNKNOWN',
        'ethnicity_OTHER',
        'ethnicity_HISPANIC',
        'ethnicity_ASIAN',
        'ethnicity_UNABLE_TO_OBTAIN',
        'ethnicity_AMERICAN_INDIAN',
        'anchor_time',
        'death_datetime',
        'target',
    ]

    vitalsCols = ['heartrate', 'sysbp', 'diabp', 'meanbp', 'resprate', 'tempc', 'spo2', 'gcseye', 'gcsverbal', 'gcsmotor']
    labsCols = ['chloride_serum', 'creatinine', 'sodium_serum', 'hemoglobin', 'platelet_count', 'urea_nitrogen', 'glucose_serum', 'bicarbonate', 'potassium_serum', 'anion_gap', 'leukocytes_blood_manual', 'hematocrit']

    X = dataDf.drop(dropCols, axis = 1)
    XVitalsMin = dataDf[[vitalCol + '_min' for vitalCol in vitalsCols if vitalCol + '_min' in dataDf.columns]]
    XVitalsMax = dataDf[[vitalCol + '_max' for vitalCol in vitalsCols if vitalCol + '_max' in dataDf.columns]]
    XVitalsAvg = dataDf[[vitalCol + '_avg' for vitalCol in vitalsCols if vitalCol + '_avg' in dataDf.columns]]
    XVitalsSd = dataDf[[vitalCol + '_stddev' for vitalCol in vitalsCols if vitalCol + '_stddev' in dataDf.columns]]
    XVitalsFirst = dataDf[[vitalCol + '_first' for vitalCol in vitalsCols if vitalCol + '_first' in dataDf.columns]]
    XVitalsLast = dataDf[[vitalCol + '_last' for vitalCol in vitalsCols if vitalCol + '_last' in dataDf.columns]]
    XLabsMax = dataDf[[labsCol + '_min' for labsCol in labsCols if labsCol + '_min' in dataDf.columns]]
    XLabsMin = dataDf[[labsCol + '_max' for labsCol in labsCols if labsCol + '_max' in dataDf.columns]]
    XLabsAvg = dataDf[[labsCol + '_avg' for labsCol in labsCols if labsCol + '_avg' in dataDf.columns]]
    XLabsSd = dataDf[[labsCol + '_stddev' for labsCol in labsCols if labsCol + '_stddev' in dataDf.columns]]
    XLabsFirst = dataDf[[labsCol + '_first' for labsCol in labsCols if labsCol + '_first' in dataDf.columns]]
    XLabsLast = dataDf[[labsCol + '_last' for labsCol in labsCols if labsCol + '_last' in dataDf.columns]]
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

    clf = GridSearchCV(LGBMClassifier(verbose=-1), parameters)

    import re
    data = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    clf.fit(data, y)

    params = clf.cv_results_['params'][list(clf.cv_results_['rank_test_score']).index(1)]

    log.info('Building the model')

    lgbm = LGBMClassifier(verbose=-1)
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
        test_size=0.5,
        random_state=42
        )

    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from lightgbm import LGBMClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import GridSearchCV

    log.info('Performing Hyperparameter optimisation for XGBoost')

    xgbParams = performXgbHyperparameterTuning(XVitalsMax, y)

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

        lgbm = LGBMClassifier(verbose=-1)
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


def getBestXgbHyperparameter(X, y, parameters):

    from xgboost import XGBClassifier

    from sklearn.model_selection import GridSearchCV

    params = {}

    log.info('Hyperparameter optimisation for: ' + str(parameters))

    clf = GridSearchCV(XGBClassifier(use_label_encoder=False), parameters)
    clf.fit(X, y)

    params = clf.cv_results_['params'][list(clf.cv_results_['rank_test_score']).index(1)]
    return(params)


def performXgbHyperparameterTuning(X, y):

    params = {}

    params.update(getBestXgbHyperparameter(X, y, {'max_depth' : range(1,10),'scale_pos_weight': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],}))

    params.update(getBestXgbHyperparameter(X, y, {'n_estimators':range(50,250,10)}))

    params.update(getBestXgbHyperparameter(X, y, {'min_child_weight':range(1,10)}))

    params.update(getBestXgbHyperparameter(X, y, {'gamma':[i/10. for i in range(0,5)]}))

    params.update(getBestXgbHyperparameter(X, y, {'subsample':[i/10.0 for i in range(1,10)],'colsample_bytree':[i/10.0 for i in range(1,10)]}))

    params.update(getBestXgbHyperparameter(X, y, {'reg_alpha':[0, 1e-5, 1e-3, 0.1, 10]}))

    log.info('params: ' + str(params))

    return params


def buildXGBoostModel(X, y):
    log.info('Performing Hyperparameter optimisation')

    from sklearn.metrics import make_scorer

    from xgboost import XGBClassifier

    log.info('Building the model')
    
    params = performXgbHyperparameterTuning(X, y)

    xgb = XGBClassifier(use_label_encoder=False)
    xgb.set_params(**params)

    log.info('Performing cross-validation')

    from sklearn.model_selection import cross_validate

    xgbScores = cross_validate(xgb, X, y, cv=5, scoring=['accuracy', 'balanced_accuracy',  'average_precision', 'f1', 'roc_auc'])
    xgbScores['test_mccf1_score'] = cross_validate(xgb, X, y, cv=5, scoring = make_scorer(calculateMccF1, greater_is_better=True))['test_score']

    return xgbScores


def saveCvScores(scores_dict, dirPath, dirName, targetStart, targetEnd):

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

    currentDir = Path(dirPath, dirName)
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


def runPredictions(dirPath, dirName, targetStart, targetEnd):
    log.info('Reading data')

    X, XVitalsMax, XVitalsMin, XVitalsAvg, XVitalsSd, XVitalsFirst, XVitalsLast, XLabsMax, XLabsMin, XLabsAvg, XLabsSd, XLabsFirst, XLabsLast, y = readData(dirPath=dirPath, targetStart=targetStart, targetEnd=targetEnd)

    XMin = performSfs(X, y)

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

    mlpScores = buildMLPModel(X, y, 150)

    log.info('Building MLP model with the selected features')

    mlpMinScores = buildMLPModel(XMin, y, 30)

    log.info('Get Outputs from first level models')

    log.info('Obtaining output probabilities')
    Xnew, yNew = getOutputProbabilities(XVitalsMax, XVitalsMin, XVitalsAvg, XVitalsSd, XVitalsFirst, XVitalsLast, XLabsMax, XLabsMin, XLabsAvg, XLabsSd, XLabsFirst, XLabsLast, y)

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

    saveCvScores(scores_dict = scores_dict, dirPath = dirPath, dirName = dirName, targetStart = targetStart, targetEnd = targetEnd)

    log.info('Completed !!!')


def runPredictionsForAllTargets(
    label="run",
    dirPath = "./",
    vitalsBefore = 48,
    vitalsAfter = 48,
    labsBefore = 72,
    labsAfter = 72,
    targetList = [7, 14, 21, 30, 60, 90, 120, (7, 14), (14, 21), (21, 30), (30, 60), (60, 90), (90, 120)],
    ):

    dirName = label + "_icd_vb_" + str(vitalsBefore) + "_va_" + str(vitalsAfter) + "_lb_" + str(labsBefore) + "_la_" + str(labsAfter)
    log.info("dirName: " + dirName)
    for target in targetList:
        if type(target) is tuple:
            targetStart = target[0]
            targetEnd = target[1]
        else:
            targetStart = 0
            targetEnd = target
        log.info("Running Predictions for vb_" + str(vitalsBefore) + "_va_" + str(vitalsAfter) + "_lb_" + str(labsBefore) + "_la_" + str(labsAfter) + ", targetStart : " + str(targetStart) + ", targetEnd : " + str(targetEnd))
        runPredictions(dirPath, dirName, targetStart, targetEnd)


if __name__ == '__main__':
    dirPath = '/superbugai-data/yash/chapter_1/workspace/EHRQC/data/icd_cohort_test/'
    runPredictionsForAllTargets(
        dirPath = dirPath,
        vitalsBefore = 24,
        vitalsAfter = 24,
        labsBefore = 24,
        labsAfter = 24,
        targetList = [7]
        # targetList = [7, 14, 21, 30, 60, 90, 120, (7, 14), (14, 21), (21, 30), (30, 60), (60, 90), (90, 120)]
    )
    # runPredictionsForAllTargets(
    #     vitalsBefore = 48,
    #     vitalsAfter = 48,
    #     labsBefore = 72,
    #     labsAfter = 72,
    #     targetList = [7, 14, 21, 30, 60, 90, 120, (7, 14), (14, 21), (21, 30), (30, 60), (60, 90), (90, 120)]
    #     )
