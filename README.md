# Mortality Prediction

## Pipeline

### How to run

In the pipeline.py, run the python class by changing the required arguments.

The possible options are;

1. ex.extractEhr(con, vitalsBefore = 24, vitalsAfter = i, labsBefore = 24, labsAfter = i, anchor="admission")
2. pa.formatAll(con, vitalsBefore = 24, vitalsAfter = i, labsBefore = 24, labsAfter = i, anchor="admission")
3. pm.runPredictionsForAllTargets(
                con,
                vitalsBefore = 24,
                vitalsAfter = i,
                labsBefore = 24,
                labsAfter = i,
                targetList = [7, 14, 21, 30, 60, 90, 120, (7, 14), (14, 21), (21, 30), (30, 60), (60, 90), (90, 120), (60, 120)],
                anchor="admission"
                )

In all the above functions;
* con - connection object to the postgres DB (use getConnection method in the same class)
* vitalsBefore - Duration (in hrs) of vitals to be considered before time zero
* vitalsAfter - Duration (in hrs) of vitals to be considered after time zero
* labsBefore - Duration (in hrs) of labs to be considered before time zero
* labsAfter - Duration (in hrs) of labs to be considered after time zero
* anchor - select the time-zero (possible options are: "admission" - admission time, "micro" - microbiology positive result)
