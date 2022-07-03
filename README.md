# Mortality Prediction

## Pipeline

### How to run

In the pipeline.py, run the python class by changing the required arguments.

The possible options are;

1. ex.extractEhr(con, vitalsBefore = 24, vitalsAfter = i, labsBefore = 24, labsAfter = i, anchor="admission")

This method can be invoked to extract the data from OMOP schema form the provided DB. It creates separate tables for statics, vitals, labs, mortality ...

3. pa.formatAll(con, vitalsBefore = 24, vitalsAfter = i, labsBefore = 24, labsAfter = i, anchor="admission")

This method can be invoked to create the data matrix from the extracted values from the function before.

4. pm.runPredictionsForAllTargets(
                con,
                vitalsBefore = 24,
                vitalsAfter = i,
                labsBefore = 24,
                labsAfter = i,
                targetList = [7, 14, 21, 30, 60, 90, 120, (7, 14), (14, 21), (21, 30), (30, 60), (60, 90), (90, 120), (60, 120)],
                anchor="admission"
                )

This method can be invoked to perform machine learning on the data from the data matrix obtained from the above function.

In all the above functions;
* con - connection object to the postgres DB (use getConnection method in the same class)
* vitalsBefore - Duration (in hrs) of vitals to be considered before time zero
* vitalsAfter - Duration (in hrs) of vitals to be considered after time zero
* labsBefore - Duration (in hrs) of labs to be considered before time zero
* labsAfter - Duration (in hrs) of labs to be considered after time zero
* anchor - select the time-zero (possible options are: "admission" - admission time, "micro" - microbiology positive result)
