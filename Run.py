import logging
import sys

log = logging.getLogger("Pipeline")
log.setLevel(logging.INFO)
format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
log.addHandler(ch)


import warnings
warnings.simplefilter(action='ignore', category=Warning)

# import mortality_prediction.sepsis_icd.extract_sepsis_icd as esi
# import mortality_prediction.Draft as dr
# import mortality_prediction.sepsis_icd.preprocess_agg as pa
import mortality_prediction.sepsis_icd.predict_mortality as pm


def getConnnection():

    log.info("Establishing DB connection")
    import psycopg2

    # information used to create a database connection
    sqluser = 'postgres'
    dbname = 'mimic4'
    hostname = 'localhost'
    port_number = 5434

    # Connect to postgres with a copy of the MIMIC-III database
    con = psycopg2.connect(dbname=dbname, user=sqluser, host=hostname, port=port_number, password='mysecretpassword')

    log.info("DB connection obtained successfully")

    return con


if __name__ == "__main__":
    con = getConnnection()
    # dr.importDemographics(con=con, schemaName='testing', filePath='data/sepsis_cohort/source/demographics.csv', fileSeparator=',')
    # dr.importVitals(con=con, schemaName='testing', filePath='data/sepsis_cohort/source/vitals_merged.csv', fileSeparator=',')
    # dr.importLabmeasurements(con=con, schemaName='testing', filePath='data/sepsis_cohort/source/lab_measurements_dense.csv', fileSeparator=',')
    # dr.extractDeaths(con=con, schemaName='testing')
    # dr.createDataMatrix(con=con, schemaName='testing')
    # esi.extractEhr(con=con, vitalsBefore=24, vitalsAfter=24, labsBefore=24, labsAfter=24)
    # pa.formatAll(con=con, vitalsBefore=24, vitalsAfter=24, labsBefore=24, labsAfter=24)
    pm.runPredictionsForAllTargets(
        con=con,
        vitalsBefore = 24,
        vitalsAfter = 24,
        labsBefore = 24,
        labsAfter = 24,
        # targetList = [7, 14, 21, 30, 60, 90, 120, (7, 14), (14, 21), (21, 30), (30, 60), (60, 90), (90, 120)]
        targetList = [30, 60, 90, 120, (7, 14), (14, 21), (21, 30), (30, 60), (60, 90), (90, 120)]
        )

