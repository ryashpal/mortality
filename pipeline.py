import logging
import sys

log = logging.getLogger("Pipeline")
log.setLevel(logging.INFO)
format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
log.addHandler(ch)

import mortality_prediction.extract_omop as ex
import mortality_prediction.preprocess_agg as pa
import mortality_prediction.predict_mortality as pm


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

    from pathlib import Path
    import os

    log.info("Start")
    # for i in [3, 6, 12, 24, 48, 72, 96, 120, 144]:
    for i in [48, 72, 96, 120, 144]:
        con = getConnnection()
        log.info('Run started for i = ' + str(i))
        try:
            ex.extractEhr(con, vitalsBefore = 24, vitalsAfter = i, labsBefore = 24, labsAfter = i, anchor="admission")
            pa.formatAll(con, vitalsBefore = 24, vitalsAfter = i, labsBefore = 24, labsAfter = i, anchor="admission")
            pm.runPredictionsForAllTargets(
                con,
                vitalsBefore = 24,
                vitalsAfter = i,
                labsBefore = 24,
                labsAfter = i,
                targetList = [7, 14, 21, 30, 60, 90, 120, (7, 14), (14, 21), (21, 30), (30, 60), (60, 90), (90, 120), (60, 120)],
                anchor="admission"
                )
        except Exception as e:
            log.error(e)
            log.error('Run failed for i = ' + str(i))
        finally:
            log.info('Closing connection')
            con.close()
            log.info("Deleting X_min.csv")
            dataDir = Path('./', 'data')
            XMinPath = Path(dataDir, 'X_min.csv')
            if XMinPath.exists():
                os.remove(XMinPath)


    # ex.extractEhr(vitalsBefore = 3, vitalsAfter = 0, labsBefore = 3, labsAfter = 0)
    # pa.formatAll(vitalsBefore = 3, vitalsAfter = 0, labsBefore = 3, labsAfter = 0)
    # pm.runPredictionsForAllTargets(
    #     vitalsBefore = 3,
    #     vitalsAfter = 0,
    #     labsBefore = 3,
    #     labsAfter = 0,
    #     targetList = [7, 14, 21, 30, 60, 90, 120, (7, 14), (14, 21), (21, 30), (30, 60), (60, 90), (90, 120), (60, 120)]
    #     )
    # ex.extractEhr(vitalsBefore = 120, vitalsAfter = 0, labsBefore = 120, labsAfter = 0)
    # pa.formatAll(vitalsBefore = 120, vitalsAfter = 0, labsBefore = 120, labsAfter = 0)
    # pm.runPredictionsForAllTargets(
    #     vitalsBefore = 120,
    #     vitalsAfter = 0,
    #     labsBefore = 120,
    #     labsAfter = 0,
    #     targetList = [7, 14, 21, 30, 60, 90, 120, (7, 14), (14, 21), (21, 30), (30, 60), (60, 90), (90, 120), (60, 120)]
    #     )
    # pm.runPredictionsForAllTargets(
    #     vitalsBefore = 48,
    #     vitalsAfter = 48,
    #     labsBefore = 72,
    #     labsAfter = 72,
    #     targetList = [7, 14, 21, 30, 60, 90, 120, (7, 14), (14, 21), (21, 30), (30, 60), (60, 90), (90, 120), (60, 120)]
    #     )
    # pm.runPredictionsForAllTargets(
    #     vitalsBefore = 24,
    #     vitalsAfter = 0,
    #     labsBefore = 24,
    #     labsAfter = 0,
    #     targetList = [7, 14, 21, 30, 60, 90, 120, (7, 14), (14, 21), (21, 30), (30, 60), (60, 90), (90, 120), (60, 120)]
    #     )
    # pm.runPredictionsForAllTargets(
    #     vitalsBefore = 48,
    #     vitalsAfter = 0,
    #     labsBefore = 48,
    #     labsAfter = 0,
    #     # targetList = [7, 14, 21, 30, 60, 90, 120, (7, 14), (14, 21), (21, 30), (30, 60), (60, 90), (90, 120)]
    #     targetList = [7, 14, 21, 30, 60, 90, 120, (7, 14), (14, 21), (21, 30), (30, 60), (60, 90), (90, 120), (60, 120)]
    #     )
    # pm.runPredictionsForAllTargets(
    #     vitalsBefore = 72,
    #     vitalsAfter = 0,
    #     labsBefore = 72,
    #     labsAfter = 0,
    #     targetList = [7, 14, 21, 30, 60, 90, 120, (7, 14), (14, 21), (21, 30), (30, 60), (60, 90), (90, 120), (60, 120)]
    #     )
    log.info("End")

