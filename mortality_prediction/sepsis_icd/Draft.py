import logging

log = logging.getLogger("Mortality Prediction")

import pandas as pd
import numpy as np


def __saveDataframe(con, schemaName, tableName, df, dfColumns):

    import numpy as np
    import psycopg2.extras
    import psycopg2.extensions

    psycopg2.extensions.register_adapter(np.int64, psycopg2._psycopg.AsIs)

    log.info("Importing data to table: " + schemaName + '.' + tableName)

    if len(df) > 0:
        table = schemaName + '.' + tableName
        columns = '"' + '", "'.join(dfColumns) + '"'
        values = "VALUES({})".format(",".join(["%s" for _ in dfColumns]))
        insert_stmt = "INSERT INTO {} ({}) {}".format(table, columns, values)
        try:
            cur = con.cursor()
            psycopg2.extras.execute_batch(cur, insert_stmt, df[dfColumns].values)
            con.commit()
        finally:
            cur.close()


def importDemographics(con, schemaName, filePath, fileSeparator):

    log.info("Creating table: " + schemaName + ".DEMOGRAPHICS")

    dropQuery = """DROP TABLE IF EXISTS """ + schemaName + """.DEMOGRAPHICS CASCADE"""
    createQuery = """CREATE TABLE """ + schemaName + """.DEMOGRAPHICS
        (
            person_id INT NOT NULL,
            gender VARCHAR(1) NOT NULL,
            ethnicity VARCHAR(50) NOT NULL,
            dob TIMESTAMP(0) NOT NULL,
            dod TIMESTAMP(0) -- This is a NaN column
        )
        ;
        """
    with con:
        with con.cursor() as cursor:
            cursor.execute(dropQuery)
            cursor.execute(createQuery)

    df = pd.read_csv(filePath, sep=fileSeparator)
    df = df.replace({np.NaN: None})

    log.info("Successfully created table: " + schemaName + ".DEMOGRAPHICS")

    log.info("Saving data to table: " + schemaName + ".DEMOGRAPHICS")
    __saveDataframe(con=con, schemaName=schemaName, tableName='DEMOGRAPHICS', df=df, dfColumns=['person_id', 'gender', 'ethnicity', 'dob', 'dod'])
    log.info("Successfully saved data to table: " + schemaName + ".DEMOGRAPHICS")


def importVitals(con, schemaName, filePath, fileSeparator):

    log.info("Creating table: " + schemaName + ".VITALS")

    dropQuery = """DROP TABLE IF EXISTS """ + schemaName + """.VITALS CASCADE"""
    createQuery = """CREATE TABLE """ + schemaName + """.VITALS
        (
            person_id INT NOT NULL,
            heartrate decimal NOT NULL,
            sysbp decimal NOT NULL,
            diabp decimal NOT NULL,
            meanbp decimal NOT NULL,
            resprate decimal NOT NULL,
            tempc decimal NOT NULL,
            spo2 decimal NOT NULL,
            gcseye decimal NOT NULL,
            gcsverbal decimal NOT NULL,
            gcsmotor decimal NOT NULL
        )
        ;
        """
    with con:
        with con.cursor() as cursor:
            cursor.execute(dropQuery)
            cursor.execute(createQuery)

    df = pd.read_csv(filePath, sep=fileSeparator)

    log.info("Successfully created table: " + schemaName + ".VITALS")

    log.info("Saving data to table: " + schemaName + ".VITALS")
    __saveDataframe(con=con, schemaName=schemaName, tableName='VITALS', df=df, dfColumns=df.columns)
    log.info("Successfully saved data to table: " + schemaName + ".VITALS")


def importLabmeasurements(con, schemaName, filePath, fileSeparator):

    log.info("Creating table: " + schemaName + ".LABS")
    dropQuery = """DROP TABLE IF EXISTS """ + schemaName + """.LABS CASCADE"""
    createQuery = """CREATE TABLE """ + schemaName + """.LABS
        (
            person_id INT NOT NULL,
            chloride_serum decimal NOT NULL,
            creatinine decimal NOT NULL,
            sodium_serum decimal NOT NULL,
            hemoglobin decimal NOT NULL,
            platelet_count decimal NOT NULL,
            urea_nitrogen decimal NOT NULL,
            glucose_serum decimal NOT NULL,
            bicarbonate decimal NOT NULL,
            potassium_serum decimal NOT NULL,
            anion_gap decimal NOT NULL,
            leukocytes_blood_manual decimal NOT NULL,
            hematocrit decimal NOT NULL
        )
        ;
        """
    with con:
        with con.cursor() as cursor:
            cursor.execute(dropQuery)
            cursor.execute(createQuery)

    df = pd.read_csv(filePath, sep=fileSeparator)

    log.info("Successfully created table: " + schemaName + ".LABS")

    log.info("Saving data to table: " + schemaName + ".LABS")
    __saveDataframe(con=con, schemaName=schemaName, tableName='LABS', df=df, dfColumns=df.columns)
    log.info("Successfully saved data to table: " + schemaName + ".LABS")


def extractDeaths(con, schemaName):

    log.info("Creating table: " + schemaName + ".DEATHS")
    dropQuery = """DROP TABLE IF EXISTS """ + schemaName + """.DEATHS CASCADE"""
    createQuery = """
        create table """ + schemaName + """.DEATHS as
        select
        person_id,
        (case when dod is null then 0 else 1 end) as outcome
        from
        """ + schemaName + """.DEMOGRAPHICS
        ;
        """
    with con:
        with con.cursor() as cursor:
            cursor.execute(dropQuery)
            cursor.execute(createQuery)


def createDataMatrix(con, schemaName):

    log.info("Creating table: " + schemaName + ".DATA_MATRIX")
    dropQuery = """DROP TABLE IF EXISTS """ + schemaName + """.DATA_MATRIX CASCADE"""
    createQuery = """
        create table """ + schemaName + """.DATA_MATRIX as
        select
        dem.person_id,
        (case when dem.gender='M' then 1 else 0 end) as gender,
        vit.heartrate,
        vit.sysbp,
        vit.diabp,
        vit.meanbp,
        vit.resprate,
        vit.tempc,
        vit.spo2,
        vit.gcseye,
        vit.gcsverbal,
        vit.gcsmotor,
        lab.chloride_serum,
        lab.creatinine,
        lab.sodium_serum,
        lab.hemoglobin,
        lab.platelet_count,
        lab.urea_nitrogen,
        lab.glucose_serum,
        lab.bicarbonate,
        lab.potassium_serum,
        lab.anion_gap,
        lab.leukocytes_blood_manual,
        lab.hematocrit,
        dth.outcome
        from
        """ + schemaName + """.demographics dem
        inner join """ + schemaName + """.vitals vit
        on vit.person_id = dem.person_id
        inner join """ + schemaName + """.labs lab
        on lab.person_id = dem.person_id
        inner join """ + schemaName + """.deaths dth
        on dth.person_id = dem.person_id
        """
    with con:
        with con.cursor() as cursor:
            cursor.execute(dropQuery)
            cursor.execute(createQuery)
