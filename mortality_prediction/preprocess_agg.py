import logging

log = logging.getLogger("Pipeline")

import pandas as pd


def rejectVitalsOutliers(df):
    data = df[df.value_as_number < df.value_as_number.quantile(0.9999)].value_as_number
    inx = df.value_as_number.sub(data.mean()).div(data.std()).abs().le(3)
    return df[inx]


def filterVitalsOutliers(con, schemaName):

    log.info("Reading vitals data from " + schemaName + ".vitals")
    vitalsQuery = """select * from """ + schemaName + """.vitals;"""
    vitalsDf = pd.read_sql_query(vitalsQuery, con)

    log.info("Removing outliers")
    filteredDf = pd.DataFrame(columns=vitalsDf.columns)
    concepts = vitalsDf.concept_name.unique()
    for concept in concepts:
        conceptDf = vitalsDf[vitalsDf.concept_name == concept]
        conceptDfFiltered = rejectVitalsOutliers(conceptDf)
        filteredDf = pd.concat([filteredDf, conceptDfFiltered])

    log.info("Creating table " + schemaName + ".vitals_filtered")
    dropTableQuery = """drop table if exists """ + schemaName + """.vitals_filtered cascade;"""
    createTableQuery = """create table """ + schemaName + """.vitals_filtered (like """ + schemaName + """.vitals including all)"""
    with con:
        with con.cursor() as cursor:
            cursor.execute(dropTableQuery)
            cursor.execute(createTableQuery)

    log.info("Loading data into the table: " + schemaName + ".vitals_filtered")

    import psycopg2.extras

    if len(filteredDf) > 0:
        table = schemaName + ".vitals_filtered"
        df_columns = list(filteredDf)
        columns = ",".join(df_columns)
        values = "VALUES({})".format(",".join(["%s" for _ in df_columns])) 
        insert_stmt = "INSERT INTO {} ({}) {}".format(table,columns,values)
        cur = con.cursor()
        psycopg2.extras.execute_batch(cur, insert_stmt, filteredDf.values)
        con.commit()
        cur.close()


def rejectLabsOutliers(df):
    data = df[df.value_as_number != 999999].value_as_number
    inx = df.value_as_number.sub(data.mean()).div(data.std()).abs().le(3)
    return df[inx]


def filterLabsOutliers(con, schemaName):

    log.info("Reading labs data from " + schemaName + ".labs")
    labsQuery = """select * from """ + schemaName + """.labs;"""
    labsDf = pd.read_sql_query(labsQuery, con)

    log.info("Removing outliers")
    filteredDf = pd.DataFrame(columns=labsDf.columns)
    concepts = labsDf.concept_name.unique()
    for concept in concepts:
        conceptDf = labsDf[labsDf.concept_name == concept]
        conceptDfFiltered = rejectLabsOutliers(conceptDf)
        filteredDf = pd.concat([filteredDf, conceptDfFiltered])

    log.info("Creating table " + schemaName + ".labs_filtered")
    dropTableQuery = """drop table if exists """ + schemaName + """.labs_filtered cascade;"""
    createTableQuery = """create table """ + schemaName + """.labs_filtered (like """ + schemaName + """.labs including all)"""
    with con:
        with con.cursor() as cursor:
            cursor.execute(dropTableQuery)
            cursor.execute(createTableQuery)

    log.info("Loading data into the table: " + schemaName + ".labs_filtered")

    import psycopg2.extras

    if len(filteredDf) > 0:
        table = schemaName + ".labs_filtered"
        df_columns = list(filteredDf)
        columns = ",".join(df_columns)
        values = "VALUES({})".format(",".join(["%s" for _ in df_columns])) 
        insert_stmt = "INSERT INTO {} ({}) {}".format(table,columns,values)
        cur = con.cursor()
        psycopg2.extras.execute_batch(cur, insert_stmt, filteredDf.values)
        con.commit()
        cur.close()


def formatVitalsStg1(con, schemaName):
    log.info("Formatting vitals to " + schemaName + ".vitals_stg_1")
    dropVitalsStg1Query = """drop table if exists """ + schemaName + """.vitals_stg_1 cascade"""
    dataVitals1Query = """
        create table """ + schemaName + """.vitals_stg_1 as
        select
        vit.micro_specimen_id as micro_specimen_id
        , vit.person_id as person_id
        , vit.rn as rn
        , COALESCE(MAX(CASE WHEN vit.concept_name = 'Body temperature' THEN vit.value_as_number ELSE null END)) AS tmp
        , COALESCE(MAX(CASE WHEN vit.concept_name = 'Heart rate' THEN vit.value_as_number ELSE null END)) AS heartrate
        , COALESCE(MAX(CASE WHEN vit.concept_name = 'Breath rate setting Ventilator' THEN vit.value_as_number ELSE null END)) AS breath_rate_vent
        , COALESCE(MAX(CASE WHEN vit.concept_name = 'Breath rate spontaneous' THEN vit.value_as_number ELSE null END)) AS breath_rate_spon
        , COALESCE(MAX(CASE WHEN vit.concept_name = 'Respiratory rate' THEN vit.value_as_number ELSE null END)) AS resp_rate
        , COALESCE(MAX(CASE WHEN vit.concept_name = 'Oxygen saturation in Arterial blood by Pulse oximetry' THEN vit.value_as_number ELSE null END)) AS oxygen
        , COALESCE(MAX(CASE WHEN vit.concept_name = 'Systolic blood pressure' THEN vit.value_as_number ELSE null END)) AS sysbp
        , COALESCE(MAX(CASE WHEN vit.concept_name = 'Diastolic blood pressure' THEN vit.value_as_number ELSE null END)) AS diabp
        , COALESCE(MAX(CASE WHEN vit.concept_name = 'Mean blood pressure' THEN vit.value_as_number ELSE null END)) AS meanbp
        , COALESCE(MAX(CASE WHEN vit.concept_name = 'Systolic blood pressure by Noninvasive' THEN vit.value_as_number ELSE null END)) AS sysbp_ni
        , COALESCE(MAX(CASE WHEN vit.concept_name = 'Diastolic blood pressure by Noninvasive' THEN vit.value_as_number ELSE null END)) AS diabp_ni
        , COALESCE(MAX(CASE WHEN vit.concept_name = 'Mean blood pressure by Noninvasive' THEN vit.value_as_number ELSE null END)) AS meanbp_ni
        , COALESCE(MAX(CASE WHEN vit.concept_name = 'Glasgow coma score motor' THEN vit.value_as_number ELSE null END)) AS gcs_motor
        , COALESCE(MAX(CASE WHEN vit.concept_name = 'Glasgow coma score verbal' THEN vit.value_as_number ELSE null END)) AS gcs_verbal
        , COALESCE(MAX(CASE WHEN vit.concept_name = 'Glasgow coma score eye opening' THEN vit.value_as_number ELSE null END)) AS gcs_eye
        from
        """ + schemaName + """.vitals_filtered vit
        group by vit.micro_specimen_id, vit.person_id, vit.rn
        ;
        """

    with con:
        with con.cursor() as cursor:
            cursor.execute(dropVitalsStg1Query)
            cursor.execute(dataVitals1Query)


def formatVitalsMax(con, schemaName):
    log.info("Formatting vitals_stg_1 to " + schemaName + ".vitals_max")
    dropVitalsMaxQuery = """drop table if exists """ + schemaName + """.vitals_max cascade"""
    dataVitalsMaxQuery = """
        create table """ + schemaName + """.vitals_max as
        select
        distinct
        vit.micro_specimen_id as micro_specimen_id
        , vit.person_id as person_id
        , max(vit.tmp) over (partition by vit.micro_specimen_id, vit.person_id) as temp_max
        , max(vit.heartrate) over (partition by vit.micro_specimen_id, vit.person_id) as heartrate_max
        , max(vit.breath_rate_vent) over (partition by vit.micro_specimen_id, vit.person_id) as breath_rate_vent_max
        , max(vit.breath_rate_spon) over (partition by vit.micro_specimen_id, vit.person_id) as breath_rate_spon_max
        , max(vit.resp_rate) over (partition by vit.micro_specimen_id, vit.person_id) as resp_rate_max
        , max(vit.oxygen) over (partition by vit.micro_specimen_id, vit.person_id) as oxygen_max
        , max(vit.sysbp) over (partition by vit.micro_specimen_id, vit.person_id) as sysbp_max
        , max(vit.diabp) over (partition by vit.micro_specimen_id, vit.person_id) as diabp_max
        , max(vit.meanbp) over (partition by vit.micro_specimen_id, vit.person_id) as meanbp_max
        , max(vit.sysbp_ni) over (partition by vit.micro_specimen_id, vit.person_id) as sysbp_ni_max
        , max(vit.diabp_ni) over (partition by vit.micro_specimen_id, vit.person_id) as diabp_ni_max
        , max(vit.meanbp_ni) over (partition by vit.micro_specimen_id, vit.person_id) as meanbp_ni_max
        , max(vit.gcs_motor) over (partition by vit.micro_specimen_id, vit.person_id) as gcs_motor_max
        , max(vit.gcs_verbal) over (partition by vit.micro_specimen_id, vit.person_id) as gcs_verbal_max
        , max(vit.gcs_eye) over (partition by vit.micro_specimen_id, vit.person_id) as gcs_eye_max
        from """ + schemaName + """.vitals_stg_1 vit
        ;
    """

    with con:
        with con.cursor() as cursor:
            cursor.execute(dropVitalsMaxQuery)
            cursor.execute(dataVitalsMaxQuery)


def formatVitalsMin(con, schemaName):
    log.info("Formatting vitals_stg_1 to " + schemaName + ".vitals_min")
    dropVitalsMinQuery = """drop table if exists """ + schemaName + """.vitals_min cascade"""
    dataVitalsMinQuery = """
        create table """ + schemaName + """.vitals_min as
        select
        distinct
        vit.micro_specimen_id as micro_specimen_id
        , vit.person_id as person_id
        , min(vit.tmp) over (partition by vit.micro_specimen_id, vit.person_id) as temp_min
        , min(vit.heartrate) over (partition by vit.micro_specimen_id, vit.person_id) as heartrate_min
        , min(vit.breath_rate_vent) over (partition by vit.micro_specimen_id, vit.person_id) as breath_rate_vent_min
        , min(vit.breath_rate_spon) over (partition by vit.micro_specimen_id, vit.person_id) as breath_rate_spon_min
        , min(vit.resp_rate) over (partition by vit.micro_specimen_id, vit.person_id) as resp_rate_min
        , min(vit.oxygen) over (partition by vit.micro_specimen_id, vit.person_id) as oxygen_min
        , min(vit.sysbp) over (partition by vit.micro_specimen_id, vit.person_id) as sysbp_min
        , min(vit.diabp) over (partition by vit.micro_specimen_id, vit.person_id) as diabp_min
        , min(vit.meanbp) over (partition by vit.micro_specimen_id, vit.person_id) as meanbp_min
        , min(vit.sysbp_ni) over (partition by vit.micro_specimen_id, vit.person_id) as sysbp_ni_min
        , min(vit.diabp_ni) over (partition by vit.micro_specimen_id, vit.person_id) as diabp_ni_min
        , min(vit.meanbp_ni) over (partition by vit.micro_specimen_id, vit.person_id) as meanbp_ni_min
        , min(vit.gcs_motor) over (partition by vit.micro_specimen_id, vit.person_id) as gcs_motor_min
        , min(vit.gcs_verbal) over (partition by vit.micro_specimen_id, vit.person_id) as gcs_verbal_min
        , min(vit.gcs_eye) over (partition by vit.micro_specimen_id, vit.person_id) as gcs_eye_min
        from """ + schemaName + """.vitals_stg_1 vit
        ;
    """

    with con:
        with con.cursor() as cursor:
            cursor.execute(dropVitalsMinQuery)
            cursor.execute(dataVitalsMinQuery)


def formatVitalsAvg(con, schemaName):
    log.info("Formatting vitals_stg_1 to " + schemaName + ".vitals_avg")
    dropVitalsAvgQuery = """drop table if exists """ + schemaName + """.vitals_avg cascade"""
    dataVitalsAvgQuery = """
        create table """ + schemaName + """.vitals_avg as
        select
        distinct
        vit.micro_specimen_id as micro_specimen_id
        , vit.person_id as person_id
        , avg(vit.tmp) over (partition by vit.micro_specimen_id, vit.person_id) as temp_avg
        , avg(vit.heartrate) over (partition by vit.micro_specimen_id, vit.person_id) as heartrate_avg
        , avg(vit.breath_rate_vent) over (partition by vit.micro_specimen_id, vit.person_id) as breath_rate_vent_avg
        , avg(vit.breath_rate_spon) over (partition by vit.micro_specimen_id, vit.person_id) as breath_rate_spon_avg
        , avg(vit.resp_rate) over (partition by vit.micro_specimen_id, vit.person_id) as resp_rate_avg
        , avg(vit.oxygen) over (partition by vit.micro_specimen_id, vit.person_id) as oxygen_avg
        , avg(vit.sysbp) over (partition by vit.micro_specimen_id, vit.person_id) as sysbp_avg
        , avg(vit.diabp) over (partition by vit.micro_specimen_id, vit.person_id) as diabp_avg
        , avg(vit.meanbp) over (partition by vit.micro_specimen_id, vit.person_id) as meanbp_avg
        , avg(vit.sysbp_ni) over (partition by vit.micro_specimen_id, vit.person_id) as sysbp_ni_avg
        , avg(vit.diabp_ni) over (partition by vit.micro_specimen_id, vit.person_id) as diabp_ni_avg
        , avg(vit.meanbp_ni) over (partition by vit.micro_specimen_id, vit.person_id) as meanbp_ni_avg
        , avg(vit.gcs_motor) over (partition by vit.micro_specimen_id, vit.person_id) as gcs_motor_avg
        , avg(vit.gcs_verbal) over (partition by vit.micro_specimen_id, vit.person_id) as gcs_verbal_avg
        , avg(vit.gcs_eye) over (partition by vit.micro_specimen_id, vit.person_id) as gcs_eye_avg
        from """ + schemaName + """.vitals_stg_1 vit
        ;
    """

    with con:
        with con.cursor() as cursor:
            cursor.execute(dropVitalsAvgQuery)
            cursor.execute(dataVitalsAvgQuery)


def formatVitalsStd(con, schemaName):
    log.info("Formatting vitals_stg_1 to " + schemaName + ".vitals_std")
    dropVitalsStdQuery = """drop table if exists """ + schemaName + """.vitals_std cascade"""
    dataVitalsStdQuery = """
        create table """ + schemaName + """.vitals_std as
        select
        distinct
        vit.micro_specimen_id as micro_specimen_id
        , vit.person_id as person_id
        , stddev(vit.tmp) over (partition by vit.micro_specimen_id, vit.person_id) as temp_sd
        , stddev(vit.heartrate) over (partition by vit.micro_specimen_id, vit.person_id) as heartrate_sd
        , stddev(vit.breath_rate_vent) over (partition by vit.micro_specimen_id, vit.person_id) as breath_rate_vent_sd
        , stddev(vit.breath_rate_spon) over (partition by vit.micro_specimen_id, vit.person_id) as breath_rate_spon_sd
        , stddev(vit.resp_rate) over (partition by vit.micro_specimen_id, vit.person_id) as resp_rate_sd
        , stddev(vit.oxygen) over (partition by vit.micro_specimen_id, vit.person_id) as oxygen_sd
        , stddev(vit.sysbp) over (partition by vit.micro_specimen_id, vit.person_id) as sysbp_sd
        , stddev(vit.diabp) over (partition by vit.micro_specimen_id, vit.person_id) as diabp_sd
        , stddev(vit.meanbp) over (partition by vit.micro_specimen_id, vit.person_id) as meanbp_sd
        , stddev(vit.sysbp_ni) over (partition by vit.micro_specimen_id, vit.person_id) as sysbp_ni_sd
        , stddev(vit.diabp_ni) over (partition by vit.micro_specimen_id, vit.person_id) as diabp_ni_sd
        , stddev(vit.meanbp_ni) over (partition by vit.micro_specimen_id, vit.person_id) as meanbp_ni_sd
        , stddev(vit.gcs_motor) over (partition by vit.micro_specimen_id, vit.person_id) as gcs_motor_sd
        , stddev(vit.gcs_verbal) over (partition by vit.micro_specimen_id, vit.person_id) as gcs_verbal_sd
        , stddev(vit.gcs_eye) over (partition by vit.micro_specimen_id, vit.person_id) as gcs_eye_sd
        from """ + schemaName + """.vitals_stg_1 vit
        ;
    """

    with con:
        with con.cursor() as cursor:
            cursor.execute(dropVitalsStdQuery)
            cursor.execute(dataVitalsStdQuery)


def formatVitalsFirst(con, schemaName):
    log.info("Formatting vitals_stg_1 to " + schemaName + ".vitals_first")
    dropVitalsFirstQuery = """drop table if exists """ + schemaName + """.vitals_first cascade"""
    dataVitalsFirstQuery = """
        create table """ + schemaName + """.vitals_first as
        select
        distinct
        vit.micro_specimen_id as micro_specimen_id
        , vit.person_id as person_id
        , first_value(vit.tmp) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.tmp is not null then 0 else 1 end asc, vit.rn asc) as temp_first
        , first_value(vit.heartrate) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.heartrate is not null then 0 else 1 end asc, vit.rn asc) as heartrate_first
        , first_value(vit.breath_rate_vent) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.breath_rate_vent is not null then 0 else 1 end asc, vit.rn asc) as breath_rate_vent_first
        , first_value(vit.breath_rate_spon) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.breath_rate_spon is not null then 0 else 1 end asc, vit.rn asc) as breath_rate_spon_first
        , first_value(vit.resp_rate) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.resp_rate is not null then 0 else 1 end asc, vit.rn asc) as resp_rate_first
        , first_value(vit.oxygen) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.oxygen is not null then 0 else 1 end asc, vit.rn asc) as oxygen_first
        , first_value(vit.sysbp) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.sysbp is not null then 0 else 1 end asc, vit.rn asc) as sysbp_first
        , first_value(vit.diabp) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.diabp is not null then 0 else 1 end asc, vit.rn asc) as diabp_first
        , first_value(vit.meanbp) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.meanbp is not null then 0 else 1 end asc, vit.rn asc) as meanbp_first
        , first_value(vit.sysbp_ni) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.sysbp_ni is not null then 0 else 1 end asc, vit.rn asc) as sysbp_ni_first
        , first_value(vit.diabp_ni) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.diabp_ni is not null then 0 else 1 end asc, vit.rn asc) as diabp_ni_first
        , first_value(vit.meanbp_ni) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.meanbp_ni is not null then 0 else 1 end asc, vit.rn asc) as meanbp_ni_first
        , first_value(vit.gcs_motor) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.gcs_motor is not null then 0 else 1 end asc, vit.rn asc) as gcs_motor_first
        , first_value(vit.gcs_verbal) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.gcs_verbal is not null then 0 else 1 end asc, vit.rn asc) as gcs_verbal_first
        , first_value(vit.gcs_eye) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.gcs_eye is not null then 0 else 1 end asc, vit.rn asc) as gcs_eye_first
        from """ + schemaName + """.vitals_stg_1 vit
        ;
    """

    with con:
        with con.cursor() as cursor:
            cursor.execute(dropVitalsFirstQuery)
            cursor.execute(dataVitalsFirstQuery)


def formatVitalsLast(con, schemaName):
    log.info("Formatting vitals_stg_1 to " + schemaName + ".vitals_last")
    dropVitalsLastQuery = """drop table if exists """ + schemaName + """.vitals_last cascade"""
    dataVitalsLastQuery = """
        create table """ + schemaName + """.vitals_last as
        select
        distinct
        vit.micro_specimen_id as micro_specimen_id
        , vit.person_id as person_id
        , first_value(vit.tmp) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.tmp is not null then 0 else 1 end asc, vit.rn desc) as temp_last
        , first_value(vit.heartrate) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.heartrate is not null then 0 else 1 end asc, vit.rn desc) as heartrate_last
        , first_value(vit.breath_rate_vent) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.breath_rate_vent is not null then 0 else 1 end asc, vit.rn desc) as breath_rate_vent_last
        , first_value(vit.breath_rate_spon) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.breath_rate_spon is not null then 0 else 1 end asc, vit.rn desc) as breath_rate_spon_last
        , first_value(vit.resp_rate) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.resp_rate is not null then 0 else 1 end asc, vit.rn desc) as resp_rate_last
        , first_value(vit.oxygen) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.oxygen is not null then 0 else 1 end asc, vit.rn desc) as oxygen_last
        , first_value(vit.sysbp) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.sysbp is not null then 0 else 1 end asc, vit.rn desc) as sysbp_last
        , first_value(vit.diabp) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.diabp is not null then 0 else 1 end asc, vit.rn desc) as diabp_last
        , first_value(vit.meanbp) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.meanbp is not null then 0 else 1 end asc, vit.rn desc) as meanbp_last
        , first_value(vit.sysbp_ni) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.sysbp_ni is not null then 0 else 1 end asc, vit.rn desc) as sysbp_ni_last
        , first_value(vit.diabp_ni) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.diabp_ni is not null then 0 else 1 end asc, vit.rn desc) as diabp_ni_last
        , first_value(vit.meanbp_ni) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.meanbp_ni is not null then 0 else 1 end asc, vit.rn desc) as meanbp_ni_last
        , first_value(vit.gcs_motor) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.gcs_motor is not null then 0 else 1 end asc, vit.rn desc) as gcs_motor_last
        , first_value(vit.gcs_verbal) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.gcs_verbal is not null then 0 else 1 end asc, vit.rn desc) as gcs_verbal_last
        , first_value(vit.gcs_eye) over (partition by vit.micro_specimen_id, vit.person_id order by case when vit.gcs_eye is not null then 0 else 1 end asc, vit.rn desc) as gcs_eye_last
        from """ + schemaName + """.vitals_stg_1 vit
        ;
    """

    with con:
        with con.cursor() as cursor:
            cursor.execute(dropVitalsLastQuery)
            cursor.execute(dataVitalsLastQuery)


def formatLabsStg1(con, schemaName):
    log.info("Formatting labs to " + schemaName + ".labs_stg_1")
    dropLabsStg1Query = """drop table if exists """ + schemaName + """.labs_stg_1 cascade"""
    dataLabs1Query = """
        create table """ + schemaName + """.labs_stg_1 as
        select
        lab.micro_specimen_id as micro_specimen_id
        , lab.person_id as person_id
        , lab.rn as rn
        , COALESCE(MAX(CASE WHEN lab.concept_name = 'Potassium [Moles/volume] in Serum or Plasma' THEN lab.value_as_number ELSE null END)) AS potassium
        , COALESCE(MAX(CASE WHEN lab.concept_name = 'Chloride [Moles/volume] in Serum or Plasma' THEN lab.value_as_number ELSE null END)) AS chloride
        , COALESCE(MAX(CASE WHEN lab.concept_name = 'Glucose [Mass/volume] in Serum or Plasma' THEN lab.value_as_number ELSE null END)) AS glucose
        , COALESCE(MAX(CASE WHEN lab.concept_name = 'Sodium [Moles/volume] in Serum or Plasma' THEN lab.value_as_number ELSE null END)) AS sodium
        , COALESCE(MAX(CASE WHEN lab.concept_name = 'Bicarbonate [Moles/volume] in Serum or Plasma' THEN lab.value_as_number ELSE null END)) AS bicarbonate
        , COALESCE(MAX(CASE WHEN lab.concept_name = 'Hemoglobin [Mass/volume] in Blood' THEN lab.value_as_number ELSE null END)) AS hemoglobin
        , COALESCE(MAX(CASE WHEN lab.concept_name = 'Creatinine [Mass/volume] in Serum or Plasma' THEN lab.value_as_number ELSE null END)) AS creatinine
        from
        """ + schemaName + """.labs_filtered lab
        group by lab.micro_specimen_id, lab.person_id, lab.rn
        ;
        """

    with con:
        with con.cursor() as cursor:
            cursor.execute(dropLabsStg1Query)
            cursor.execute(dataLabs1Query)


def formatLabsMax(con, schemaName):
    log.info("Formatting labs_stg_1 to " + schemaName + ".labs_max")
    dropLabsMaxQuery = """drop table if exists """ + schemaName + """.labs_max cascade"""
    dataLabsMaxQuery = """
        create table """ + schemaName + """.labs_max as
        select
        distinct
        lab.micro_specimen_id as micro_specimen_id
        , lab.person_id as person_id
        , max(lab.potassium) over (partition by lab.micro_specimen_id, lab.person_id) as potassium_max
        , max(lab.chloride) over (partition by lab.micro_specimen_id, lab.person_id) as chloride_max
        , max(lab.glucose) over (partition by lab.micro_specimen_id, lab.person_id) as glucose_max
        , max(lab.sodium) over (partition by lab.micro_specimen_id, lab.person_id) as sodium_max
        , max(lab.bicarbonate) over (partition by lab.micro_specimen_id, lab.person_id) as bicarbonate_max
        , max(lab.hemoglobin) over (partition by lab.micro_specimen_id, lab.person_id) as hemoglobin_max
        , max(lab.creatinine) over (partition by lab.micro_specimen_id, lab.person_id) as creatinine_max
        from """ + schemaName + """.labs_stg_1 lab
        ;
    """

    with con:
        with con.cursor() as cursor:
            cursor.execute(dropLabsMaxQuery)
            cursor.execute(dataLabsMaxQuery)


def formatLabsMin(con, schemaName):
    log.info("Formatting labs_stg_1 to " + schemaName + ".labs_min")
    dropLabsMinQuery = """drop table if exists """ + schemaName + """.labs_min cascade"""
    dataLabsMinQuery = """
        create table """ + schemaName + """.labs_min as
        select
        distinct
        lab.micro_specimen_id as micro_specimen_id
        , lab.person_id as person_id
        , min(lab.potassium) over (partition by lab.micro_specimen_id, lab.person_id) as potassium_min
        , min(lab.chloride) over (partition by lab.micro_specimen_id, lab.person_id) as chloride_min
        , min(lab.glucose) over (partition by lab.micro_specimen_id, lab.person_id) as glucose_min
        , min(lab.sodium) over (partition by lab.micro_specimen_id, lab.person_id) as sodium_min
        , min(lab.bicarbonate) over (partition by lab.micro_specimen_id, lab.person_id) as bicarbonate_min
        , min(lab.hemoglobin) over (partition by lab.micro_specimen_id, lab.person_id) as hemoglobin_min
        , min(lab.creatinine) over (partition by lab.micro_specimen_id, lab.person_id) as creatinine_min
        from """ + schemaName + """.labs_stg_1 lab
        ;
    """

    with con:
        with con.cursor() as cursor:
            cursor.execute(dropLabsMinQuery)
            cursor.execute(dataLabsMinQuery)


def formatLabsAvg(con, schemaName):
    log.info("Formatting labs_stg_1 to " + schemaName + ".labs_avg")
    dropLabsAvgQuery = """drop table if exists """ + schemaName + """.labs_avg cascade"""
    dataLabsAvgQuery = """
        create table """ + schemaName + """.labs_avg as
        select
        distinct
        lab.micro_specimen_id as micro_specimen_id
        , lab.person_id as person_id
        , avg(lab.potassium) over (partition by lab.micro_specimen_id, lab.person_id) as potassium_avg
        , avg(lab.chloride) over (partition by lab.micro_specimen_id, lab.person_id) as chloride_avg
        , avg(lab.glucose) over (partition by lab.micro_specimen_id, lab.person_id) as glucose_avg
        , avg(lab.sodium) over (partition by lab.micro_specimen_id, lab.person_id) as sodium_avg
        , avg(lab.bicarbonate) over (partition by lab.micro_specimen_id, lab.person_id) as bicarbonate_avg
        , avg(lab.hemoglobin) over (partition by lab.micro_specimen_id, lab.person_id) as hemoglobin_avg
        , avg(lab.creatinine) over (partition by lab.micro_specimen_id, lab.person_id) as creatinine_avg
        from """ + schemaName + """.labs_stg_1 lab
        ;
    """

    with con:
        with con.cursor() as cursor:
            cursor.execute(dropLabsAvgQuery)
            cursor.execute(dataLabsAvgQuery)

def formatLabsStd(con, schemaName):
    log.info("Formatting labs_stg_1 to " + schemaName + ".labs_std")
    dropLabsStdQuery = """drop table if exists """ + schemaName + """.labs_std cascade"""
    dataLabsStdQuery = """
        create table """ + schemaName + """.labs_std as
        select
        distinct
        lab.micro_specimen_id as micro_specimen_id
        , lab.person_id as person_id
        , stddev(lab.potassium) over (partition by lab.micro_specimen_id, lab.person_id) as potassium_sd
        , stddev(lab.chloride) over (partition by lab.micro_specimen_id, lab.person_id) as chloride_sd
        , stddev(lab.glucose) over (partition by lab.micro_specimen_id, lab.person_id) as glucose_sd
        , stddev(lab.sodium) over (partition by lab.micro_specimen_id, lab.person_id) as sodium_sd
        , stddev(lab.bicarbonate) over (partition by lab.micro_specimen_id, lab.person_id) as bicarbonate_sd
        , stddev(lab.hemoglobin) over (partition by lab.micro_specimen_id, lab.person_id) as hemoglobin_sd
        , stddev(lab.creatinine) over (partition by lab.micro_specimen_id, lab.person_id) as creatinine_sd
        from """ + schemaName + """.labs_stg_1 lab
        ;
    """

    with con:
        with con.cursor() as cursor:
            cursor.execute(dropLabsStdQuery)
            cursor.execute(dataLabsStdQuery)


def formatLabsFirst(con, schemaName):
    log.info("Formatting labs_stg_1 to " + schemaName + ".labs_first")
    dropLabsFirstQuery = """drop table if exists """ + schemaName + """.labs_first cascade"""
    dataLabsFirstQuery = """
        create table """ + schemaName + """.labs_first as
        select
        distinct
        lab.micro_specimen_id as micro_specimen_id
        , lab.person_id as person_id
        , first_value(lab.potassium) over (partition by lab.micro_specimen_id, lab.person_id order by case when lab.potassium is not null then 0 else 1 end asc, lab.rn asc) as potassium_first
        , first_value(lab.chloride) over (partition by lab.micro_specimen_id, lab.person_id order by case when lab.chloride is not null then 0 else 1 end asc, lab.rn asc) as chloride_first
        , first_value(lab.glucose) over (partition by lab.micro_specimen_id, lab.person_id order by case when lab.glucose is not null then 0 else 1 end asc, lab.rn asc) as glucose_first
        , first_value(lab.sodium) over (partition by lab.micro_specimen_id, lab.person_id order by case when lab.sodium is not null then 0 else 1 end asc, lab.rn asc) as sodium_first
        , first_value(lab.bicarbonate) over (partition by lab.micro_specimen_id, lab.person_id order by case when lab.bicarbonate is not null then 0 else 1 end asc, lab.rn asc) as bicarbonate_first
        , first_value(lab.hemoglobin) over (partition by lab.micro_specimen_id, lab.person_id order by case when lab.hemoglobin is not null then 0 else 1 end asc, lab.rn asc) as hemoglobin_first
        , first_value(lab.creatinine) over (partition by lab.micro_specimen_id, lab.person_id order by case when lab.creatinine is not null then 0 else 1 end asc, lab.rn asc) as creatinine_first
        from """ + schemaName + """.labs_stg_1 lab
        ;
    """

    with con:
        with con.cursor() as cursor:
            cursor.execute(dropLabsFirstQuery)
            cursor.execute(dataLabsFirstQuery)


def formatLabsLast(con, schemaName):
    log.info("Formatting labs_stg_1 to " + schemaName + ".labs_last")
    dropLabsLastQuery = """drop table if exists """ + schemaName + """.labs_last cascade"""
    dataLabsLastQuery = """
        create table """ + schemaName + """.labs_last as
        select
        distinct
        lab.micro_specimen_id as micro_specimen_id
        , lab.person_id as person_id
        , first_value(lab.potassium) over (partition by lab.micro_specimen_id, lab.person_id order by case when lab.potassium is not null then 0 else 1 end asc, lab.rn desc) as potassium_last
        , first_value(lab.chloride) over (partition by lab.micro_specimen_id, lab.person_id order by case when lab.chloride is not null then 0 else 1 end asc, lab.rn desc) as chloride_last
        , first_value(lab.glucose) over (partition by lab.micro_specimen_id, lab.person_id order by case when lab.glucose is not null then 0 else 1 end asc, lab.rn desc) as glucose_last
        , first_value(lab.sodium) over (partition by lab.micro_specimen_id, lab.person_id order by case when lab.sodium is not null then 0 else 1 end asc, lab.rn desc) as sodium_last
        , first_value(lab.bicarbonate) over (partition by lab.micro_specimen_id, lab.person_id order by case when lab.bicarbonate is not null then 0 else 1 end asc, lab.rn desc) as bicarbonate_last
        , first_value(lab.hemoglobin) over (partition by lab.micro_specimen_id, lab.person_id order by case when lab.hemoglobin is not null then 0 else 1 end asc, lab.rn desc) as hemoglobin_last
        , first_value(lab.creatinine) over (partition by lab.micro_specimen_id, lab.person_id order by case when lab.creatinine is not null then 0 else 1 end asc, lab.rn desc) as creatinine_last
        from """ + schemaName + """.labs_stg_1 lab
        ;
    """

    with con:
        with con.cursor() as cursor:
            cursor.execute(dropLabsLastQuery)
            cursor.execute(dataLabsLastQuery)


def formatData(con, schemaName):
    log.info("Formatting data matrix to " + schemaName + ".data_matrix")
    dropDataMatrixQuery = """drop table if exists """ + schemaName + """.data_matrix cascade"""
    dataDataMatrixQuery = """
        create table """ + schemaName + """.data_matrix as
        select
        coh."micro_specimen_id"
        , sta."person_id"
        , sta."visit_duration_hrs"
        , sta."gender"
        , vit_max."temp_max"
        , vit_max."heartrate_max"
        , vit_max."breath_rate_vent_max"
        , vit_max."breath_rate_spon_max"
        , vit_max."resp_rate_max"
        , vit_max."oxygen_max"
        , vit_max."sysbp_max"
        , vit_max."diabp_max"
        , vit_max."meanbp_max"
        , vit_max."sysbp_ni_max"
        , vit_max."diabp_ni_max"
        , vit_max."meanbp_ni_max"
        , vit_max."gcs_motor_max"
        , vit_max."gcs_verbal_max"
        , vit_max."gcs_eye_max"
        , vit_min."temp_min"
        , vit_min."heartrate_min"
        , vit_min."breath_rate_vent_min"
        , vit_min."breath_rate_spon_min"
        , vit_min."resp_rate_min"
        , vit_min."oxygen_min"
        , vit_min."sysbp_min"
        , vit_min."diabp_min"
        , vit_min."meanbp_min"
        , vit_min."sysbp_ni_min"
        , vit_min."diabp_ni_min"
        , vit_min."meanbp_ni_min"
        , vit_min."gcs_motor_min"
        , vit_min."gcs_verbal_min"
        , vit_min."gcs_eye_min"
        , vit_avg."temp_avg"
        , vit_avg."heartrate_avg"
        , vit_avg."breath_rate_vent_avg"
        , vit_avg."breath_rate_spon_avg"
        , vit_avg."resp_rate_avg"
        , vit_avg."oxygen_avg"
        , vit_avg."sysbp_avg"
        , vit_avg."diabp_avg"
        , vit_avg."meanbp_avg"
        , vit_avg."sysbp_ni_avg"
        , vit_avg."diabp_ni_avg"
        , vit_avg."meanbp_ni_avg"
        , vit_avg."gcs_motor_avg"
        , vit_avg."gcs_verbal_avg"
        , vit_avg."gcs_eye_avg"
        , vit_std."temp_sd"
        , vit_std."heartrate_sd"
        , vit_std."breath_rate_vent_sd"
        , vit_std."breath_rate_spon_sd"
        , vit_std."resp_rate_sd"
        , vit_std."oxygen_sd"
        , vit_std."sysbp_sd"
        , vit_std."diabp_sd"
        , vit_std."meanbp_sd"
        , vit_std."sysbp_ni_sd"
        , vit_std."diabp_ni_sd"
        , vit_std."meanbp_ni_sd"
        , vit_std."gcs_motor_sd"
        , vit_std."gcs_verbal_sd"
        , vit_std."gcs_eye_sd"
        , vit_first."temp_first"
        , vit_first."heartrate_first"
        , vit_first."breath_rate_vent_first"
        , vit_first."breath_rate_spon_first"
        , vit_first."resp_rate_first"
        , vit_first."oxygen_first"
        , vit_first."sysbp_first"
        , vit_first."diabp_first"
        , vit_first."meanbp_first"
        , vit_first."sysbp_ni_first"
        , vit_first."diabp_ni_first"
        , vit_first."meanbp_ni_first"
        , vit_first."gcs_motor_first"
        , vit_first."gcs_verbal_first"
        , vit_first."gcs_eye_first"
        , vit_last."temp_last"
        , vit_last."heartrate_last"
        , vit_last."breath_rate_vent_last"
        , vit_last."breath_rate_spon_last"
        , vit_last."resp_rate_last"
        , vit_last."oxygen_last"
        , vit_last."sysbp_last"
        , vit_last."diabp_last"
        , vit_last."meanbp_last"
        , vit_last."sysbp_ni_last"
        , vit_last."diabp_ni_last"
        , vit_last."meanbp_ni_last"
        , vit_last."gcs_motor_last"
        , vit_last."gcs_verbal_last"
        , vit_last."gcs_eye_last"
        , lab_max."potassium_max"
        , lab_max."chloride_max"
        , lab_max."glucose_max"
        , lab_max."sodium_max"
        , lab_max."bicarbonate_max"
        , lab_max."hemoglobin_max"
        , lab_max."creatinine_max"
        , lab_min."potassium_min"
        , lab_min."chloride_min"
        , lab_min."glucose_min"
        , lab_min."sodium_min"
        , lab_min."bicarbonate_min"
        , lab_min."hemoglobin_min"
        , lab_min."creatinine_min"
        , lab_avg."potassium_avg"
        , lab_avg."chloride_avg"
        , lab_avg."glucose_avg"
        , lab_avg."sodium_avg"
        , lab_avg."bicarbonate_avg"
        , lab_avg."hemoglobin_avg"
        , lab_avg."creatinine_avg"
        , lab_std."potassium_sd"
        , lab_std."chloride_sd"
        , lab_std."glucose_sd"
        , lab_std."sodium_sd"
        , lab_std."bicarbonate_sd"
        , lab_std."hemoglobin_sd"
        , lab_std."creatinine_sd"
        , lab_first."potassium_first"
        , lab_first."chloride_first"
        , lab_first."glucose_first"
        , lab_first."sodium_first"
        , lab_first."bicarbonate_first"
        , lab_first."hemoglobin_first"
        , lab_first."creatinine_first"
        , lab_last."potassium_last"
        , lab_last."chloride_last"
        , lab_last."glucose_last"
        , lab_last."sodium_last"
        , lab_last."bicarbonate_last"
        , lab_last."hemoglobin_last"
        , lab_last."creatinine_last"
        , dth."death_datetime"
        from
        """ + schemaName + """.cohort coh
        inner join """ + schemaName + """.statics sta
        on sta.micro_specimen_id = coh.micro_specimen_id
        inner join """ + schemaName + """.vitals_max vit_max
        on vit_max.micro_specimen_id = coh.micro_specimen_id
        inner join """ + schemaName + """.vitals_min vit_min
        on vit_min.micro_specimen_id = coh.micro_specimen_id
        inner join """ + schemaName + """.vitals_avg vit_avg
        on vit_avg.micro_specimen_id = coh.micro_specimen_id
        inner join """ + schemaName + """.vitals_std vit_std
        on vit_std.micro_specimen_id = coh.micro_specimen_id
        inner join """ + schemaName + """.vitals_first vit_first
        on vit_first.micro_specimen_id = coh.micro_specimen_id
        inner join """ + schemaName + """.vitals_last vit_last
        on vit_last.micro_specimen_id = coh.micro_specimen_id
        inner join """ + schemaName + """.labs_max lab_max
        on lab_max.micro_specimen_id = coh.micro_specimen_id
        inner join """ + schemaName + """.labs_min lab_min
        on lab_min.micro_specimen_id = coh.micro_specimen_id
        inner join """ + schemaName + """.labs_avg lab_avg
        on lab_avg.micro_specimen_id = coh.micro_specimen_id
        inner join """ + schemaName + """.labs_std lab_std
        on lab_std.micro_specimen_id = coh.micro_specimen_id
        inner join """ + schemaName + """.labs_first lab_first
        on lab_first.micro_specimen_id = coh.micro_specimen_id
        inner join """ + schemaName + """.labs_last lab_last
        on lab_last.micro_specimen_id = coh.micro_specimen_id
        inner join """ + schemaName + """.deaths dth
        on dth.micro_specimen_id = coh.micro_specimen_id
        ;
    """

    with con:
        with con.cursor() as cursor:
            cursor.execute(dropDataMatrixQuery)
            cursor.execute(dataDataMatrixQuery)


def imputeMissingData(con, schemaName):

    from mortality_prediction.MissForest import MissForest
    from sklearn.preprocessing import StandardScaler

    log.info("Reading data from " + schemaName + ".data_matrix")
    dataQuery = """select * from """ + schemaName + """.data_matrix;"""
    dataDf = pd.read_sql_query(dataQuery, con)

    log.info("Standarising numerical columns")
    cols = ["temp_max", "heartrate_max", "breath_rate_vent_max", "breath_rate_spon_max", "resp_rate_max", "oxygen_max", "sysbp_max", "diabp_max", "meanbp_max", "sysbp_ni_max", "diabp_ni_max", "meanbp_ni_max", "gcs_motor_max", "gcs_verbal_max", "gcs_eye_max", "temp_min", "heartrate_min", "breath_rate_vent_min", "breath_rate_spon_min", "resp_rate_min", "oxygen_min", "sysbp_min", "diabp_min", "meanbp_min", "sysbp_ni_min", "diabp_ni_min", "meanbp_ni_min", "gcs_motor_min", "gcs_verbal_min", "gcs_eye_min", "temp_avg", "heartrate_avg", "breath_rate_vent_avg", "breath_rate_spon_avg", "resp_rate_avg", "oxygen_avg", "sysbp_avg", "diabp_avg", "meanbp_avg", "sysbp_ni_avg", "diabp_ni_avg", "meanbp_ni_avg", "gcs_motor_avg", "gcs_verbal_avg", "gcs_eye_avg", "temp_sd", "heartrate_sd", "breath_rate_vent_sd", "breath_rate_spon_sd", "resp_rate_sd", "oxygen_sd", "sysbp_sd", "diabp_sd", "meanbp_sd", "sysbp_ni_sd", "diabp_ni_sd", "meanbp_ni_sd", "gcs_motor_sd", "gcs_verbal_sd", "gcs_eye_sd", "temp_first", "heartrate_first", "breath_rate_vent_first", "breath_rate_spon_first", "resp_rate_first", "oxygen_first", "sysbp_first", "diabp_first", "meanbp_first", "sysbp_ni_first", "diabp_ni_first", "meanbp_ni_first", "gcs_motor_first", "gcs_verbal_first", "gcs_eye_first", "temp_last", "heartrate_last", "breath_rate_vent_last", "breath_rate_spon_last", "resp_rate_last", "oxygen_last", "sysbp_last", "diabp_last", "meanbp_last", "sysbp_ni_last", "diabp_ni_last", "meanbp_ni_last", "gcs_motor_last", "gcs_verbal_last", "gcs_eye_last", "potassium_max", "chloride_max", "glucose_max", "sodium_max", "bicarbonate_max", "hemoglobin_max", "creatinine_max", "potassium_min", "chloride_min", "glucose_min", "sodium_min", "bicarbonate_min", "hemoglobin_min", "creatinine_min", "potassium_avg", "chloride_avg", "glucose_avg", "sodium_avg", "bicarbonate_avg", "hemoglobin_avg", "creatinine_avg", "potassium_sd", "chloride_sd", "glucose_sd", "sodium_sd", "bicarbonate_sd", "hemoglobin_sd", "creatinine_sd", "potassium_first", "chloride_first", "glucose_first", "sodium_first", "bicarbonate_first", "hemoglobin_first", "creatinine_first", "potassium_last", "chloride_last", "glucose_last", "sodium_last", "bicarbonate_last", "hemoglobin_last", "creatinine_last"]

    data = StandardScaler().fit_transform(dataDf[cols])
    standardDf = pd.DataFrame(data, columns = cols)
    dataDf = dataDf.drop(cols, axis = 1).join(standardDf)

    log.info("Imputing missing data")
    rawDf = dataDf.drop(['micro_specimen_id', 'person_id', 'death_datetime'], axis = 1)
    mfImputer = MissForest()
    mfImputedData = mfImputer.fit(rawDf).transform(rawDf)
    mfImputedDf = pd.DataFrame(mfImputedData, columns=rawDf.columns, index=rawDf.index)
    finalDf = mfImputedDf.join(dataDf[['micro_specimen_id', 'person_id', 'death_datetime']])

    # import pandas as pd
    import numpy as np

    finalDf = finalDf.replace({np.nan: None})

    log.info("Dropping table: " + schemaName + ".data_matrix_qc")
    dropTableQuery = """drop table if exists """ + schemaName + """.data_matrix_qc cascade;"""
    log.info("Creating table: " + schemaName + ".data_matrix_qc")
    createTableQuery = """create table """ + schemaName + """.data_matrix_qc (
        "micro_specimen_id" bigint,
        "person_id" bigint,
        "visit_duration_hrs" numeric,
        "gender" integer,
        "temp_max" numeric,
        "heartrate_max" numeric,
        "breath_rate_vent_max" numeric,
        "breath_rate_spon_max" numeric,
        "resp_rate_max" numeric,
        "oxygen_max" numeric,
        "sysbp_max" numeric,
        "diabp_max" numeric,
        "meanbp_max" numeric,
        "sysbp_ni_max" numeric,
        "diabp_ni_max" numeric,
        "meanbp_ni_max" numeric,
        "gcs_motor_max" numeric,
        "gcs_verbal_max" numeric,
        "gcs_eye_max" numeric,
        "temp_min" numeric,
        "heartrate_min" numeric,
        "breath_rate_vent_min" numeric,
        "breath_rate_spon_min" numeric,
        "resp_rate_min" numeric,
        "oxygen_min" numeric,
        "sysbp_min" numeric,
        "diabp_min" numeric,
        "meanbp_min" numeric,
        "sysbp_ni_min" numeric,
        "diabp_ni_min" numeric,
        "meanbp_ni_min" numeric,
        "gcs_motor_min" numeric,
        "gcs_verbal_min" numeric,
        "gcs_eye_min" numeric,
        "temp_avg" numeric,
        "heartrate_avg" numeric,
        "breath_rate_vent_avg" numeric,
        "breath_rate_spon_avg" numeric,
        "resp_rate_avg" numeric,
        "oxygen_avg" numeric,
        "sysbp_avg" numeric,
        "diabp_avg" numeric,
        "meanbp_avg" numeric,
        "sysbp_ni_avg" numeric,
        "diabp_ni_avg" numeric,
        "meanbp_ni_avg" numeric,
        "gcs_motor_avg" numeric,
        "gcs_verbal_avg" numeric,
        "gcs_eye_avg" numeric,
        "temp_sd" numeric,
        "heartrate_sd" numeric,
        "breath_rate_vent_sd" numeric,
        "breath_rate_spon_sd" numeric,
        "resp_rate_sd" numeric,
        "oxygen_sd" numeric,
        "sysbp_sd" numeric,
        "diabp_sd" numeric,
        "meanbp_sd" numeric,
        "sysbp_ni_sd" numeric,
        "diabp_ni_sd" numeric,
        "meanbp_ni_sd" numeric,
        "gcs_motor_sd" numeric,
        "gcs_verbal_sd" numeric,
        "gcs_eye_sd" numeric,
        "temp_first" numeric,
        "heartrate_first" numeric,
        "breath_rate_vent_first" numeric,
        "breath_rate_spon_first" numeric,
        "resp_rate_first" numeric,
        "oxygen_first" numeric,
        "sysbp_first" numeric,
        "diabp_first" numeric,
        "meanbp_first" numeric,
        "sysbp_ni_first" numeric,
        "diabp_ni_first" numeric,
        "meanbp_ni_first" numeric,
        "gcs_motor_first" numeric,
        "gcs_verbal_first" numeric,
        "gcs_eye_first" numeric,
        "temp_last" numeric,
        "heartrate_last" numeric,
        "breath_rate_vent_last" numeric,
        "breath_rate_spon_last" numeric,
        "resp_rate_last" numeric,
        "oxygen_last" numeric,
        "sysbp_last" numeric,
        "diabp_last" numeric,
        "meanbp_last" numeric,
        "sysbp_ni_last" numeric,
        "diabp_ni_last" numeric,
        "meanbp_ni_last" numeric,
        "gcs_motor_last" numeric,
        "gcs_verbal_last" numeric,
        "gcs_eye_last" numeric,
        "potassium_max" numeric,
        "chloride_max" numeric,
        "glucose_max" numeric,
        "sodium_max" numeric,
        "bicarbonate_max" numeric,
        "hemoglobin_max" numeric,
        "creatinine_max" numeric,
        "potassium_min" numeric,
        "chloride_min" numeric,
        "glucose_min" numeric,
        "sodium_min" numeric,
        "bicarbonate_min" numeric,
        "hemoglobin_min" numeric,
        "creatinine_min" numeric,
        "potassium_avg" numeric,
        "chloride_avg" numeric,
        "glucose_avg" numeric,
        "sodium_avg" numeric,
        "bicarbonate_avg" numeric,
        "hemoglobin_avg" numeric,
        "creatinine_avg" numeric,
        "potassium_sd" numeric,
        "chloride_sd" numeric,
        "glucose_sd" numeric,
        "sodium_sd" numeric,
        "bicarbonate_sd" numeric,
        "hemoglobin_sd" numeric,
        "creatinine_sd" numeric,
        "potassium_first" numeric,
        "chloride_first" numeric,
        "glucose_first" numeric,
        "sodium_first" numeric,
        "bicarbonate_first" numeric,
        "hemoglobin_first" numeric,
        "creatinine_first" numeric,
        "potassium_last" numeric,
        "chloride_last" numeric,
        "glucose_last" numeric,
        "sodium_last" numeric,
        "bicarbonate_last" numeric,
        "hemoglobin_last" numeric,
        "creatinine_last" numeric,
        "death_datetime" timestamp
    )"""

    with con:
        with con.cursor() as cursor:
            cursor.execute(dropTableQuery)
            cursor.execute(createTableQuery)

    df_columns = [
        "micro_specimen_id",
        "person_id",
        "visit_duration_hrs",
        "gender",
        "temp_max",
        "heartrate_max",
        "breath_rate_vent_max",
        "breath_rate_spon_max",
        "resp_rate_max",
        "oxygen_max",
        "sysbp_max",
        "diabp_max",
        "meanbp_max",
        "sysbp_ni_max",
        "diabp_ni_max",
        "meanbp_ni_max",
        "gcs_motor_max",
        "gcs_verbal_max",
        "gcs_eye_max",
        "temp_min",
        "heartrate_min",
        "breath_rate_vent_min",
        "breath_rate_spon_min",
        "resp_rate_min",
        "oxygen_min",
        "sysbp_min",
        "diabp_min",
        "meanbp_min",
        "sysbp_ni_min",
        "diabp_ni_min",
        "meanbp_ni_min",
        "gcs_motor_min",
        "gcs_verbal_min",
        "gcs_eye_min",
        "temp_avg",
        "heartrate_avg",
        "breath_rate_vent_avg",
        "breath_rate_spon_avg",
        "resp_rate_avg",
        "oxygen_avg",
        "sysbp_avg",
        "diabp_avg",
        "meanbp_avg",
        "sysbp_ni_avg",
        "diabp_ni_avg",
        "meanbp_ni_avg",
        "gcs_motor_avg",
        "gcs_verbal_avg",
        "gcs_eye_avg",
        "temp_sd",
        "heartrate_sd",
        "breath_rate_vent_sd",
        "breath_rate_spon_sd",
        "resp_rate_sd",
        "oxygen_sd",
        "sysbp_sd",
        "diabp_sd",
        "meanbp_sd",
        "sysbp_ni_sd",
        "diabp_ni_sd",
        "meanbp_ni_sd",
        "gcs_motor_sd",
        "gcs_verbal_sd",
        "gcs_eye_sd",
        "temp_first",
        "heartrate_first",
        "breath_rate_vent_first",
        "breath_rate_spon_first",
        "resp_rate_first",
        "oxygen_first",
        "sysbp_first",
        "diabp_first",
        "meanbp_first",
        "sysbp_ni_first",
        "diabp_ni_first",
        "meanbp_ni_first",
        "gcs_motor_first",
        "gcs_verbal_first",
        "gcs_eye_first",
        "temp_last",
        "heartrate_last",
        "breath_rate_vent_last",
        "breath_rate_spon_last",
        "resp_rate_last",
        "oxygen_last",
        "sysbp_last",
        "diabp_last",
        "meanbp_last",
        "sysbp_ni_last",
        "diabp_ni_last",
        "meanbp_ni_last",
        "gcs_motor_last",
        "gcs_verbal_last",
        "gcs_eye_last",
        "potassium_max",
        "chloride_max",
        "glucose_max",
        "sodium_max",
        "bicarbonate_max",
        "hemoglobin_max",
        "creatinine_max",
        "potassium_min",
        "chloride_min",
        "glucose_min",
        "sodium_min",
        "bicarbonate_min",
        "hemoglobin_min",
        "creatinine_min",
        "potassium_avg",
        "chloride_avg",
        "glucose_avg",
        "sodium_avg",
        "bicarbonate_avg",
        "hemoglobin_avg",
        "creatinine_avg",
        "potassium_sd",
        "chloride_sd",
        "glucose_sd",
        "sodium_sd",
        "bicarbonate_sd",
        "hemoglobin_sd",
        "creatinine_sd",
        "potassium_first",
        "chloride_first",
        "glucose_first",
        "sodium_first",
        "bicarbonate_first",
        "hemoglobin_first",
        "creatinine_first",
        "potassium_last",
        "chloride_last",
        "glucose_last",
        "sodium_last",
        "bicarbonate_last",
        "hemoglobin_last",
        "creatinine_last",
        "death_datetime",
    ]

    log.info("Loading table: " + schemaName + ".data_matrix_qc")

    import psycopg2.extras

    if len(finalDf) > 0:

        table = schemaName + '.data_matrix_qc'

        # df_columns = ['micro_specimen_id', 'person_id', 'discharge_mortality', 'one_day_mortality', 'two_day_mortality', 'thirty_day_mortality', 'sixty_day_mortality', 'ninety_day_mortality', 'sepsis', 'Ambulatory.Clinic...Center', 'Ambulatory.Surgical.Center', 'Emergency.Room...Hospital', 'Emergency.Room.and.Inpatient.Visit', 'Inpatient.Visit', 'Observation.Room', 'AMBULATORY.OBSERVATION', 'DIRECT.EMER.', 'DIRECT.OBSERVATION', 'ELECTIVE', 'EU.OBSERVATION', 'EW.EMER.', 'OBSERVATION.ADMIT', 'SURGICAL.SAME.DAY.ADMISSION', 'URGENT', 'AMBULATORY.SURGERY.TRANSFER', 'CLINIC.REFERRAL', 'EMERGENCY.ROOM', 'INFORMATION.NOT.AVAILABLE', 'INTERNAL.TRANSFER.TO.OR.FROM.PSYCH', 'PACU', 'PHYSICIAN.REFERRAL', 'PROCEDURE.SITE', 'TRANSFER.FROM.HOSPITAL', 'TRANSFER.FROM.SKILLED.NURSING.FACILITY', 'WALK.IN.SELF.REFERRAL', 'visit_duration_hrs', 'temp_max', 'heartrate_max', 'breath_rate_vent_max', 'breath_rate_spon_max', 'resp_rate_max', 'oxygen_max', 'sysbp_max', 'diabp_max', 'meanbp_max', 'sysbp_ni_max', 'diabp_ni_max', 'meanbp_ni_max', 'gcs_motor_max', 'gcs_verbal_max', 'gcs_eye_max', 'temp_min', 'heartrate_min', 'breath_rate_vent_min', 'breath_rate_spon_min', 'resp_rate_min', 'oxygen_min', 'sysbp_min', 'diabp_min', 'meanbp_min', 'sysbp_ni_min', 'diabp_ni_min', 'meanbp_ni_min', 'gcs_motor_min', 'gcs_verbal_min', 'gcs_eye_min', 'temp_avg', 'heartrate_avg', 'breath_rate_vent_avg', 'breath_rate_spon_avg', 'resp_rate_avg', 'oxygen_avg', 'sysbp_avg', 'diabp_avg', 'meanbp_avg', 'sysbp_ni_avg', 'diabp_ni_avg', 'meanbp_ni_avg', 'gcs_motor_avg', 'gcs_verbal_avg', 'gcs_eye_avg', 'temp_sd', 'heartrate_sd', 'breath_rate_vent_sd', 'breath_rate_spon_sd', 'resp_rate_sd', 'oxygen_sd', 'sysbp_sd', 'diabp_sd', 'meanbp_sd', 'sysbp_ni_sd', 'diabp_ni_sd', 'meanbp_ni_sd', 'gcs_motor_sd', 'gcs_verbal_sd', 'gcs_eye_sd', 'temp_first', 'heartrate_first', 'breath_rate_vent_first', 'breath_rate_spon_first', 'resp_rate_first', 'oxygen_first', 'sysbp_first', 'diabp_first', 'meanbp_first', 'sysbp_ni_first', 'diabp_ni_first', 'meanbp_ni_first', 'gcs_motor_first', 'gcs_verbal_first', 'gcs_eye_first', 'temp_last', 'heartrate_last', 'breath_rate_vent_last', 'breath_rate_spon_last', 'resp_rate_last', 'oxygen_last', 'sysbp_last', 'diabp_last', 'meanbp_last', 'sysbp_ni_last', 'diabp_ni_last', 'meanbp_ni_last', 'gcs_motor_last', 'gcs_verbal_last', 'gcs_eye_last', 'potassium_max', 'chloride_max', 'glucose_max', 'sodium_max', 'bicarbonate_max', 'hemoglobin_max', 'creatinine_max', 'potassium_min', 'chloride_min', 'glucose_min', 'sodium_min', 'bicarbonate_min', 'hemoglobin_min', 'creatinine_min', 'potassium_avg', 'chloride_avg', 'glucose_avg', 'sodium_avg', 'bicarbonate_avg', 'hemoglobin_avg', 'creatinine_avg', 'potassium_sd', 'chloride_sd', 'glucose_sd', 'sodium_sd', 'bicarbonate_sd', 'hemoglobin_sd', 'creatinine_sd', 'potassium_first', 'chloride_first', 'glucose_first', 'sodium_first', 'bicarbonate_first', 'hemoglobin_first', 'creatinine_first', 'potassium_last', 'chloride_last', 'glucose_last', 'sodium_last', 'bicarbonate_last', 'hemoglobin_last', 'creatinine_last', 'time_to_antibiotic']
        # create (col1,col2,...)
        columns = '"' + '", "'.join(df_columns) + '"'

        # create VALUES('%s', '%s",...) one '%s' per column
        values = "VALUES({})".format(",".join(["%s" for _ in df_columns])) 

        #create INSERT INTO table (columns) VALUES('%s',...)
        insert_stmt = "INSERT INTO {} ({}) {}".format(table, columns, values)

        cur = con.cursor()
        # print(finalDf[df_columns].values)
        # print(columns)
        # return
        psycopg2.extras.execute_batch(cur, insert_stmt, finalDf[df_columns].values)
        con.commit()
        cur.close()


def formatAll(con, vitalsBefore = 48, vitalsAfter = 48, labsBefore = 72, labsAfter = 72, anchor = "micro"):
    schemaName = anchor + "_vb_" + str(vitalsBefore) + "_va_" + str(vitalsAfter) + "_lb_" + str(labsBefore) + "_la_" + str(labsAfter)
    log.info("Schema Name: " + schemaName)
    filterVitalsOutliers(con=con, schemaName=schemaName)
    formatVitalsStg1(con=con, schemaName=schemaName)
    formatVitalsMax(con=con, schemaName=schemaName)
    formatVitalsMin(con=con, schemaName=schemaName)
    formatVitalsAvg(con=con, schemaName=schemaName)
    formatVitalsStd(con=con, schemaName=schemaName)
    formatVitalsFirst(con=con, schemaName=schemaName)
    formatVitalsLast(con=con, schemaName=schemaName)
    filterLabsOutliers(con=con, schemaName=schemaName)
    formatLabsStg1(con=con, schemaName=schemaName)
    formatLabsMax(con=con, schemaName=schemaName)
    formatLabsMin(con=con, schemaName=schemaName)
    formatLabsAvg(con=con, schemaName=schemaName)
    formatLabsStd(con=con, schemaName=schemaName)
    formatLabsFirst(con=con, schemaName=schemaName)
    formatLabsLast(con=con, schemaName=schemaName)
    formatData(con=con, schemaName=schemaName)
    imputeMissingData(con=con, schemaName=schemaName)


if __name__ == "__main__":
    formatAll(vitalsBefore = 48, vitalsAfter = 48, labsBefore = 72, labsAfter = 72)
