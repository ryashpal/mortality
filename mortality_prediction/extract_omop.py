import logging

log = logging.getLogger("Pipeline")


def createSchema(con, schemaName):
    log.info("Creating schema" + schemaName)
    createSchemaQuery = """create schema if not exists """ + schemaName
    with con:
        with con.cursor() as cursor:
            cursor.execute(createSchemaQuery)


def dropSchema(con, schemaName):
    log.info("Dropping schema" + schemaName)
    dropSchemaQuery = """drop schema if exists """ + schemaName + """ cascade"""
    with con:
        with con.cursor() as cursor:
            cursor.execute(dropSchemaQuery)


def extractCohort(con, schemaName):
    log.info("Extracting cohort to " + schemaName + ".cohort")
    dropCohortQuery = """drop table if exists """ + schemaName + """.cohort cascade"""
    cohortQuery = """
            create table """ + schemaName + """.cohort as
            with stg1 as
            (
                select
                micro_specimen_id
                , subject_id
                , min(chartdate) as chartdate
                , min(charttime) as charttime
                from
                mimiciv.microbiologyevents
                where
                spec_type_desc = 'BLOOD CULTURE'
                and org_itemid != 90760
                and org_name is not null
                group by subject_id, micro_specimen_id
            ),
            stg2 as (
                select
                stg1.micro_specimen_id
                , per.person_id
                , coalesce(stg1.charttime, stg1.chartdate) as chart_time
                , ic.intime
                from stg1
                inner join mimiciv.patients pat
                on stg1.subject_id = pat.subject_id
                inner join omop_cdm.person per
                on per.person_source_value::int = pat.subject_id
                left join mimiciv.admissions ad
                on ad.subject_id = pat.subject_id and coalesce(stg1.charttime, stg1.chartdate) BETWEEN (ad.admittime - INTERVAL '0 DAY') AND (ad.dischtime + INTERVAL '0 DAY')
                left join mimiciv.icustays ic
                on ic.hadm_id = ad.hadm_id and coalesce(stg1.charttime, stg1.chartdate) BETWEEN (ic.intime - INTERVAL '0 DAY') AND (ic.outtime + INTERVAL '0 DAY')
                where (floor(date_part('day', stg1.chartdate - make_timestamp(pat.anchor_year, 1, 1, 0, 0, 0))/365.0) + pat.anchor_age) > 18
            )
            select
            *
            from
            stg2
            ;
        """

    with con:
        with con.cursor() as cursor:
            cursor.execute(dropCohortQuery)
            cursor.execute(cohortQuery)


def extractStatics(con, schemaName):
    log.info("Extracting statics to " + schemaName + ".statics")
    dropStaticQuery = """drop table if exists """ + schemaName + """.statics cascade"""
    staticQuery = """
        create table """ + schemaName + """.statics as
        select
        coh.micro_specimen_id as micro_specimen_id,
        coh.person_id as person_id,
        case when per.gender_source_value = 'M' then 1 else 0 end as gender,
        (DATE_PART('day', (vo.visit_end_datetime - vo.visit_start_datetime)) * 24) + DATE_PART('hour', (vo.visit_end_datetime - vo.visit_start_datetime)) AS visit_duration_hrs,
        (floor(date_part('day', coh.chart_time - make_timestamp(pat.anchor_year, 1, 1, 0, 0, 0))/365.0) + pat.anchor_age) as age
        from
        """ + schemaName + """.cohort coh
        inner join omop_cdm.person per
        on coh.person_id = per.person_id
        inner join mimiciv.patients pat
        on pat.subject_id = per.person_source_value::int
        inner join omop_cdm.visit_occurrence vo
        on vo.person_id = per.person_id
        and (vo.visit_start_datetime < coh.chart_time) and (vo.visit_end_datetime > coh.chart_time)
        inner join omop_cdm.concept con_vo
        on con_vo.concept_id = vo.visit_concept_id
        inner join omop_cdm.concept con_src
        on con_src.concept_id = vo.visit_source_concept_id
        ;
        """

    with con:
        with con.cursor() as cursor:
            cursor.execute(dropStaticQuery)
            cursor.execute(staticQuery)


def extractVitals(con, schemaName, vitalsBefore = 48, vitalsAfter = 48, anchor = "micro"):
    log.info("Extracting vitals to " + schemaName + ".vitals")

    anchorColumn = "coh.chart_time"
    if anchor == "admission":
        anchorColumn = "coh.intime"

    dropVitalsQuery = """drop table if exists """ + schemaName + """.vitals cascade"""
    vitalsQuery = """
        create table """ + schemaName + """.vitals as
        with vitals_stg_1 as
        (
            select
            coh.micro_specimen_id as micro_specimen_id,
            mmt.person_id as person_id,
            mmt.measurement_datetime as measurement_datetime,
            mmt.unit_source_value as unit_source_value,
            mmt.value_as_number as value_as_number,
            cpt.concept_name as concept_name
            from
            omop_cdm.measurement mmt
            inner join omop_cdm.concept cpt
            on cpt.concept_id = mmt.measurement_concept_id
            inner join """ + schemaName + """.cohort coh
            on coh.person_id = mmt.person_id
            where
            measurement_concept_id in (
            3027018 -- Heart rate
            , 21492239, 3004249 -- Systolic blood pressure
            , 21492240, 3012888 -- Diastolic blood pressure
            , 3027598, 21492241 -- Mean blood pressure
            , 1175625, 3024171, 3007469 -- Respiratory rate
            , 3020891 -- Body temperature
            , 40762499 -- Oxygen saturation in Arterial blood by Pulse oximetry
            , 3016335 -- Glasgow coma score eye opening
            , 3009094 -- Glasgow coma score verbal
            , 3008223 -- Glasgow coma score motor
            )
            and value_as_number is not null
            and (mmt.measurement_datetime > """ + anchorColumn + """ - interval '""" + str(vitalsBefore) + """ hour')
            and (mmt.measurement_datetime < """ + anchorColumn + """ + interval '""" + str(vitalsAfter) + """ hour')
        )
        , vitals_stg_2 AS
        (
        select
            micro_specimen_id,
            person_id,
            measurement_datetime,
            unit_source_value,
            value_as_number,
            concept_name,
            row_number() over (partition by person_id, concept_name order by measurement_datetime) as rn
        from vitals_stg_1
        )
        select * from vitals_stg_2
        ;
        """

    with con:
        with con.cursor() as cursor:
            cursor.execute(dropVitalsQuery)
            cursor.execute(vitalsQuery)


def extractLabs(con, schemaName, labsBefore = 72, labsAfter = 72, anchor = "micro"):
    log.info("Extracting labs to " + schemaName + ".labs")

    anchorColumn = "coh.chart_time"
    if anchor == "admission":
        anchorColumn = "coh.intime"

    dropLabsQuery = """drop table if exists """ + schemaName + """.labs cascade"""
    labsQuery = """
        create table """ + schemaName + """.labs as
        with labs_stg_1 as
            (
                select
                coh.micro_specimen_id as micro_specimen_id,
                mmt.person_id AS person_id,
                measurement_datetime as measurement_datetime,
                unit_source_value as unit_source_value,
                value_as_number as value_as_number,
                cpt.concept_name as concept_name
                from
                etl_dataset_temp.measurement mmt
                inner join omop_cdm.concept cpt
                on cpt.concept_id = mmt.measurement_concept_id
                inner join """ + schemaName + """.cohort coh
                on coh.person_id = mmt.person_id
                where
                measurement_concept_id in (
                3047181	-- Lactate [Moles/volume] in Blood
                , 3013290	-- Carbon dioxide [Partial pressure] in Blood
                , 3024561	-- Albumin [Mass/volume] in Serum or Plasma
                , 3024629	-- Glucose [Mass/volume] in Urine by Test strip
                , 3008939	-- Band form neutrophils [#/volume] in Blood by Manual count
                , 3012501	-- Base excess in Blood by calculation
                , 3005456	-- Potassium [Moles/volume] in Blood
                , 3010421	-- pH of Blood
                , 3014576	-- Chloride [Moles/volume] in Serum or Plasma
                , 3031147	-- Carbon dioxide, total [Moles/volume] in Blood by calculation
                , 3024128	-- Bilirubin.total [Mass/volume] in Serum or Plasma
                , 3000905	-- Leukocytes [#/volume] in Blood by Automated count
                , 3016723	-- Creatinine [Mass/volume] in Serum or Plasma
                , 3022217	-- INR in Platelet poor plasma by Coagulation assay
                , 3019550	-- Sodium [Moles/volume] in Serum or Plasma
                , 3000285	-- Sodium [Moles/volume] in Blood
                , 3000963	-- Hemoglobin [Mass/volume] in Blood
                , 3000963	-- Hemoglobin [Mass/volume] in Blood
                , 3018672	-- pH of Body fluid
                , 3024929	-- Platelets [#/volume] in Blood by Automated count
                , 3013682	-- Urea nitrogen [Mass/volume] in Serum or Plasma
                , 3004501	-- Glucose [Mass/volume] in Serum or Plasma
                , 3018572	-- Chloride [Moles/volume] in Blood
                , 3027315	-- Oxygen [Partial pressure] in Blood
                , 3016293	-- Bicarbonate [Moles/volume] in Serum or Plasma
                , 3023103	-- Potassium [Moles/volume] in Serum or Plasma
                , 3037278	-- Anion gap 4 in Serum or Plasma
                , 3003282	-- Leukocytes [#/volume] in Blood by Manual count
                , 3023314	-- Hematocrit [Volume Fraction] of Blood by Automated count
                , 3013466	-- aPTT in Blood by Coagulation assay
                )
                and value_as_number is not null
                and (mmt.measurement_datetime > """ + anchorColumn + """ - interval '""" + str(labsBefore) + """ hour')
                and (mmt.measurement_datetime < """ + anchorColumn + """ + interval '""" + str(labsAfter) + """ hour')
            )
            , labs_stg_2 as
            (
            select
                micro_specimen_id,
                person_id,
                measurement_datetime,
                unit_source_value,
                value_as_number,
                concept_name,
                row_number() over (partition by person_id, concept_name order by measurement_datetime) as rn
            from labs_stg_1
            )
            select * from labs_stg_2
        ;
        """

    with con:
        with con.cursor() as cursor:
            cursor.execute(dropLabsQuery)
            cursor.execute(labsQuery)


def extractDeaths(con, schemaName):
    log.info("Extracting deaths to " + schemaName + ".deaths")
    dropMortalityQuery = """drop table if exists """ + schemaName + """.deaths cascade"""
    mortalityQuery = """
        create table """ + schemaName + """.deaths as
        select
        coh.micro_specimen_id as micro_specimen_id
        , coh.person_id as person_id
        , dth.death_datetime
        from
        """ + schemaName + """.cohort coh
        left join omop_cdm.death dth
        on dth.person_id = coh.person_id
        ;
        """
    with con:
        with con.cursor() as cursor:
            cursor.execute(dropMortalityQuery)
            cursor.execute(mortalityQuery)


def extractEhr(con, vitalsBefore = 48, vitalsAfter = 48, labsBefore = 72, labsAfter = 72, anchor = "micro"):
    schemaName = anchor + "_vb_" + str(vitalsBefore) + "_va_" + str(vitalsAfter) + "_lb_" + str(labsBefore) + "_la_" + str(labsAfter)
    log.info("Schema Name: " + schemaName)
    dropSchema(con = con, schemaName = schemaName)
    createSchema(con = con, schemaName = schemaName)
    extractCohort(con = con, schemaName = schemaName)
    extractStatics(con=con, schemaName=schemaName)
    extractVitals(con=con, schemaName=schemaName, vitalsBefore = vitalsBefore, vitalsAfter = vitalsAfter, anchor = anchor)
    extractLabs(con=con, schemaName=schemaName, labsBefore = labsBefore, labsAfter = labsAfter, anchor = anchor)
    extractDeaths(con=con, schemaName=schemaName)


if __name__ == "__main__":
    extractEhr(vitalsBefore = 48, vitalsAfter = 48, labsBefore = 72, labsAfter = 72)
