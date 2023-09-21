# This file is to run an experiment to study the effect of standardisation

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
    dirPath = '/superbugai-data/yash/chapter_1/workspace/EHRQC/data/icd_cohort_test/'
