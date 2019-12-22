# Author: Sheikh Usman Shakeel
from crab_analyser.crab_pdf_parser_v2 import CrabPDFParser
from crab_analyser.crab_ml import CrabAgePredictor
import logging
import argparse
import os
import sys
import pandas as pd

# set up the logger
logger = logging.getLogger('crabdata')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def parse_args():
    """
    sets up command line parameters
    destination file only saves the clean crab data set

    :return:
    """
    # this can be extended to include output locations for other csv files in the crab_ml script but
    #     i couldn't implement and test it due to time constraints
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", help="complete location of the input pdf file", required=True)
    parser.add_argument("-d", "--destination_file", help="complete location where output csv file will be created",
                        required=False)
    args = parser.parse_args()
    input_file = None
    output_file_location = None
    if args.input_file:
        input_file = args.input_file
    if args.destination_file:
        output_file_location = args.destination_file

    return input_file, output_file_location


if __name__ == "__main__":
    input_file, output_file = parse_args()

    try:
        logger.info("main called")
        # input_file, output_file = "data.pdf", None
        if not output_file:
            output_file = "crab_data.csv"
        if not os.path.exists(input_file):
            logger.critical("please provide a valid input path")
            logger.critical("exiting execution")
            sys.exit(1)
        # parse the pdf file and save the csv file
        crab_data_parser = CrabPDFParser(input_file, output_file)
        crab_data_parser.process()
        logger.info("data extraction complete")
        logger.info("starting crab age prediction")
        ml = CrabAgePredictor(pd.read_csv(output_file))
        ml.run()
        logger.info("crab age prediction finished")
        logger.info("main execution finished successfully")

    except:
        logger.critical("Exception occurred.", exc_info=True)
        sys.exit(1)


