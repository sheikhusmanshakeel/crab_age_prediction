import tabula
from tika import parser
import pandas as pd
import logging
import argparse
import os
import sys


logger = logging.getLogger('crabdata')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

'''
Assumptions:
    1. PDF files will arrive with the same overall structure i.e same number and position of columns
    2. They will also have the first part available as a single block so that only the last column needs to be individually parsed
    

'''


class CrabPDFParser:
    def __init__(self, source_location, destination_location=None):
        self.source_location = source_location
        self.destination_location = destination_location
        logger.debug("CrabPDF called")

    def __intTryParse(self, value):
        try:
            return int(value), True
        except ValueError:
            return value, False

    def extract_raw_features(self, frame_list):
        raw_features = pd.concat(frame_list)
        raw_features.drop([8], axis=1, inplace=True)
        raw_features.columns = ["sex", "length", "diameter", "height", "weight",
                                "shucked_weight", "viscera_weight", "shell_weight"]
        return raw_features.reset_index(drop=True)

    def get_index_of_age_variable(self, lines):
        index_of_age_variable = -1
        for c in range(len(lines)):
            if lines[c].strip() == "Age":
                index_of_age_variable = c
        return index_of_age_variable

    def extract_age(self, lines, length_of_features, index_of_age_variable):
        age_variable = 0 * [length_of_features]

        for c in range(index_of_age_variable, len(lines)):
            value, success = self.__intTryParse(lines[c].strip())
            if success:
                age_variable.append(value)
        return age_variable

    def process_raw_features(self):
        frame_list = tabula.read_pdf(self.source_location, multiple_tables="True", pages="all")
        raw_features = self.extract_raw_features(frame_list)
        return raw_features

    def process_age(self, length_of_features):
        raw = parser.from_file(self.source_location)
        lines = (raw['content'].strip().split('\n'))
        age_list = self.extract_age(lines, length_of_features, self.get_index_of_age_variable(lines))
        return age_list

    def process(self):
        raw_features = self.process_raw_features()
        age_list = self.process_age(len(raw_features))

        if len(age_list) != len(raw_features):
            logger.critical("Number of feature rows({0}) does not match number of rows for age ({1})".format(len(age_list),len(raw_features)))
            raise

        raw_features["age"] = pd.Series(age_list)
        if self.destination_location:
            raw_features.to_csv(self.destination_location, index=False)
        else:
            raw_features.to_csv("crab_data.csv", index=False)

        logger.info("PDF parsing finished successfully")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", help="complete location of the input pdf file", required=True)
    parser.add_argument("-d", "--destination_file", help="complete location where output csv file will be created", required=False)
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
    # input_file, output_file = "data.pdf", None
    if not os.path.exists(input_file):
        logger.critical("Please provide a valid input path")
        sys.exit(1)
    crab_data_parser = CrabPDFParser(input_file,output_file)
    crab_data_parser.process()

