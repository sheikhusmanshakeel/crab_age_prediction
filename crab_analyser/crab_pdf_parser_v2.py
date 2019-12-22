# Author: Sheikh Usman Shakeel
import logging

import numpy as np
import pandas as pd
from tika import parser

'''
Assumptions:
    1. PDF files will arrive with the same overall structure i.e same number and position of columns
    2. They will also have the first part available as a single block so that only the last column needs to be individually parsed
    3. Since I am using Adobe REST-Api to read the data, it might be security efficient since they might keep a copy of the data


'''

logger = logging.getLogger('crabdata')

class CrabPDFParser:
    def __init__(self, source_location, destination_location):
        self.source_location = source_location
        self.destination_location = destination_location
        self.dirty_rows = []

        logger.debug("CrabPDF called")

    def intTryParse(self, value):
        try:
            return int(value), True
        except ValueError:
            return np.nan, False

    def floatTryParse(self, value):
        try:
            return float(value), True
        except ValueError:
            return np.nan, False

    def get_converted_row(self, vals):
        """
        parses a single data row and returns either floats, string (for sex variable) or np.nan
        also returns a boolean if the row was dirty i.e. some values were not float
        :param vals:
        :return:
        """
        sex = str(vals[0])
        success_count = 0
        length, success = self.floatTryParse(vals[1])
        if not success:
            success_count += 1
        diameter, success = self.floatTryParse(vals[2])
        if not success:
            success_count += 1
        height, success = self.floatTryParse(vals[3])
        if not success:
            success_count += 1
        weight, success = self.floatTryParse(vals[4])
        if not success:
            success_count += 1
        shucked_weight, success = self.floatTryParse(vals[5])
        if not success:
            success_count += 1
        viscera_weight, success = self.floatTryParse(vals[6])
        if not success:
            success_count += 1
        shell_weight, success = self.floatTryParse(vals[7])
        if not success:
            success_count += 1

        return sex, length, diameter, height, weight, shucked_weight, viscera_weight, shell_weight, (success_count != 0)

    def extract_raw_features(self, lines):
        """
        extracts the feature matrix from pdf file
        :return:
        """
        sex = []
        # i have randomly initialised the array to 2000 in order to assign memory up front and
        # save up on space and time (dynamic array resize)
        length = 0 * [2000]
        diameter = 0 * [2000]
        height = 0 * [2000]
        weight = 0 * [2000]
        shucked_weight = 0 * [2000]
        viscera_weight = 0 * [2000]
        shell_weight = 0 * [2000]
        for c in range(len(lines)):
            vals = lines[c].strip().split(' ')
            if 3 <= len(vals) <= 7 and not (vals.__contains__("Page") or vals.__contains__("Sheet")):
                # I also wanted to keep track of dirty data rows to report
                # these dirty rows are printed towards the end of this section
                self.dirty_rows.append(lines[c])
            elif len(vals) == 8:
                s, l, d, h, w, sw, vw, sh_w, is_dirty = self.get_converted_row(vals)
                if is_dirty:
                    self.dirty_rows.append(lines[c])
                sex.append(s)
                length.append(l)
                diameter.append(d)
                height.append(h)
                weight.append(w)
                shucked_weight.append(sw)
                viscera_weight.append(vw)
                shell_weight.append(sh_w)

        # create data frame from the collected lists
        # i know that the expected output column names are different. i kept them this way to make analysis easier
        # this could be easily changed by providing a mapper dict object to pandas rename function
        data_frame = pd.DataFrame({"sex": sex, "length": length, "diameter": diameter, "height": height,
                                   "weight": weight, "shucked_weight": shucked_weight, "viscera_weight": viscera_weight,
                                   "shell_weight": shell_weight})
        return data_frame

    def get_index_of_age_variable(self, lines):
        """
        get the index where age variable starts
        even though i could have set this variable in extract_raw_features function i wanted to follow
        good OOP principles where each function has does one thing and has a strict post-condition
        this is also a good technique for writing unit tests
        :return:
        """
        index_of_age_variable = -1
        for c in range(len(lines)):
            if lines[c].strip() == "Age":
                index_of_age_variable = c
        return index_of_age_variable

    def extract_age(self, length_of_features, index_of_age_variable, lines):
        """
        extracts the age variable
        :param lines: Raw lines from which to parse age variable
        :param length_of_features:
        :param index_of_age_variable:
        :return:
        """
        # since at this point we already know the number of features, we can initialise the array up front
        age_variable = 0 * [length_of_features]

        for c in range(index_of_age_variable, len(lines)):
            value, success = self.intTryParse(lines[c].strip())
            if success:
                age_variable.append(value)
        return age_variable

    def process_age(self, length_of_features, lines):
        """
        process and extract age variable
        :param lines:
        :param length_of_features:
        :return:
        """
        age_list = self.extract_age(length_of_features, self.get_index_of_age_variable(lines), lines)
        return age_list

    def read_raw_pdf(self):
        """
        uses adobe REST Api to return just the text of a PDF
        :return:
        """
        raw = parser.from_file(self.source_location)
        return (raw['content'].strip().split('\n'))

    def process(self):
        """
        main entry function for this class
        :return:
        """
        lines = self.read_raw_pdf()
        raw_features = self.extract_raw_features(lines)
        age_list = self.process_age(len(raw_features),lines)

        if len(age_list) != len(raw_features):
            logger.critical(
                "Number of feature rows({0}) does not match number of rows for age ({1})".format(len(age_list),
                                                                                                 len(raw_features)))
            raise

        raw_features["age"] = pd.Series(age_list)
        raw_features.to_csv(self.destination_location, index=False)
        logger.info("Number of dirty data rows: {0}".format(len(self.dirty_rows)))
        for d in self.dirty_rows:
            logger.debug(d)
        logger.info("PDF parsing finished successfully")



