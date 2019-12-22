import pytest
from mock import patch
import numpy as np

import crab_analyser.crab_pdf_parser_v2


class TestCrabPDFParser:
    @patch("crab_analyser.crab_pdf_parser_v2.CrabPDFParser.extract_age")
    @patch("crab_analyser.crab_pdf_parser_v2.CrabPDFParser.get_index_of_age_variable")
    def test_process_age(self, mock_get_index, mock_extract_age):
        # Arrange
        lines = ["Sheet 1"
            , "Page 1"
            , "Sex Length Diameter Height Weight Shucked Weight Viscera Weight Shell Weight"
            , "F 1.1512 1.175 0.4125 24.123 12.123 5 6"
            , "Age"
            , "Sheet 2"
            , "Page 2"
            , "5"]
        source_location = ""
        destination_location = ""
        age_list = [5]
        length_of_features = 1
        age_index = 3
        mock_extract_age.return_value = age_list
        mock_get_index.return_value = age_index
        parser = crab_analyser.crab_pdf_parser_v2.CrabPDFParser(source_location, destination_location)

        # Act
        ret_val = parser.process_age(length_of_features, lines)

        # Assert
        assert age_list == ret_val
        mock_get_index.assert_called_once_with(lines)
        mock_extract_age.assert_called_once_with(length_of_features, age_index)

    @patch("crab_analyser.crab_pdf_parser_v2.parser.from_file")
    def test_raw_pdf(self, mock_tika_from_file):
        # Arrange
        raw_file = {"status": "something",
                    "content": "\n\n\nSheet 1\nPage 1\nSex Length Diameter Height Weight Shucked Weight Viscera Weight Shell Weight"}
        mock_tika_from_file.return_value = raw_file
        raw_file = raw_file['content'].strip().split('\n')
        source_location = "file_input_location"
        destination_location = ""
        parser = crab_analyser.crab_pdf_parser_v2.CrabPDFParser(source_location, destination_location)

        # Act
        ret_val = parser.read_raw_pdf()

        # Assert
        assert ret_val == raw_file
        mock_tika_from_file.assert_called_once_with(source_location)

    def test_extract_age(self):
        # Arrange
        lines = ["Sheet 1"
            , "Page 1"
            , "Sex Length Diameter Height Weight Shucked Weight Viscera Weight Shell Weight"
            , "F 1.1512 1.175 0.4125 24.123 12.123 5 6"
            , "Age"
            , "Sheet 2"
            , "Page 2"
            , "5"]
        source_location = ""
        destination_location = ""
        age = [5]
        length_of_features = 1
        age_index = 4
        parser = crab_analyser.crab_pdf_parser_v2.CrabPDFParser(source_location, destination_location)

        ret_val = parser.extract_age(length_of_features, age_index, lines)

        assert ret_val == age
        assert len(ret_val) == 1

    test_data = [(["Sheet 1"
                      , "Page 1"
                      , "Sex Length Diameter Height Weight Shucked Weight Viscera Weight Shell Weight"
                      , "F 1.1512 1.175 0.4125 24.123 12.123 5 6"
                      , "Age"
                      , "Sheet 2"
                      , "Page 2"
                      , "5"], 4)
        ,
                 (["Sheet 1"
                      , "Page 1"
                      , "Sex Length Diameter Height Weight Shucked Weight Viscera Weight Shell Weight"
                      , "F 1.1512 1.175 0.4125 24.123 12.123 5 6"
                      , "F 1.1512 1.175 0.4125 24.123 12.123 5 6"
                      , "Age"
                      , "Sheet 2"
                      , "Page 2"
                      , "5"], 5)
        ,
                 (
                     ["Sheet 1"
                         , "Page 1"
                         , "Sex Length Diameter Height Weight Shucked Weight Viscera Weight Shell Weight"
                         , "F 1.1512 1.175 0.4125 24.123 12.123 5 6"
                         , "F 1.1512 1.175 0.4125 24.123 12.123 5 6"
                         , "Sheet 2"
                         , "Page 2"
                         , "5"], -1
                 )
                 ]

    @pytest.mark.parametrize("lines, expected_value", test_data)
    def test_get_index_of_age_variable(self, lines, expected_value):
        # Arrange
        source_location = ""
        destination_location = ""
        parser = crab_analyser.crab_pdf_parser_v2.CrabPDFParser(source_location, destination_location)

        # Act
        ret_val = parser.get_index_of_age_variable(lines)

        # Assert
        assert ret_val == expected_value

    test_data_vals = [(["F", "1.1512", "1.175", "0.4125", "24.123", "12.123", "5", "6"],
                       ("F", 1.1512, 1.175, 0.4125, 24.123, 12.123, 5, 6, False)),
                      (["F", "Gooood", "1.175", "0.4125", "24.123", "12.123", "5", "6"],
                       ("F", np.nan, 1.175, 0.4125, 24.123, 12.123, 5, 6, True)),
                      (["F", "1.1512", "Gooood", "0.4125", "24.123", "12.123", "5", "6"],
                       ("F", 1.1512, np.nan, 0.4125, 24.123, 12.123, 5, 6, True))

                      ]

    @pytest.mark.parametrize("vals, expected_value", test_data_vals)
    def test_get_converted_row(self, vals, expected_value):
        # Arrange
        source_location = ""
        destination_location = ""
        parser = crab_analyser.crab_pdf_parser_v2.CrabPDFParser(source_location, destination_location)

        # Act
        ret_val = parser.get_converted_row(vals)

        # Assert
        assert ret_val == expected_value
