import csv
import math
import time
import json
import numpy as np
from beautifultable import BeautifulTable
from pymongo import MongoClient

class Helper:
    def __init__(self,
                 db_name="WorkerSet100K",
                 collection_name="workers",
                 configuration="transparent",
                 N=50,
                 f=None,
                 selected=None,
                 k=None):
        """
        Initializes helper.
        :param db_name: name of the database.
        :param collection_name: name of the collections.
        :param configuration: can be either 'transparent', 'opaque_process', 'opaque_dataset'.
        :param N: number of workers to select.
        """
        self.db_name = db_name
        self.configuration = configuration
        self.limit = N
        self.collection = self.__get_collection(db_name, collection_name)
        self.k = k
        self.selected = selected
        self.f = f

    @staticmethod
    def __get_collection(db_name, collection_name):
        client = MongoClient()
        db = client[db_name]
        return db[collection_name]

    def __retrieve_simulated_dataset(self):
        def convert_to_ranges(worker, year_of_birth=True, years_of_experience=True):
            if year_of_birth:
                # YearOfBirth are combined categorically on a 10-year basis,
                # i.e. 1993 will be considered in the 1990-1999 range
                worker["YearOfBirth"] = int(worker["YearOfBirth"] / 10) * 10

            if years_of_experience:
                # YearsOfExperience are combined categorically on a 5-year basis,
                # i.e. 13 will be considered in the 10-14 range
                worker["YearsOfExperience"] = math.floor(worker["YearsOfExperience"] / 5) * 5

            return worker

        documents = []
        if self.configuration != 'opaque_dataset':
            for w in self.collection.find().limit(self.limit):
                w = convert_to_ranges(w)
                documents.append(w)
        else:
            with open('./datasets/simulated/opaque_dataset/' + str(self.limit) + '/' + str(self.k) + '.csv', mode='r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                for row in rows:
                    w = dict(row)
                    yob = True
                    yoe = True
                    w["LanguageTest"] = int(w["LanguageTest"])
                    w["ApprovalRate"] = int(w["ApprovalRate"])
                    if w["YearOfBirth"][:1] in ['*', '[']:
                        yob = False
                    if w["YearsOfExperience"][:1] in ['*', '[']:
                        yoe = False
                    w = convert_to_ranges(w, year_of_birth=yob, years_of_experience=yoe)
                    documents.append(w)
        return documents

    def get_documents(self):
        if self.db_name.startswith('WorkerSet'):
            return self.__retrieve_simulated_dataset()

        documents = []
        for w in self.collection.find().limit(self.limit):
            documents.append(w)
        return documents

    @staticmethod
    def __get_simulated_dataset_attributes_list(worker):
        attributes = {}

        for key in worker:
            attributes[key] = []

        # Remove non-attributes from attributes dictionary. This includes qualifications such as LanguageTest and
        # ApprovalRate
        attributes.pop("_id")
        attributes.pop("id")
        attributes.pop("LanguageTest")
        attributes.pop("ApprovalRate")

        return attributes


    def get_attributes(self, documents):
        worker = documents[0]

        if self.db_name.startswith('WorkerSet'):
            attributes = self.__get_simulated_dataset_attributes_list(worker)
        else:
            raise RuntimeError('Function that handles attributes is not specified for the dataset provided.')

        # Add different attribute values under attribute name
        for i in documents:
            for j in attributes:
                # add new values of the attribute to the list values
                if i[j] not in attributes[j]:
                    attributes[j].append(i[j])

        if self.configuration == 'opaque_dataset':
            for j in attributes.copy():
                if len(attributes[j]) <= 1:
                    attributes.pop(j)

        return attributes

    def build_tables(self, name, values, time_values, functions=None, percentages=None):
        """

        :param name:
        :param percentages:
        :param functions:
        :param time_values:
        :param values:
        :return:
        """

        def build_table(title, values):
            b_table = BeautifulTable(max_width=200)
            t_headers = [title]
            if self.configuration == 'opaque_process':
                for key in percentages:
                    t_headers.append(str(key) + '%')
            else:
                for key in functions:
                    t_headers.append('f' + str(key))
            b_table.column_headers = t_headers
            for i in range(len(values)):
                b_table.append_row(values[i][:])

            return b_table

        table = build_table(name, values)
        table.numeric_precision = 4

        print(name)
        print(table)

        timetable = build_table('TIMINGS', time_values)
        print('-----------------------')
        print('TIMINGS')
        print(timetable)

        return str(table), str(timetable)

    @staticmethod
    def run_algorithm(algorithm, method, num_of_runs=1):
        """

        :param algorithm:
        :param method:
        :param num_of_runs:
        :return:
        """
        value_per_run = []
        time_per_run = []
        for i in range(num_of_runs):
            start = time.time()
            value_per_run.append(algorithm.metric(method()))
            end = time.time()
            time_per_run.append(end - start)

        return np.mean(value_per_run), np.mean(time_per_run)

    def run_experiments(self, quantify_disparity_metric, workers, attributes, functions=None, percentages=None,
                        bins='preset', criterion='avg', normalize=True, scaling='standardization'):
        """

        :param normalize:
        :param scaling:
        :param criterion:
        :param quantify_disparity_metric:
        :param workers:
        :param attributes:
        :param functions:
        :param percentages:
        :param bins:
        :return:
        """
        if self.configuration == 'opaque_process':
            variants = percentages
        else:
            variants = functions

        all_values = [
            ["unbalanced"],
            ["r-unbalanced"],
            ["balanced"],
            ["r-balanced"],
            ["exhaustive"]
        ]

        all_time_values = [
            ["unbalanced"],
            ["r-unbalanced"],
            ["balanced"],
            ["r-balanced"],
            ["exhaustive"]
        ]

        name = 'undefined'
        for key in variants:
            if quantify_disparity_metric.__name__ == 'KL':
                name = self.db_name + '-KL-' + self.configuration + '-scaling-' + scaling + '-workers-' + str(self.limit)
                quantify_disparity = quantify_disparity_metric(workers, attributes,
                                                               configuration=self.configuration,
                                                               f=variants[key],
                                                               selected=variants[key],
                                                               bins=bins,
                                                               scaling=scaling)
            else:
                name = self.db_name + '-EMD-' + self.configuration + '-bins-' + bins + '-normalize-' + str(normalize) + '-criterion-' \
                       + criterion + str(self.limit)
                quantify_disparity = quantify_disparity_metric(workers, attributes,
                                                               configuration=self.configuration,
                                                               f=variants[key],
                                                               selected=variants[key],
                                                               bins=bins, normalize=normalize, criterion=criterion)

            methods = [
                quantify_disparity.unbalanced,
                quantify_disparity.random_unbalanced,
                quantify_disparity.balanced,
                quantify_disparity.random_balanced,
                quantify_disparity.exhaustive
            ]

            for i in range(len(methods)):
                num_of_times = 5 if i in [1, 3] else 1
                value, exec_time = self.run_algorithm(quantify_disparity, methods[i], num_of_times)
                all_values[i].append(value)
                all_time_values[i].append(exec_time)

        return name, all_values, all_time_values
