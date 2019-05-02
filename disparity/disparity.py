from abc import abstractmethod, ABCMeta
import numpy as np
import copy

CONFIGURATIONS = ['transparent', 'opaque_dataset', 'opaque_process']


class QuantifyingDisparity(metaclass=ABCMeta):
    def __init__(self, workers, attributes, configuration="transparent", f=None, selected=0.1, bins="preset"):
        """
        Initializes a QuantifyingDisparity instance.
        :param workers: list, a list of workers dicts
        :param attributes: dict, attributes and their values. For example, {'Gender': ['Male', 'Female']}
        :param configuration: string, can be one of [transparent, opaque_process, opaque_dataset].
        :param f: list, scoring function parameters. For now, f is expected to have a length of 2.
        :param selected: float, must be between 0 and 1. Percentage of workers who are accepted. Used when configuration
               is opaque_process.
        ":param bins: string, can be one of [preset, auto]
        """
        assert configuration in CONFIGURATIONS, "configuration must be one of [transparent, opaque_process, " \
                                                "opaque_dataset] "
        self.configuration = configuration

        assert type(attributes) is dict, "attributes must be a dictionary"
        self.original_attributes = dict(attributes)

        if configuration in ['transparent', 'opaque_dataset']:
            pass
            # assert (type(f) is list and len(f) == 2) or type(f) is int, "f must be a list of length 2 or an integer"
        else:
            assert 0 <= selected <= 1, "selected must be a float between 0 and 1"

        assert type(workers) is list and type(workers[0]) is dict, "workers must be a list of dicts"

        # adds Accepted attribute to workers
        self.workers = [self.__set_task_qualification(copy.deepcopy(workers), f, selected)]

        assert 'Accepted' in self.workers[0][0], "Task qualification function must set an Accepted attribute in every " \
                                                 "worker. "

        assert bins in ['auto', 'preset'], "bins must be one of [auto, preset]"
        if bins != 'auto':
            if self.configuration == 'opaque_process':
                bins = [0, 0.5, 1]
            else:
                bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.bins = bins

    def __str__(self):
        return str(self.__class__.__name__) + ' instance with the following parameters: \n' + \
               'Number of workers: ' + str(len(self.workers[0])) + '\n' + \
               'Attributes: ' + str([attribute + '(' + str(len(self.original_attributes[attribute])) + ')' for attribute in self.original_attributes]) + '\n' + \
               'Configuration: ' + self.configuration + '\n' + \
               'Sample worker: ' + str(self.workers[0][0])

    def __set_task_qualification(self, workers, f, selected):
        """
        Method that sets the task qualification decision of a certain worker. If the configuration is opaque_process,
        the value will be either 0 or 1, else it would be the value of function f.
        :param workers: list of workers objects
        :param selected: float that represents the percentage of workers who are qualified.
        :return: list of workers with 'Accepted' property appended to every one of them
        """
        list_of_valid_workers = []

        # SIMULATED DATASET FUNCTIONS
        for worker in workers:
            if self.configuration == "opaque_process":
                if selected == 'g':
                    if worker['Gender'] == 'Male':
                        worker['Accepted'] = 1
                    else:
                        worker['Accepted'] = 0
                elif selected == 'gc':
                    if worker['Gender'] == 'Male':
                        if worker['Country'] == 'India':
                            worker['Accepted'] = 1
                        elif worker['Country'] == 'America':
                            worker['Accepted'] = 0
                        else:
                            worker['Accepted'] = 1
                    else:
                        if worker['Country'] == 'India':
                            worker['Accepted'] = 0
                        elif worker['Country'] == 'America':
                            worker['Accepted'] = 1
                        else:
                            worker['Accepted'] = 0
                else:
                    worker['Accepted'] = 0 if np.random.uniform() <= (1 - selected) else 1
            else:
                if type(f) is list:
                    worker['Accepted'] = worker["LanguageTest"] / 100 * f[0] + worker["ApprovalRate"] / 100 * \
                                         f[1]
                elif f == '6':
                    if worker['Gender'] == 'Male':
                        worker['Accepted'] = np.random.uniform(low=0.8, high=1)
                    else:
                        worker['Accepted'] = np.random.uniform(low=0, high=0.2)
                elif f == '7':
                    if worker['Gender'] == 'Male' and worker['Country'] == 'America':
                        worker['Accepted'] = np.random.uniform(low=0.8, high=1)
                    elif worker['Gender'] == 'Female' and worker['Country'] == 'America':
                        worker['Accepted'] = np.random.uniform(low=0, high=0.2)
                    elif worker['Country'] == 'India':
                        worker['Accepted'] = np.random.uniform(low=0.5, high=0.7)
                    elif worker['Gender'] == 'Female':
                        worker['Accepted'] = np.random.uniform(low=0.8, high=1)
                    else:
                        worker['Accepted'] = np.random.uniform(low=0, high=0.2)
            list_of_valid_workers.append(worker)
        return list_of_valid_workers

    def split(self, partitions, attribute):
        """
        Splits a list of partitions based on the passed attribute.
        :param partitions:
        :param attribute:
        :return: list of partitions
        """
        new_set = []
        for partition in partitions:
            for k in self.original_attributes[attribute]:
                workers_with_attribute = list(filter(lambda worker: worker[attribute] == k, partition))

                if len(workers_with_attribute) != 0:
                    new_set.append(workers_with_attribute)
        return new_set

    @abstractmethod
    def metric(self, partitions):
        """
        Calculates the disparity of a given set of partitions. The returned value is based upon the approach used.
        :param partitions: list of partitions
        :return: disparity quantity
        """
        raise NotImplementedError

    def exhaustive(self):
        """
        Splits the workers on all attributes.
        :return: list of partitions
        """
        workers = self.workers.copy()
        for i in self.original_attributes:
            workers = self.split(workers, i)

        return workers

    @abstractmethod
    def balanced(self, random_attribute=False):
        """
        Generates a partitioning of the workers in a greedy manner using the EMD of the worker partitions. It iteratively
        keeps trying to split the workers using the other attributes in the same manner and only stops whenever the
        average EMD achieved by the current partitioning is greater than that of the next candidate partitioning.
        :return: list of partitions of workers
        """
        raise NotImplementedError

    def random_balanced(self):
        """
        Runs the balanced algorithm but with the worst algorithm being selected randomly.
        :return: list of partitions of workers
        """
        return self.balanced(random_attribute=True)

    @abstractmethod
    def unbalanced(self, random_attribute=False):
        """
        Generates a partitioning of the workers in a non-homogenous manner by locally deciding for each partition
        whether to further split it or not (i.e., resulting in a unbalanced partitioning tree). It decides whether
        or not to split a given partition by comparing the average EMD of that partition with its siblings to that of
        its children with its siblings.
        :return: list of partitions of workers
        """
        raise NotImplementedError

    def random_unbalanced(self):
        """
        Runs the unbalanced algorithm but with the worst algorithm being selected randomly.
        :return:
        """
        return self.unbalanced(random_attribute=True)
