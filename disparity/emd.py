from pyemd import emd_samples
import random

from disparity.disparity import QuantifyingDisparity


class EMD(QuantifyingDisparity):
    def __init__(self, workers, attributes, configuration="transparent", normalize=True, f=None, selected=0.1,
                 bins="preset", criterion='avg'):
        """
        Initializes an EMD instance.
        :param workers: list, a list of workers dicts
        :param attributes: dict, attributes and their values. For example, {'Gender': ['Male', 'Female']}
        :param configuration: string, can be one of [transparent, opaque_process, opaque_dataset].
        :param f: list, scoring function parameters. For now, f is expected to have a length of 2.
        :param selected: float, must be between 0 and 1. Percentage of workers who are accepted. Used when configuration
               is opaque_process.
        :param bins: string, can be one of [preset, auto]
        :param normalize: bool, if true histograms will be normalized before calculating EMD values
        :param criterion: string, must be one of [avg, max, min]
        """
        super().__init__(workers, attributes, configuration, f, selected, bins)
        assert type(normalize) is bool, "normalized must be a boolean"
        self.normalize = normalize

        assert criterion in ['avg', 'max', 'min'], "criteria must be one of [avg, max, min], was " + str(criterion) + " instead"
        self.criterion = criterion

        print('RUNNING EMD with the following parameters:')
        print('Norm', normalize)
        print('f', f)
        print('configuration', configuration)
        print('criteria', criterion)
        print('bins', bins)

    def metric(self, partitions, siblings=None):
        if self.criterion == 'avg':
            return self.__avg_emd(partitions, siblings)
        elif self.criterion == 'min':
            return self.__min_emd(partitions, siblings)
        else:
            return self.__max_emd(partitions, siblings)

    def balanced(self, random_attribute=False):
        """
        Generates a partitioning of the workers in a greedy manner using the EMD of the worker partitions. It iteratively
        keeps trying to split the workers using the other attributes in the same manner and only stops whenever the
        average EMD achieved by the current partitioning is greater than that of the next candidate partitioning.
        :return: list of partitions of workers
        """
        attributes = self.original_attributes.copy()
        a = self.__worst_attribute(self.workers, attributes, random_attribute=random_attribute)
        del attributes[a]
        current = self.split(self.workers, a)
        current_max = self.metric(current)

        while len(attributes) > 0:
            a = self.__worst_attribute(current, attributes, random_attribute=random_attribute)
            del attributes[a]
            children = self.split(current, a)
            children_max = self.metric(children)
            if current_max >= children_max:
                break
            else:
                current = children
                current_max = children_max
        return current

    def unbalanced(self, random_attribute=False):
        """
        Generates a partitioning of the workers in a non-homogenous manner by locally deciding for each partition
        whether to further split it or not (i.e., resulting in a unbalanced partitioning tree). It decides whether
        or not to split a given partition by comparing the average EMD of that partition with its siblings to that of
        its children with its siblings.
        :return: list of partitions of workers
        """
        attributes = self.original_attributes.copy()
        a = self.__worst_attribute(self.workers, attributes, random_attribute=random_attribute)

        del attributes[a]
        current = self.split(self.workers, a)
        output = []

        # used for retrieving the name of the
        for i in current:
            siblings = current.copy()
            # Remove current partition from the list of partitions
            siblings.remove(i)
            partitions = self.__unbalanced_recursive([i], siblings, attributes,
                                                     random_attribute=random_attribute)
            for j in range(len(partitions)):
                output.append(partitions[j])

        return output

    def __unbalanced_recursive(self, current, siblings, A, output=None, random_attribute=False):
        """

        :param current:
        :param siblings:
        :param A:
        :param output:
        :param random_attribute:
        :return:
        """
        if output is None:
            output = []

        attributes = A.copy()

        if len(attributes) == 0:
            output.append(current[0])
        else:
            current_max = self.metric(current, siblings)
            a = self.__worst_attribute(current, attributes, random_attribute=random_attribute)
            del attributes[a]
            children = self.split(current, a)
            children_max = self.metric(children, siblings)
            if current_max >= children_max:
                output.append(current[0])
            else:
                for i in children:
                    siblings = children.copy()
                    # Remove current partition from the list of partitions
                    siblings.remove(i)
                    self.__unbalanced_recursive([i], siblings, attributes,
                                                output=output,
                                                random_attribute=random_attribute)
        return output

    def __calculate_emd(self, first_partition, second_partition):
        """
        Calculates the earth mover's distance between two partitions. The underlying calculations are done using
        https://github.com/wmayner/pyemd library. Euclidean distance is used by default.
        :param first_partition: list of workers
        :param second_partition: list of workers
        :return: emd value
        """
        f_values = [[]]
        for worker in first_partition:
            f_values[0].append(worker["Accepted"])
        f_values.append([])
        for worker in second_partition:
            f_values[1].append(worker["Accepted"])

        return emd_samples(f_values[0], f_values[1], normalized=self.normalize, bins=self.bins)

    def __worst_attribute(self, partition, attributes, random_attribute=False):
        """
        Finds the worst attribute in a given partition. The worst attribute is the one that when splitted on,
        the resulting partitions exhibit the highest average EMD value. If random_attribute is true, returns a random
        attribute as the worst.
        :param partition:
        :param attributes:
        :param random_attribute:
        :return:
        """
        if random_attribute:
            return random.choice(list(attributes.keys()))
        maximum = float('-inf')
        worst = None
        if len(attributes) > 0:
            for a in attributes:
                new_partitions = self.split(partition, a)
                emd = self.metric(new_partitions)
                if maximum <= emd:
                    maximum = emd
                    worst = a
        return worst

    def __avg_emd(self, partitions, siblings=None):
        """
        Finds the average EMD value between all partitions. If siblings is passed, it will find the average EMD between
        the supplied partitions and their siblings.
        :param partitions: list of partitions
        :param siblings: list of sibling partitions
        :return: average emd value
        """
        # in case of balanced, compare children with each other
        if not siblings:
            siblings = partitions
        total_sum_emd = 0
        count = 0

        for p in partitions:
            for q in siblings:
                if p != q:
                    emd = self.__calculate_emd(p, q)
                    total_sum_emd += emd
                    count += 1
        avg = total_sum_emd / count if count != 0 else 0
        return avg

    def __max_emd(self, partitions, siblings=None):
        """
        Finds the maximum EMD value between partitions. If siblings is passed, it will find the maximum EMD between
        the supplied partitions and their siblings.
        :param partitions: list of partitions
        :param siblings: list of sibling partitions
        :return: maximum emd value
        """
        # in case of balanced, compare children with each other
        if not siblings:
            siblings = partitions
        max_emd = float('-inf')

        for p in partitions:
            for q in siblings:
                if p != q:
                    emd = self.__calculate_emd(p, q)
                    if emd >= max_emd:
                        max_emd = emd
        return max_emd

    def __min_emd(self, partitions, siblings=None):
        """
        Finds the minimum EMD value between partitions. If siblings is passed, it will find the minimum EMD between
        the supplied partitions and their siblings.
        :param partitions: list of partitions
        :param siblings: list of sibling partitions
        :return: minimum emd value
        """
        # in case of balanced, compare children with each other
        if not siblings:
            siblings = partitions
        min_emd = float('+inf')

        for p in partitions:
            for q in siblings:
                if p != q:
                    emd = self.__calculate_emd(p, q)
                    if emd <= min_emd:
                        min_emd = emd
        return min_emd
