"""
Benchmark tool for evaluating & comparing Neural Network Reachability Analysis 
algorithms 
"""
import copy
import json
import numpy as np
from interval import interval
from julia import Julia
from collections import defaultdict

jl = Julia(compiled_modules=False)


class ndInterval:
    """
    Class that creates arrays of intervals and extend interval methods across the array.
    """

    def __init__(self, n, inf=[], sup=[]):
        self.n = n
        self.inf = inf
        self.sup = sup
        if inf != [] and sup != []:
            self.interval = [interval[inf[i], sup[i]] for i in range(n)]
        else:
            self.interval = []

    def __contains__(self, item):
        # assert self.n == len(item)
        for i in range(self.n):
            if not item[i] in self.interval[i]:
                return False
        return True

    def volume(self):
        volume = 1
        for i in range(self.n):
            volume *= self.sup[i] - self.inf[i]

        return volume

    def adjacency(self, B):
        """Checks for adjacent intervals that can be merged"""
        counter = self.n
        dim = -1
        for i in range(self.n):
            if self.inf[i] == B.inf[i] and self.sup[i] == B.sup[i]:
                counter -= 1
            elif self.sup[i] == B.inf[i] or self.inf[i] == B.sup[i]:
                dim = i
        if counter == 1:
            return dim
        else:
            return -1

    def merge(self, B, dim):
        """Merges two interval vectors across an appropriate dimension"""
        C = ndInterval(self.n, [], [])
        for i in range(C.n):
            if i != dim:
                C.sup.append(self.sup[i])
                C.inf.append(self.inf[i])
            else:
                C.sup.append(self.sup[i] * (self.sup[i] >= B.sup[i]) + B.sup[i] * (self.sup[i] < B.sup[i]))
                C.inf.append(self.inf[i] * (self.inf[i] <= B.inf[i]) + B.inf[i] * (self.inf[i] > B.inf[i]))

        C.interval = [interval[C.inf[i], C.sup[i]] for i in range(C.n)]
        return C

    def search_merge(list):
        """Searches for intervals that can be merged together and merges them"""
        change = True
        n = len(list)
        while change and len(list) > 1:
            for A in list:
                for B in list:
                    dim = A.adjacency(B)
                    if dim > -1:
                        C = A.merge(B, dim)
                        change = True
                        list.remove(A)
                        list.remove(B)
                        list.append(C)
                        n = len(list)
                        break
                if A not in list:
                    continue
            if len(list) == n:
                change = False
        
        return list

    def split(self, dims, lows=[], ups=[], split_value=dict()):
        """Splits an interval across a dimension"""
        if not dims:
            return [self]
        if lows == [] or ups == []:
            lows = self.inf
            ups = self.sup
        if dims:
            d = dims[0]
            lows1 = copy.deepcopy(lows)
            ups1 = copy.deepcopy(ups)
            if d not in split_value.keys():
                ups1[d] = lows[d] + ((ups[d] - lows[d]) / 2)
            else:
                ups1[d] = split_value[d]
            partition1 = ndInterval(self.n, inf=lows1, sup=ups1)
            list1 = partition1.split(dims[1:], lows1, ups1)

            lows2 = copy.deepcopy(lows1)
            if d not in split_value.keys():
                lows2[d] = lows[d] + ((ups[d] - lows[d]) / 2)
            else:
                lows2[d] = split_value[d]
            ups2 = copy.deepcopy(ups)
            partition2 = ndInterval(self.n, inf=lows2, sup=ups2)
            list2 = partition2.split(dims[1:], lows2, ups2)

            return list1 + list2

    def complement(self, subinterval):
        """Computes the complement of a sub interval inside the original interval"""
        complement = []
        for v in range(self.n):
            inf1 = copy.copy(self.inf)
            sup1 = copy.copy(self.sup)
            sup1[v] = subinterval.inf[v]
            if sup1[v] > inf1[v]:
                int1 = ndInterval(self.n, inf=inf1, sup=sup1)
                complement.append(int1)

            inf2 = copy.copy(self.inf)
            inf2[v] = subinterval.sup[v]
            sup2 = copy.copy(self.sup)
            if sup2[v] > inf2[v]:
                int2 = ndInterval(self.n, inf=inf2, sup=sup2)
                complement.append(int2)

        ndInterval.search_merge(complement)
        return ndInterval.remove_duplicates(complement)

    def intersection(self, interval):
        intersection_inf = list(np.maximum(self.inf, interval.inf))
        intersection_sup = list(np.minimum(self.sup, interval.sup))

        # Empty intersection
        if max(np.array(intersection_inf) > np.array(intersection_sup)):
            return []
        else:
            return [intersection_inf, intersection_sup]

    def remove_duplicates(interval_list):
        """Takes a list of intervals and eliminates duplicate intersections"""
        for i in range(len(interval_list)):
            partition1 = interval_list[i]
            for j in range(i+1,len(interval_list)):
                partition2 = interval_list[j]
                intersection = partition1.intersection(partition2)
                if intersection:
                    new_inf = []
                    new_sup = []
                    for v in range(partition2.n):
                        if partition2.inf[v] < intersection[0][v]:
                            new_sup += [intersection[0][v]]
                        else:
                            new_sup += [partition2.sup[v]]
                        if partition2.sup[v] > intersection[1][v]:
                            new_inf += [intersection[1][v]]
                        else:
                            new_inf += [partition2.inf[v]]
                    interval_list[j] = ndInterval(partition2.n, new_inf, new_sup)


        return interval_list

class ReachabilityAnalysis:

    def __init__(self, alg, input, output, network):
        self.reach = None
        self.alg = alg
        self.n_input = input.n
        self.input = input
        self.n_output = output[0].n
        self.output = output
        self.network = network
        self.errors = []

    def coverage(self, safe):
        R = self.reach
        total_volume_output = 0
        total_volume_safe = 0
        coverage = 1
        output_intersections = []
        safe_intersections = []

        for j in range(len(safe)):
            intersection_volume = 1
            volume_output = 1
            volume_safe = 1
            for i in range(R.n):
                new_lower = max(R.inf[i], safe[j].inf[i])
                new_upper = min(R.sup[i], safe[j].sup[i])
                output_range = R.sup[i] - R.inf[i]
                safe_range = safe[j].sup[i] - safe[j].inf[i]

                # Empty intersection
                if (new_lower > new_upper):
                    output_intersections.append(0)
                    safe_intersections.append(0)
                    return 0, 0
                elif output_range < 1e-5:
                    continue
                else:
                    intersection_volume = intersection_volume * (new_upper - new_lower)
                    volume_output = volume_output * output_range
                    volume_safe = volume_safe * safe_range

            total_volume_output += volume_output
            total_volume_safe += volume_safe
            output_intersections.append(intersection_volume)

        # Compute the ratio of the volumes
        return np.sum(output_intersections) / total_volume_output, np.sum(output_intersections) / total_volume_safe

    def volume_ratio(self, input):
        ratio = 1
        for i in range(self.n_input):
            if (self.input.sup[i] - self.input.inf[i]) > 0:
                ratio *= (input.sup[i] - input.inf[i]) / (self.input.sup[i] - self.input.inf[i])
        return ratio

    def compute_error(self, input, R):
        true_output = ndInterval(len(R.sup))
        if input.inf[0] > 0:
            true_output.sup = [1] * len(R.sup)
            true_output.inf = [1] * len(R.inf)
        elif input.sup[0] <= 0:
            true_output.sup = [-1] * len(R.sup)
            true_output.inf = [-1] * len(R.inf)
        else:
            true_output.sup = [1] * len(R.sup)
            true_output.inf = [-1] * len(R.inf)

        error_high = (np.square(R.sup - true_output.sup)).mean(axis=0)
        error_low = (np.square(R.inf - true_output.inf)).mean(axis=0)
        return error_high + error_low

    def Reachability(self, unsafe_threshold, safe_threshold, input, output, depth=0, compute_error=False):
        jl.include('reachability/reachability.jl')

        input_low = input.inf
        input_high = input.sup
        output_low = output[0].inf
        output_high = output[0].sup
        R_low, R_high = jl.reachability(self.alg, input_low, input_high, output_low, output_high,
                                        self.network)
        R = ndInterval(self.n_output, inf=R_low, sup=R_high)
        self.reach = R

        if compute_error:
            error = self.compute_error(input, R)
            self.errors.append(error)

        c1, c2 = self.coverage(output)

        print("input: ", input_low, input_high)
        print("output: ", R_low, R_high)
        print("criteria: c1 = ", c1, " c2 = ", c2)

        if c1 >= safe_threshold:
            partitions = dict()
            partitions['reach'] = [copy.deepcopy(input)]
            partitions['reach_size'] = 1
            partitions['no_reach'] = []
            partitions['no_reach_size'] = 0
            print(input_low, input_high, "safe")
        elif c2 <= unsafe_threshold:
            partitions = dict()
            partitions['reach'] = []
            partitions['reach_size'] = 0
            partitions['no_reach'] = [copy.deepcopy(input)]
            partitions['no_reach_size'] = 1
            # print("unsafe")
        elif self.volume_ratio(input) > 0.05:
            # print("split")
            partitions = dict()
            depth += 1
            split_dim = [jl.compute_grad(input_low, input_high, self.network)]
            splits = input.split(split_dim)
            partition1, partition2 = splits[0], splits[1]

            partitions1 = self.Reachability(unsafe_threshold, safe_threshold, partition1, output, depth, compute_error)
            partitions2 = self.Reachability(unsafe_threshold, safe_threshold, partition2, output, depth, compute_error)

            partitions['reach'] = partitions1['reach'] + partitions2['reach']
            partitions['reach_size'] = partitions1['reach_size'] + partitions2['reach_size']
            partitions['no_reach'] = partitions1['no_reach'] + partitions2['no_reach']
            partitions['no_reach_size'] = partitions1['no_reach_size'] + partitions2['no_reach_size']
        else:
            partitions = dict()
            partitions['reach'] = []
            partitions['reach_size'] = 0
            partitions['no_reach'] = [copy.deepcopy(input)]
            partitions['no_reach_size'] = 1

        return partitions


def params_load(exp):
    with open('nnets/params_model' + str(exp) + '.json') as json_file:
        params = json.load(json_file)
    return params["n_inputs"], params["n_outputs"]


def write_stats(stats):
    with open('stats.json', 'w') as f:
        json.dump(stats, f)


def reachability_analysis(network, input, output, alg, t1=0.7, t2=0.1):
    R = ReachabilityAnalysis(alg, input, output, network)
    partitions = R.Reachability(t2, t1, R.input, R.output, compute_error=True)
    return partitions
