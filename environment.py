import copy

import numpy as np
from gym import spaces
from interval import interval


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
        assert self.n == len(item)
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


class ObstacleMaze:
    """
    Maze that contains obstacles in the form of walls. The agent starts in position (0,0)
    and navigates to the exit while evading walls. The states are of the format (x, y, v_x, v_y)
    with x, y the agent's position coordinates and v_x, v_y the x-axis and y-axis velocities.
    """
    # Actions
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

    def __init__(self, n, start, exit, walls, keys=[]):
        """
        Constructor for the grid

        :param n: Grid length
        :param exit: Grid exit
        :param walls: Each wall is in a rectangular is a tuple (x_left,
        x_right,y_bottom,y_top) denoting starting and end positions of the wall
        """

        self.n = n
        # Define action and observation space
        self.n_actions = 4
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=np.array([0, 0, -1, -1]), high=np.array([n, n, 1, 1]))
        self.start = start
        self.exit = exit
        self.walls = walls
        self.keys = keys
        self.gathered_keys = []
        # Initialize the state
        self.state = [self.start[0], self.start[1], 0, 0]

    def reset(self):
        # Restart the agent's position
        self.state = [self.start[0], self.start[1], 0, 0]
        state = self.state
        return state

    def violation(self, state, next_state, wall):
        """
        Checks if a point in the trajectory violates the collision condition by computing a possible point of
        collision. The trajectory is of the form (x', y') = (x+v_x*t, y+v_y*t)
        """
        x = state[0]
        y = state[1]
        v_x = next_state[2]
        v_y = next_state[3]
        base = interval[0, 1]
        x_interval = interval()
        y_interval = interval()
        if v_x > 0:
            l_x = (wall[0] - x) / v_x
            u_x = (wall[1] - x) / v_x
            x_interval = interval[l_x, u_x]
        elif v_x < 0:
            u_x = (wall[0] - x) / v_x
            l_x = (wall[1] - x) / v_x
            x_interval = interval[l_x, u_x]
        else:
            if wall[0] < x < wall[1]: x_interval = base

        if v_y > 0:
            l_y = (wall[2] - y) / v_y
            u_y = (wall[3] - y) / v_y
            y_interval = interval[l_y, u_y]
        elif v_y < 0:
            u_y = (wall[2] - y) / v_y
            l_y = (wall[3] - y) / v_y
            y_interval = interval[l_y, u_y]
        else:
            if wall[2] < y < wall[3]: y_interval = base

        t = base & x_interval & y_interval
        return t == interval(), t

    def possible(self, state, next_state):
        """
        Checks if it is possible to go from actual state to pos

        :param next_state: next calculated position
        :return: boolean
        """
        allow = True
        col_x = -1
        col_y = -1

        # Account for grid boundaries
        if next_state[0] < 0:
            allow = False
            col_x = 0
        if next_state[1] < 0:
            allow = False
            col_y = 0
        if next_state[0] > self.n:
            allow = False
            col_x = self.n
        if next_state[1] > self.n:
            allow = False
            col_y = self.n
        if not allow:
            col_x = (col_x == -1) * next_state[0] + (col_x != -1) * col_x
            col_y = (col_y == -1) * next_state[1] + (col_y != -1) * col_y
            return allow, col_x, col_y

        # Check if collision happens with walls
        for wall in self.walls:
            allow, t = self.violation(state, next_state, wall)
            if not allow:
                v_x = next_state[2]
                v_y = next_state[3]
                t = t[0].inf
                col_x = state[0] + v_x * t
                col_y = state[1] + v_y * t

                return allow, col_x, col_y

        return allow, col_x, col_y

    def step(self, action):
        """
        Compute the effect of an action after executing action

        :param action: action performed
        :return: next_state, reward, termination signal
        """

        v_x = self.state[2]
        v_y = self.state[3]
        # Check for collisions with walls
        if action == self.LEFT:
            v_x = max(-1, v_x - 0.1)
            v_y = (v_y > 0) * max(0, (v_y - 0.05)) + (v_y < 0) * max(0, (v_y + 0.05))
            move, col_x, col_y = self.possible(self.state, (self.state[0] + v_x, self.state[1] + v_y, v_x, v_y))
            if move:
                self.state = [self.state[0] + v_x, self.state[1] + v_y, v_x, v_y]
            elif col_x != -1 or col_y != -1:
                self.state = [col_x, col_y, 0, 0]

        if action == self.RIGHT:
            v_x = min(1, v_x + 0.1)
            v_y = (v_y > 0) * max(0, (v_y - 0.05)) + (v_y < 0) * max(0, (v_y + 0.05))
            move, col_x, col_y = self.possible(self.state, (self.state[0] + v_x, self.state[1] + v_y, v_x, v_y))
            if move:
                self.state = [self.state[0] + v_x, self.state[1] + v_y, v_x, v_y]
            elif col_x != -1 or col_y != -1:
                self.state = [col_x, col_y, 0, 0]

        if action == self.UP:
            v_y = min(1, v_y + 0.1)
            v_x = (v_x > 0) * max(0, (v_x - 0.05)) + (v_x < 0) * max(0, (v_x + 0.05))
            move, col_x, col_y = self.possible(self.state, (self.state[0] + v_x, self.state[1] + v_y, v_x, v_y))
            if move:
                self.state = [self.state[0] + v_x, self.state[1] + v_y, v_x, v_y]
            elif col_x != -1 or col_y != -1:
                self.state = [col_x, col_y, 0, 0]

        if action == self.DOWN:
            v_y = max(-1, v_y - 0.1)
            v_x = (v_x > 0) * max(0, (v_x - 0.05)) + (v_x < 0) * max(0, (v_x + 0.05))
            move, col_x, col_y = self.possible(self.state, (self.state[0] + v_x, self.state[1] + v_y, v_x, v_y))
            if move:
                self.state = [self.state[0] + v_x, self.state[1] + v_y, v_x, v_y]
            elif col_x != -1 or col_y != -1:
                self.state = [col_x, col_y, 0, 0]

        # Account for boundaries of the grid
        # self.state = tuple(np.clip(self.state, [0, 0, -1, -1], [self.n, self.n, 1, 1]))
        # if (self.state[0] <= 0) or (self.state[0] >= self.n):
        #     self.state = tuple(np.clip(self.state, [0, 0, -1, -1], [self.n, self.n, 1, 1]))

        if self.keys:
            for key in self.keys:
                if (key[0] - 0.5 < self.state[0] < key[0] + 0.5) and (
                        key[1] - 0.5 < self.state[1] < key[1] + 0.5):
                    self.keys.remove(key)

        if (self.exit[0] - 0.5 < self.state[0] < self.exit[0] + 0.5) and (
                self.exit[1] - 0.5 < self.state[1] < self.exit[1] + 0.5) and not self.keys:
            reward = 10
            done = True
        else:
            reward = 0
            done = False
        info = {}
        state = self.state

        return state, reward, done, info
