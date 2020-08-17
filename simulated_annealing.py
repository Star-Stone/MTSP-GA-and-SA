import utils
import random
import math
import matplotlib.pyplot as plt

INF = 99999


class SA:
    def __init__(self, salesman_num, city_num, start_index, filename, distance_weight, balance_weight):
        self.T_begin = 200  # 初始温度
        self.T_end = 0.1  # 终止温度
        self.T = self.T_begin  # 过程中的温度，初始时候是T_begin
        self.T_list = []  # 退火过程中温度列表
        self.Lk = 300  # 每个温度下的迭代次数
        self.alpha = 0.95  # 温度衰减系数

        self.salesman_num = salesman_num  # 旅行商数量
        self.city_num = city_num  # 城市的数量
        self.start_index = start_index  # 起始点也是终点的序号
        self.solution_len = salesman_num + city_num - 2  # 解的长度
        self.dummy_points = [x for x in range(self.city_num + 1, self.city_num + self.salesman_num)]  # 虚点
        self.object_func = None  # 优化的目标函数

        self.filename = filename  # 读取文件名称
        self.graph_obj = utils.Graph(filename)  # 针对实验1 检查点：创建utils里面Graph对象
        self.china_cities = []  # 针对实验2 运输机航线

        self.per_iter_solution = []  # 每个温度下最优解
        # self.per_iter_solu_path = []  # 每个温度下最优解的路径组成
        self.all_per_iter_solution = []  # 记录每个温度下每代最优解变化情况

        self.best_solution = []  # 全局最优解
        # self.best_solution_path = []  # 全局最优解路径组成
        self.all_best_solution = []  # 记录每个温度下全局最优解

        self.swap_solu_prob = 0.1  # 执行交换产生新解概率
        self.reverse_solu_prob = 0.4  # 执行逆转产生新解的概率
        self.shift_solu_prob = 1 - self.reverse_solu_prob - self.swap_solu_prob  # 执行移位产生新解的概率

        self.distance_weight = distance_weight  # 总路程权重
        self.balance_weight = balance_weight  # 均衡度数权重

    def check_vertex_init(self):
        """
        针对实验1：检查点路线
        调用floyd算法得到任意两点的最短路径，随机产生一个解并赋值给全局最优解和每个温度下最优解
        Returns:

        """
        # 调用floyd算法得到任意两点的最短路径
        self.graph_obj.short_path_floyd()
        # 计算解的目标函数为check_vertex_obj_func
        self.object_func = self.check_vertex_obj_func  # check_vertex_obj_func
        # 注意：城市起点从1开始，而不是从0
        solution = [x for x in range(1, self.city_num + 1)]
        # 把起始点剔除
        solution.remove(self.start_index)
        # 多个旅行商，增加salesman_num-1个虚点
        solution.extend(self.dummy_points)
        random.shuffle(solution)
        # 初始化全局最优解和每个温度下最优解
        self.best_solution = self.per_iter_solution = solution
        return solution

    def china_city_init(self):
        """
        针对实验2：运输机路线
        初始化
        Returns:

        """
        # 读取文件，得到城市列表
        self.china_cities = utils.read_china_cities_coord(self.filename)
        # 计算解的目标函数为china_city_obj_func
        self.object_func = self.china_city_obj_func
        # 注意：城市起点从1开始，而不是从0
        solution = [x for x in range(1, self.city_num + 1)]
        # 把起始点剔除
        solution.remove(self.start_index)
        # 多个旅行商，增加salesman_num-1个虚点
        solution.extend(self.dummy_points)
        random.shuffle(solution)
        # 初始化全局最优解和每个温度下最优解
        self.best_solution = self.per_iter_solution = solution
        return solution

    def get_check_vertex_distance(self, solution):
        """
        针对实验1：检查点路线
        根据解decode，得到所有旅行商走的路线以及每条路线总路程
        Args:
            solution: 解

        Returns:
            all_routines：所有旅行商走的路线
            routines_dis：每条路线总路程组成列表
        """
        # 起始点5，城市9个，旅行商3，虚点10,11
        # [4, 6, 11, 9, 2, 1, 10, 7, 8, 3]
        tmp_solu = solution[:]
        # 将增加的虚点还原成起始点
        for i in range(self.solution_len):
            if solution[i] in self.dummy_points:
                tmp_solu[i] = self.start_index
        # 根据起始点把chrom分成多段
        one_routine = []  # 一个旅行商路线，可以为空
        all_routines = []  # 所有旅行商路线
        for v in tmp_solu:
            if v == self.start_index:
                all_routines.append(one_routine)
                one_routine = []
            elif v != self.start_index:
                one_routine.append(v)
        # 还有一次需要添加路线
        all_routines.append(one_routine)

        routines_dis = []  # 所有路径总距离组成的列表
        # 计算每一条路总的距离
        for r in all_routines:
            distance = 0
            # 有一个旅行商路线为空列表，即一个旅行商不出门
            if len(r) == 0:
                distance = INF
                routines_dis.append(distance)
            else:
                r_len = len(r)
                for i in range(r_len):
                    # 别忘了最后加上起始点到第一个点的距离
                    if i == 0:
                        distance += self.graph_obj.adj_matrix[self.start_index][r[i]]
                    if i + 1 < r_len:
                        distance += self.graph_obj.adj_matrix[r[i]][r[i + 1]]
                    # 最后一个顶点，下一站是起始点
                    elif i == r_len - 1:
                        distance += self.graph_obj.adj_matrix[r[i]][self.start_index]
                routines_dis.append(distance)

        return all_routines, routines_dis

    def check_vertex_obj_func(self, solution):
        """
        针对实验1：检查点路线
        计算解的目标函数值
        Args:
            solution: 解

        Returns:
            obj：目标函数值
        """
        all_routines, routines_dis = self.get_check_vertex_distance(solution)
        sum_path = sum(routines_dis)
        max_path = max(routines_dis)
        min_path = min(routines_dis)
        balance = (max_path - min_path) / max_path
        obj = self.distance_weight * sum_path + \
              self.balance_weight * balance

        return obj

    def get_china_city_distance(self, solution):
        """
        针对实验 2：运输机航线
        根据解decode，得到所有运输机走的路线以及每条路线总路程
        Args:
            solution: 解

        Returns:
            all_routines：所有运输机路线组成列表
            routines_dis：每每条路线总路程组成列表

        """
        # 起始点5，城市9个，旅行商3，虚点10,11
        # [4, 6, 11, 9, 2, 1, 10, 7, 8, 3]
        tmp_solu = solution[:]
        # 将增加的虚点还原成起始点
        for i in range(self.solution_len):
            if solution[i] in self.dummy_points:
                tmp_solu[i] = self.start_index
        # 根据起始点把chrom分成多段
        one_routine = []  # 一个旅行商路线，可以为空
        all_routines = []  # 所有旅行商路线
        for v in tmp_solu:
            if v == self.start_index:
                all_routines.append(one_routine)
                one_routine = []
            elif v != self.start_index:
                one_routine.append(v)
        # 还有一次需要添加路线
        all_routines.append(one_routine)

        routines_dis = []  # 所有路径总距离组成的列表
        # 计算每一条路总的距离
        for r in all_routines:
            distance = 0
            # 有一个旅行商路线为空列表，即一个旅行商不出门
            if len(r) == 0:
                distance = INF
                routines_dis.append(distance)
            else:
                r_len = len(r)
                for i in range(r_len):
                    # 别忘了最后加上起始点到第一个点的距离
                    if i == 0:
                        distance += utils.geo_distance(self.china_cities[self.start_index],
                                                       self.china_cities[r[i]])
                    if i + 1 < r_len:
                        distance += utils.geo_distance(self.china_cities[r[i]], self.china_cities[r[i + 1]])
                    # 最后一个顶点，下一站是起始点
                    elif i == r_len - 1:
                        distance += utils.geo_distance(self.china_cities[r[i]],
                                                       self.china_cities[self.start_index])
                routines_dis.append(distance)

        return all_routines, routines_dis

    def china_city_obj_func(self, solution):
        """
        针对实验 2：运输机航线
        计算解的目标函数值
        目标函数 Z = distance_weight*总路程 + balance_weight*均衡度
        均衡度 = (max(l)-min(l))/ max(l)
        Args:
            solution: 解

        Returns:
            obj：解的目标函数值
        """
        all_routines, routines_dis = self.get_china_city_distance(solution)
        sum_path = sum(routines_dis)
        max_path = max(routines_dis)
        min_path = min(routines_dis)
        balance = (max_path - min_path) / max_path
        obj = self.distance_weight * sum_path + \
              self.balance_weight * balance

        return obj

    def swap_solution(self, solution):
        """
        交换产生新解，与交换变异类似
        Args:
            solution: 解

        Returns:
            new_solution：新解
        """
        # 如果index1和index2相等，则交换变异相当于没有执行
        index1 = random.randint(0, self.solution_len - 1)
        index2 = random.randint(0, self.solution_len - 1)
        new_solution = solution[:]
        new_solution[index1], new_solution[index2] = new_solution[index2], new_solution[index1]
        return new_solution

    def shift_solution(self, solution):
        """
        移位产生新解：随机选取三个点，将前两个点之间的点移位到第三个点的后方
        solution     = [5, 8, 6, 1, 12, 4, 11, 15, 2]
        new_solution = [5, 4, 11, 8, 6, 1, 12, 15, 2]
        Args:
            solution:解

        Returns:
            new_solution:新解
        """
        tmp = sorted(random.sample(range(self.solution_len), 3))  # 随机选取3个不同的数
        index1, index2, index3 = tmp[0], tmp[1], tmp[2]
        tmp = solution[index1:index2]
        new_solution = []
        for i in range(self.solution_len):
            if index1 <= i < index2:
                continue
            if (i < index1 or i >= index2) and i < index3:
                new_solution.append(solution[i])
            elif i == index3:
                new_solution.append(solution[i])
                new_solution.extend(tmp)
            else:
                new_solution.append(solution[i])
        return new_solution

    def reverse_solution(self, solution):
        """
        逆转：随机选择两点(可能为同一点)，逆转其中所有的元素
        solution     = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        new_solution = [1, 2, 6, 5, 4, 3, 7, 8, 9]
        Args:
            solution:父代

        Returns:
            new_solution：逆转变异后的子代
        """
        index1, index2 = random.randint(0, self.solution_len - 1), random.randint(0, self.solution_len - 1)
        if index1 > index2:
            index1, index2 = index2, index1
        new_solution = solution[:]
        tmp = new_solution[index1: index2]
        tmp.reverse()
        new_solution[index1: index2] = tmp
        return new_solution

    def generate_new_solu(self, solution):
        """
        产生新解的过程类似变异过程
        Args:
            solution: 解

        Returns:
            new_solution：新解
        """
        """
        prob_sum表示一种累加和的列表，比如：
        四种变异可能性[0.2, 0.3, 0.4, 0.1]
        prob_sum = [0.2, 0.5, 0.9, 1]
        变异只有三种变异，这里采用了硬编码
        """
        prob_sum = []
        prob_sum.extend([self.swap_solu_prob, self.swap_solu_prob + self.reverse_solu_prob, 1])
        p = random.random()
        if p <= prob_sum[0]:
            # 交换产生新解
            new_solution = self.swap_solution(solution)
        elif p <= prob_sum[1]:
            # 逆转产生新解
            new_solution = self.reverse_solution(solution)
        else:
            # 移位产生新解
            new_solution = self.shift_solution(solution)
        return new_solution

    def sa_process_iterator(self, solution, get_distance_func):
        """
        SA算法的迭代流程
        Args:
            solution:解
            get_distance_func:计算距离函数

        Returns:

        """
        while self.T > self.T_end:
            # 每个温度下最优解都要赋值
            self.per_iter_solution = solution
            # 在每个温度下迭代
            for _ in range(self.Lk):
                # 当前解的目标函数值
                current_solu_obj = self.object_func(solution)
                # 产生新解
                new_solu = self.generate_new_solu(solution)
                # 新解目标函数值
                new_solu_obj = self.object_func(new_solu)
                # 新解更优，接受新解
                if new_solu_obj < current_solu_obj:
                    solution = new_solu
                # Metropolis准则
                else:
                    prob_accept = math.exp(-(new_solu_obj - current_solu_obj) / self.T)
                    p = random.random()
                    if p < prob_accept:
                        solution = new_solu
            # 该温度下迭代完成
            solu_obj = self.object_func(solution)
            # 解和该温度下最优解比较
            if solu_obj < self.object_func(self.per_iter_solution):
                self.per_iter_solution = solution
            # 解和全局最优解比较
            if solu_obj < self.object_func(self.best_solution):
                self.best_solution = solution
            # 记录每个温度下最优解和全局最优解
            self.all_per_iter_solution.append(self.per_iter_solution)
            self.all_best_solution.append(self.best_solution)

            per_iter_solu_path, per_iter_solu_dis = get_distance_func(self.per_iter_solution)
            best_solu_path, best_solu_dis = get_distance_func(self.best_solution)
            # ********************参数打印********************************
            print("在T = {} 温度下:".format(self.T))
            print("该温度下，最优解为{}".format(self.per_iter_solution))
            print("该温度下，最优解路线为{}".format(per_iter_solu_path))
            print("该温度下，最优解路线长度为{}".format(sum(per_iter_solu_dis)))
            print("该温度下，最优解路线长度列表为{}".format(per_iter_solu_dis))
            print("---------------------------------------------------------")
            print("全局最优解为{}".format(self.best_solution))
            print("全局最优解路线为{}".format(best_solu_path))
            print("全局最优解路线长度为{}".format(sum(best_solu_dis)))
            print("全局最优解路线长度列表为{}".format(best_solu_dis))
            print("**************************************************************************")

            # *******************有关参数更新****************************
            self.T_list.append(self.T)
            self.T = self.T * self.alpha

    def check_vertex_sa_process(self):
        """
        针对实验 1：检查点路线
        SA流程
        Returns:

        """
        self.check_vertex_init()
        self.sa_process_iterator(self.per_iter_solution, self.get_check_vertex_distance)

    def china_city_sa_process(self):
        """
        针对实验 2：运输机航线
        SA流程
        Returns:

        """
        self.china_city_init()
        self.sa_process_iterator(self.per_iter_solution, self.get_china_city_distance)

    def plot_dis_sum_diff(self, get_distance_func):
        """
        画出每个温度下最优解的总路程和全局最优解总路程变化情况
        Args:
            get_distance_func: 计算距离的函数

        Returns:

        """
        plt.figure()
        ax = plt.gca()
        ax.invert_xaxis()  # x轴反向
        all_per_iter_solu_dis = []
        for i in self.all_per_iter_solution:
            _, per_iter_solu_dis = get_distance_func(i)
            all_per_iter_solu_dis.append(sum(per_iter_solu_dis))

        # 每个温度下最优解的总路程变化情况
        plt.plot(self.T_list, all_per_iter_solu_dis, color='r')

        all_best_solu_dis = []
        for i in self.all_best_solution:
            _, best_solu_dis = get_distance_func(i)
            all_best_solu_dis.append(sum(best_solu_dis))

        # 全局最优解总路程变化情况
        plt.plot(self.T_list, all_best_solu_dis, color='b')
        plt.legend(['每个温度下最优解', '全局最优解'])
        plt.xlabel("温度", fontsize=14)
        plt.ylabel("最优解总路程", fontsize=14)
        plt.show()

    def print_check_best_solu_routine(self):
        """
        打印最优检查点路线
        Returns:

        """
        best_path, best_dist_list = self.get_check_vertex_distance(self.best_solution)
        print("所有路线长度为：{}".format(sum(best_dist_list)))
        # 打印全局最优个体的所有路线（包括起点和终点）
        for i in range(len(best_path)):
            print("第{}个巡检员路线长度为：{}".format(i + 1, best_dist_list[i]))
            print("第{}个巡检员路线为：".format(i + 1), end="")
            if len(best_path[i]) == 0:
                print("该巡检员不出门")  # 这种情况可以通过设置目标函数避免
            else:
                for j in range(len(best_path[i])):
                    if j == 0:
                        print(self.start_index, end="")
                        self.graph_obj.print_path(self.start_index, best_path[i][j])
                    if j + 1 < len(best_path[i]):
                        self.graph_obj.print_path(best_path[i][j], best_path[i][j + 1])
                    elif j == len(best_path[i]) - 1:
                        self.graph_obj.print_path(best_path[i][j], self.start_index)
                print()

    def print_china_city_best_routine(self):
        """
        打印运输机最优航线
        Returns:

        """
        best_path, best_dist_list = self.get_china_city_distance(self.best_solution)
        print("运输机所有路线长度为：{}".format(sum(best_dist_list)))
        # 打印全局最优个体的所有路线城市（包括起点和终点）
        for i in range(len(best_path)):
            print("第{}架运输机路线长度为：{}".format(i + 1, best_dist_list[i]))
            print("第{}架运输机路线为：".format(i + 1), end="")
            if len(best_path[i]) == 0:
                print("该运输机不出发")  # 这种情况可以通过设置目标函数避免
            else:
                for j in range(len(best_path[i])):
                    if j == 0:
                        print("{} ——> {} ".format(self.china_cities[self.start_index][0],
                                                  self.china_cities[best_path[i][j]][0]), end="")
                    if j + 1 < len(best_path[i]):
                        print("——> {} ".format(self.china_cities[best_path[i][j + 1]][0]), end="")
                    elif j == len(best_path[i]) - 1:
                        print("——> {}".format(self.china_cities[self.start_index][0]))


if __name__ == "__main__":
    # 测试使用
    # 实验1：检查点实验
    sa_obj = SA(salesman_num=3,
                city_num=26,
                start_index=22,
                filename='./data/routine_nodes.txt',
                distance_weight=1,
                balance_weight=100)
    sa_obj.check_vertex_sa_process()
    sa_obj.plot_dis_sum_diff(sa_obj.get_check_vertex_distance)
    sa_obj.print_check_best_solu_routine()

    # 实验2：运输机航线
    # sa_obj = SA(salesman_num=3,
    #             city_num=34,
    #             start_index=22,
    #             filename='./data/provincial_capital.txt',
    #             distance_weight=1,
    #             balance_weight=2000)
    # sa_obj.china_city_sa_process()
    # sa_obj.print_china_city_best_routine()

