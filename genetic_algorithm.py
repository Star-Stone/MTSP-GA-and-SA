import random
import utils
from utils import Graph
import time
import matplotlib.pyplot as plt
import math

# INF无穷大
INF = 999999


class GA:
    def __init__(self, salesman_num, city_num, start_index,
                 filename, max_gen, distance_weight, balance_weight):
        self.population_size = 30  # 种群规模大小
        self.population = []  # 种群
        # self.all_pop_fitness = []  # 记录每次迭代过程种群适应度之和变化情况，看情况可以添加
        self.salesman_num = salesman_num  # 旅行商数量
        self.city_num = city_num  # 城市的数量（也是检查点数量）
        self.chrom_len = salesman_num + city_num - 2  # 染色体长度=(城市的数量-1) + (旅行商数量-1),染色体长度由编码方式决定
        self.start_index = start_index  # 旅行商起始城市的序号，序号取值[1,总的城市数]
        self.MAX_GEN = max_gen  # 迭代最大次数
        self.gen_count = 1  # 迭代计数器
        self.dummy_points = [x for x in range(self.city_num + 1, self.city_num + self.salesman_num)]  # 虚点。虚点详细见下初始化函数check_vertex_init_population
        self.fitness_func = None  # 使用到的适应度函数

        self.filename = filename  # 读取文件名
        self.graph_obj = Graph(filename)  # 针对实验1 检查点路线：创建utils里面Graph对象
        self.china_cities = []  # 针对实验2 运输机航线

        self.per_pop_best_chrom = []  # 每代的最优个体
        self.per_pop_best_chrom_fit = 0  # 每代的最优个体的适应度
        self.per_pop_best_path = []  # 每代最优个体的路线
        self.per_pop_best_dis_sum = INF  # 每代最优个体的总距离
        self.all_per_pop_best_chrom = []  # 记录每次迭代过程中每代最优个体变化情况
        self.all_per_pop_best_chrom_fit = []  # 记录每次迭代过程中每代最优个体的适应度变化情况（可以不必设置，因为check_vertex_fitness_func可以计算单体适应度）
        self.all_per_pop_best_dist_sum = []  # 记录每次迭代过程中每代最优个体的总距离变化情况

        self.best_chrom = []  # 全局最优个体，不一定是每代的最优个体，每代的最优个体可能比以往的最优个体差
        self.best_chrom_fit = 0  # 全局最优个体的适应度
        self.best_path = []  # 全局最优个体的路径
        self.best_dis_sum = INF  # 全局最优个体的路径之和
        self.all_best_chrom = []  # 记录每次迭代过程中全局最优个体的变化情况
        self.all_best_chrom_fit = []  # 记录每次迭代过程中全局最优个体的适应度变化情况
        self.all_best_dist_sum = []  # 记录每次迭代过程中全局最优个体的总距离

        self.cross_prob = 0.8  # 交叉概率
        self.mutation_prob = 0.15  # 变异概率
        self.cross_pmx_prob = 0.5  # 交叉选择部分匹配交叉PMX的概率，这部分没用到，只用到cross_ox_prob
        self.cross_ox_prob = 0.5  # 交叉选择顺序匹配交叉OX的概率

        self.mutation_swap_prob = 0.3  # 变异选择"交换两个元素"的概率
        self.mutation_reverse_prob = 0.4  # 变异选择"反转两个元素之间所有元素"的概率
        self.mutation_insert_prob = 1 - self.mutation_swap_prob - self.mutation_reverse_prob  # 变异选择"一个元素插入到另一个元素后面"的概率

        self.distance_weight = distance_weight  # 总路程权重
        self.balance_weight = balance_weight  # 均衡度数权重

    def check_vertex_init_population(self):
        """
        针对实验1：检查点路线
        调用floyd算法得到任意两点的最短路径，初始化种群以及全局最优解：初始化self.population_num个染色体，染色体长度为self.chrom_len
        Returns:
        """
        """
        chrom组成
        如：总共8个城市，3号城市为起始点.下列染色体组成我们把起点和终点省略，看情况是否增加虚点
        1个旅行商：一个chrom = [把3剔除，其余数字由1到8组成]
            如[1,5,4,2,6,8,7]表示旅行商路线为3->1->5->4->2->6->8->7->3
        2个旅行商：一个chrom = [1个9(9代表虚点，其实也是起点3)，其余数字由1到8组成]。以此类推到多个旅行商的情况。
            如[1,5,4,9,2,6,8,7]表示：
                旅行商1路线为3->1->5->4->3(9)
                旅行商2路线为3(9)->2->6->8->7->3
        3个旅行商：一个chrom = [9,10，其余数字由1到8组成]
            如[1,5,4,9,2,6,10,8,7]
                旅行商1路线为3->1->5->4->3(9)
                旅行商2路线为3->2->6->3(10)
                旅行商3路线为3->8->7->3
        """
        # 调用floyd算法得到任意两点的最短路径
        self.graph_obj.short_path_floyd()
        # 染色体的适应度函数是实验一检查点路线对应的check_vertex_fitness_func
        self.fitness_func = self.check_vertex_fitness_func

        for i in range(self.population_size):
            # 注意：城市起点从1开始，而不是从0
            chrom = [x for x in range(1, self.city_num + 1)]
            # 把起始点剔除
            chrom.remove(self.start_index)
            # 多个旅行商，增加salesman_num-1个虚点
            chrom.extend(self.dummy_points)
            random.shuffle(chrom)
            self.population.append(chrom)

        # 初始化全局最优个体和它的适应度
        self.best_chrom = self.population[0]
        self.best_chrom_fit = self.fitness_func(self.best_chrom)

    def china_city_init_pop(self):
        """
        针对实验2：运输机路线
        chrom组成
        如：总共8个城市，3号城市为起始点.下列染色体组成我们把起点和终点省略，看情况是否增加虚点
        1个旅行商：一个chrom = [把3剔除，其余数字由1到8组成]
            如[1,5,4,2,6,8,7]表示旅行商路线为3->1->5->4->2->6->8->7->3
        2个旅行商：一个chrom = [1个9(9代表虚点，其实也是起点3)，其余数字由1到8组成]。以此类推到多个旅行商的情况。
            如[1,5,4,9,2,6,8,7]表示：
                旅行商1路线为3->1->5->4->3(9)
                旅行商2路线为3(9)->2->6->8->7->3
        3个旅行商：一个chrom = [9,10，其余数字由1到8组成]
            如[1,5,4,9,2,6,10,8,7]
                旅行商1路线为3->1->5->4->3(9)
                旅行商2路线为3->2->6->3(10)
                旅行商3路线为3->8->7->3
        """
        # 读取中国城市文件，并初始化china_cities
        self.china_cities = utils.read_china_cities_coord(self.filename)
        # 染色体的适应度函数是实验二中国城市对应的china_city_fitness_func
        self.fitness_func = self.china_city_fitness_func

        for i in range(self.population_size):
            # 注意：城市起点从1开始，而不是从0
            chrom = [x for x in range(1, self.city_num + 1)]
            # 把起始点剔除
            chrom.remove(self.start_index)
            # 多个旅行商，增加salesman_num-1个虚点
            chrom.extend(self.dummy_points)
            random.shuffle(chrom)
            self.population.append(chrom)
        # 初始化全局最优个体和它的适应度
        self.best_chrom = self.population[0]
        self.best_chrom_fit = self.fitness_func(self.best_chrom)

    def binary_tourment_select(self, population):
        """
        二元锦标赛：从种群中抽取2个个体参与竞争，获胜者个体进入到下一代种群
        Args:
            population: 目前种群

        Returns:
            new_population:下一代种群
        """
        new_population = []  # 下一代种群
        for i in range(self.population_size):
            # 随机选择2个个体
            competitors = random.choices(population, k=2)
            # 选择适应度大的个体
            winner = max(competitors, key=lambda x: self.fitness_func(x))
            new_population.append(winner)
        return new_population

    def cross_ox(self, parent_chrom1, parent_chrom2):
        """
        对两个父代染色体进行OX交叉，得到两个子代染色体
        Args:
            parent_chrom1: 父代染色体1
            parent_chrom2: 父代染色体2

        Returns:
            child_chrom1：子代染色体1
            child_chrom2：子代染色体2
        """
        # random.randint(a,b)返回值域[a,b]
        index1, index2 = random.randint(0, self.chrom_len - 1), random.randint(0, self.chrom_len - 1)
        if index1 > index2:
            index1, index2 = index2, index1
        # temp_gene1为parent_chrom1被选中的染色体片段[index1:index2)
        temp_gene1 = parent_chrom1[index1:index2]
        # temp_gene2为parent_chrom2被选中的染色体片段[index1:index2)
        temp_gene2 = parent_chrom2[index1:index2]
        """
        将parent_chrom1被选中的基因片段复制给child_chrom1，
        这里复制是指child_chrom1[index1:index2] = parent_chrom1[index1:index2]
        然后parent_chrom2除了temp_gene1包含的基因，parent_chrom2剩下基因按照顺序放到child_chrom1中
        即(|parent_chrom2|-|temp_gene1|)基因按顺序放到child_chrom1。
        同理，child_chrom2交换一下parent_chrom1和parent_chrom2，也可以得到
        如：
        parent_chrom1 = [1, 2, 3, 4, 5, 6, 7, 8, 9], sel1选中部分[3, 4, 5, 6]
        parent_chrom2 = [5, 7, 4, 9, 1, 3, 6, 2, 8], sel2选中部分[6, 9, 2, 1]
        child_chrom1  = [7, 9, 3, 4, 5, 6, 1, 2, 8]
            1、child_chrom1对应部分放入sel1
            2、遍历parent_chrom2，parent_chrom2不属于sel1部分基因，按照顺序放入
        """
        child_chrom1, child_chrom2 = [], []
        child_p1, child_p2 = 0, 0
        # 得到child_chrom1
        for i in parent_chrom2:
            if child_p1 == index1:
                child_chrom1.extend(temp_gene1)
                child_p1 += 1
            if i not in temp_gene1:
                child_chrom1.append(i)
                child_p1 += 1

        # 得到child_chrom2
        for i in parent_chrom1:
            if child_p2 == index1:
                child_chrom2.extend(temp_gene2)
                child_p2 += 1
            if i not in temp_gene2:
                child_chrom2.append(i)
                child_p2 += 1

        return child_chrom1, child_chrom2

    def cross_pmx(self, parent_chrom1, parent_chrom2):
        """
        pmx部分匹配，里面需要冲突检测
        Args:
            parent_chrom1: 父代1
            parent_chrom2: 父代2

        Returns:
            chrom1, chrom2:子代1，子代2
        """
        index1, index2 = random.randint(0, self.chrom_len - 1), random.randint(0, self.chrom_len - 1)
        if index1 > index2:
            index1, index2 = index2, index1
        """
        如：
        index1 = 2, index2 = 6
        parent_chrom1 = [1, 2, 3, 4, 5, 6, 7, 8, 9], 选中部分[3, 4, 5, 6]
        parent_chrom2 = [5, 4, 6, 9, 2, 1, 7, 8, 3], 选中部分[6, 9, 2, 1]
        选中部分的映射关系即1<->6<->3 ; 2<->5 ; 9<->4
        可以看出存在1<->6<->3，说明6在父代1和2选中部分，6后续不需要冲突检测，所以应该1<->3
        """
        parent_part1, parent_part2 = parent_chrom1[index1:index2], parent_chrom2[index1:index2]

        child_chrom1, child_chrom2 = [], []
        child_p1, child_p2 = 0, 0  # 指针用来解决复制到指定位置问题
        # 子代1
        for i in parent_chrom1:
            # 指针到达父代的选中部分
            if index1 <= child_p1 < index2:
                # 将父代2选中基因片段复制到子代1指定位置上
                child_chrom1.append(parent_part2[child_p1 - index1])
                child_p1 += 1
                continue
            # 指针未到达父代的选中部分
            if child_p1 < index1 or child_p1 >= index2:
                # 父代1未选中部分含有父代2选中部分基因
                if i in parent_part2:
                    tmp = parent_part1[parent_part2.index(i)]
                    """
                    这里可能出现很长的映射链,如：
                    parent_part1 = [2, 3, 7, 5, 6, 14, 10, 11, 13]
                    parent_part2 = [4, 2, 1, 3, 5, 7,  14, 6,  10]
                    映射链：1 <-> 7 <-> 14 <-> 10 <-> 13 
                    所以采用循环的形式
                    """
                    while tmp in parent_part2:
                        tmp = parent_part1[parent_part2.index(tmp)]
                    child_chrom1.append(tmp)
                elif i not in parent_part2:
                    child_chrom1.append(i)
                child_p1 += 1
        # 子代2
        for i in parent_chrom2:
            # 指针到达父代的选中部分
            if index1 <= child_p2 < index2:
                # 将父代1选中基因片段复制到子代2指定位置上
                child_chrom2.append(parent_part1[child_p2 - index1])
                child_p2 += 1
                continue
            # 指针未到达父代的选中部分
            if child_p2 < index1 or child_p2 >= index2:
                # 父代2未选中部分含有父代1选中部分基因
                if i in parent_part1:
                    tmp = parent_part2[parent_part1.index(i)]
                    # 解决1<->6<->3
                    while tmp in parent_part1:
                        tmp = parent_part2[parent_part1.index(tmp)]
                    child_chrom2.append(tmp)
                elif i not in parent_part1:
                    child_chrom2.append(i)
                child_p2 += 1

        return child_chrom1, child_chrom2

    def crossover(self, population):
        """
        种群按概率执行交叉操作
        Args:
            population: 种群

        Returns:
            new_population：新一代种群
        """
        # 交叉:比较特殊，只有PMX和OX
        new_population = []
        # 二元锦标赛选择出新的一代
        selected_pop = self.binary_tourment_select(population)
        for i in range(self.population_size):
            prob = random.random()  # 随机数，决定是PMX还是OX
            two_chrom = random.choices(selected_pop, k=2)
            if prob <= self.cross_ox_prob:
                # 执行OX
                child_chrom1, child_chrom2 = self.cross_ox(two_chrom[0], two_chrom[1])
                new_population.append(child_chrom1)
                new_population.append(child_chrom2)
            else:
                # 执行PMX
                child_chrom1, child_chrom2 = self.cross_pmx(two_chrom[0], two_chrom[1])
                new_population.append(child_chrom1)
                new_population.append(child_chrom2)
        return new_population

    def mutate_swap(self, parent_chrom):
        """
        交换变异：当前染色体 [1,5,4,2,6,8,7]，交换1和5位置上元素变成了[1,8,4,2,6,5,7]
        Args:
            parent_chrom: 父代染色体

        Returns:
            child_chrom：交换变异产生的子代染色体
        """
        # 如果index1和index2相等，则交换变异相当于没有执行
        index1 = random.randint(0, self.chrom_len - 1)
        index2 = random.randint(0, self.chrom_len - 1)
        child_chrom = parent_chrom[:]
        child_chrom[index1], child_chrom[index2] = child_chrom[index2], child_chrom[index1]
        return child_chrom

    def mutate_reverse(self, parent_chrom):
        """
        逆转变异：随机选择两点(可能为同一点)，逆转其中所有的元素
        parent_chrom = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        child_chrom  = [1, 2, 6, 5, 4, 3, 7, 8, 9]
        Args:
            parent_chrom:父代

        Returns:
            child_chrom：逆转变异后的子代
        """
        index1, index2 = random.randint(0, self.chrom_len - 1), random.randint(0, self.chrom_len - 1)
        if index1 > index2:
            index1, index2 = index2, index1
        child_chrom = parent_chrom[:]
        tmp = child_chrom[index1: index2]
        tmp.reverse()
        child_chrom[index1: index2] = tmp
        return child_chrom

    def mutate_insert(self, parent_chrom):
        """
        插入变异：随机选择两个位置，然后将这第二个位置上的元素插入到第一个元素后面。
        parent_chrom = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        child_chrom  = [1, 2, 4, 5, 3, 6, 7, 8, 9]
        Args:
            parent_chrom:父代

        Returns:
            child_chrom：子代
        """
        index1, index2 = random.randint(0, self.chrom_len - 1), random.randint(0, self.chrom_len - 1)
        child_chrom = parent_chrom[:]
        child_chrom.pop(index2)
        child_chrom.insert(index1 + 1, parent_chrom[index2])
        return child_chrom

    def mutation(self, population):
        """
        种群按概率执行变异操作
        Args:
            population: 种群

        Returns:
            new_population：新一代种群
        """
        """
        prob_sum表示一种累加和的列表，比如：
        四种变异可能性[0.2, 0.3, 0.4, 0.1]
        prob_sum = [0.2, 0.5, 0.9, 1]
        变异只有三种变异，这里采用了硬编码
        """
        prob_sum = []
        prob_sum.extend([self.mutation_swap_prob, self.mutation_swap_prob + self.mutation_reverse_prob, 1])
        new_population = []
        for i in range(self.population_size):
            p = random.random()
            if p <= prob_sum[0]:
                # 交换变异
                child_chrom = self.mutate_swap(population[i])
                new_population.append(child_chrom)
            elif p <= prob_sum[1]:
                # 逆序变异
                child_chrom = self.mutate_reverse(population[i])
                new_population.append(child_chrom)
            else:
                # 插入变异
                child_chrom = self.mutate_insert(population[i])
                new_population.append(child_chrom)
        return new_population

    def compute_pop_fitness(self, population):
        """
        计算当前种群所有个体的的适应度
        Args:
            population: 种群

        Returns:
            种群所有个体的的适应度
        """
        return [self.fitness_func(chrom) for chrom in population]

    def get_best_chrom(self, population):
        """
        找到种群中最优个体
        Args:
            population: 种群

        Returns:
            population[index]：最优个体
            index:最优个体在种群中下标
        """
        tmp = self.compute_pop_fitness(population)
        index = tmp.index(max(tmp))
        return population[index], index

    def get_check_vertex_distance(self, chrom):
        """
        针对实验1：检查点路线
        根据染色体解码，得到所有旅行商走的路线以及每条路线总路程
        Args:
            chrom: 染色体

        Returns:
            all_routines：所有旅行商走的路线
            routines_dis：每条路线总路程组成列表
        """
        # 起始点5，城市9个，旅行商3，虚点10,11
        # [4, 6, 11, 9, 2, 1, 10, 7, 8, 3]
        tmp_chrom = chrom[:]
        # 将增加的虚点还原成起始点
        for i in range(len(chrom)):
            if chrom[i] in self.dummy_points:
                tmp_chrom[i] = self.start_index

        # 根据起始点把chrom分成多段
        one_routine = []  # 一个旅行商路线，可以为空
        all_routines = []  # 所有旅行商路线
        for v in tmp_chrom:
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

    def check_vertex_obj_func(self, chrom):
        """
        针对实验1：检查点路线
        计算个体的目标函数值
        目标函数 Z = distance_weight*总路程 + balance_weight*均衡度
        均衡度 = (max(l)-min(l))/ max(l)
        Args:
            chrom: 染色体(个体)

        Returns:
            obj：个体的目标函数值
        """
        all_routines, routines_dis = self.get_check_vertex_distance(chrom)
        sum_path = sum(routines_dis)
        max_path = max(routines_dis)
        min_path = min(routines_dis)
        balance = (max_path - min_path) / max_path
        obj = self.distance_weight * sum_path + \
              self.balance_weight * balance

        return obj

    def check_vertex_fitness_func(self, chrom):
        """
        针对实验 1：检查点路线
        计算个体的适应度值，即个体目标函数值的倒数
        Args:
            chrom:染色体

        Returns:
            个体的适应度
        """
        return math.exp(1.0 / self.check_vertex_obj_func(chrom))

    def check_vertex_ga_process(self):
        """
        针对实验 1：检查点路线
        GA的流程
        Returns:

        """
        self.check_vertex_init_population()
        best_dist_list = []  # 全局最优解每一条路径长度
        self.ga_process_iterator(best_dist_list, self.get_check_vertex_distance)

    def get_china_city_distance(self, chrom):
        """
        针对实验 2：运输机航线
        Args:
            chrom: 染色体

        Returns:
            all_routines：所有运输机路线组成列表
            routines_dis：每条路线总路程组成列表
        """
        # 起始点5，城市9个，旅行商3，虚点10,11
        # [4, 6, 11, 9, 2, 1, 10, 7, 8, 3]
        tmp_chrom = chrom[:]
        # 将增加的虚点还原成起始点
        for i in range(len(chrom)):
            if chrom[i] in self.dummy_points:
                tmp_chrom[i] = self.start_index
        # 根据起始点把chrom分成多段
        one_routine = []  # 一个旅行商路线，可以为空
        all_routines = []  # 所有旅行商路线
        for v in tmp_chrom:
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

    def china_city_obj_func(self, chrom):
        """
        针对实验 2：运输机航线
        计算个体的目标函数值
        目标函数 Z = distance_weight*总路程 + balance_weight*均衡度
        均衡度 = (max(l)-min(l))/ max(l)
        Args:
            chrom: 染色体(个体)

        Returns:
            obj：个体的目标函数值
        """
        all_routines, routines_dis = self.get_china_city_distance(chrom)
        sum_path = sum(routines_dis)
        max_path = max(routines_dis)
        min_path = min(routines_dis)
        balance = (max_path - min_path) / max_path
        obj = self.distance_weight * sum_path + \
              self.balance_weight * balance

        return obj

    def china_city_fitness_func(self, chrom):
        """
        针对实验 2：运输机航线
        计算个体的适应度值，即个体目标函数值的倒数
        Args:
            chrom:染色体

        Returns:
            个体的适应度
        """
        return math.exp(1.0 / self.china_city_obj_func(chrom))

    def china_city_ga_process(self):
        """
        针对实验 2：运输机航线
        GA流程
        Returns:

        """
        self.china_city_init_pop()
        best_dist_list = []  # 全局最优解每一条路径长度
        self.ga_process_iterator(best_dist_list, self.get_china_city_distance)

    def ga_process_iterator(self, best_dist_list, get_distance_func):
        """
        GA算法的迭代过程
        Args:
            best_dist_list: 全局最优解每一条路径长度
            get_distance_func: 计算距离使用的函数
                * get_check_vertex_distance：实验1
                * get_china_city_distance：实验2
        Returns:

        """
        # 遗传算法的迭代过程
        while self.gen_count <= self.MAX_GEN:
            # 每次迭代记录种群适应度之和，也可以不用记录
            # self.all_pop_fitness.append(sum(self.compute_pop_fitness(self.population)))

            # 锦标赛选择
            pop_new = self.binary_tourment_select(self.population)
            # -------------------交叉------------------------------------------
            # 随机数决定是否交叉
            p_cross = random.random()
            if p_cross <= self.cross_prob:
                pop_new = self.crossover(pop_new)
            # -------------------变异------------------------------------------
            # 随机数决定是否变异
            p_mutate = random.random()
            if p_mutate <= self.mutation_prob:
                pop_new = self.mutation(pop_new)
            # -------------------新的一代有关参数更新-------------------------------
            # *******************新的一代的最优个体有关参数更新**********************
            # 计算种群所有个体的适应度
            pop_fitness_list = self.compute_pop_fitness(pop_new)
            # 每代最优个体per_pop_best_chrom及其在种群中的下标best_index
            self.per_pop_best_chrom, best_index = self.get_best_chrom(pop_new)
            # 每代最优个体的适应度
            self.per_pop_best_chrom_fit = pop_fitness_list[best_index]
            # 每代最优个体最好的路径组成和每条路路径长度per_pop_best_dist_list
            self.per_pop_best_path, per_pop_best_dist_list = get_distance_func(self.per_pop_best_chrom)
            # 每代最优个体所有旅行商路线之和
            self.per_pop_best_dis_sum = sum(per_pop_best_dist_list)

            # 记录下每代最优个体
            self.all_per_pop_best_chrom.append(self.per_pop_best_chrom)
            # 记录下每代最优个体的适应度
            self.all_per_pop_best_chrom_fit.append(self.per_pop_best_chrom_fit)
            # 记录每次迭代过程中每代最优个体的总距离变化情况
            self.all_per_pop_best_dist_sum.append(self.per_pop_best_dis_sum)

            # *******************全局最优个体有关参数更新****************************
            # 每代最优个体与全局最优个体根据适应度比较，如果每代最优个体适应度更小，则更新全局最优个体
            if self.per_pop_best_chrom_fit > self.best_chrom_fit:
                self.best_chrom = self.per_pop_best_chrom
                self.best_chrom_fit = self.per_pop_best_chrom_fit
                # 全局最优个体最好的路径组成和每条路路径长度
                self.best_path, best_dist_list = get_distance_func(self.best_chrom)
                # self.best_path = self.per_pop_best_path
                # 全局最优个体的路径之和
                self.best_dis_sum = self.per_pop_best_dis_sum

                # 记录下每次迭代过程中全局最优个体
                self.all_best_chrom.append(self.best_chrom)

            # 记录每次迭代过程中全局最优个体的适应度变化情况
            self.all_best_chrom_fit.append(self.best_chrom_fit)
            # 记录每次迭代过程中全局最优个体的总距离
            self.all_best_dist_sum.append(self.best_dis_sum)

            # 输出
            if self.gen_count % 50 == 0:
                print("经过%d次迭代" % self.gen_count)
                print("全局最优解距离为：%f，全局最优解长度为%d" % (self.best_dis_sum, len(self.best_chrom)))
                print("全局最优解为{}".format(self.best_chrom))
                print("全局最优解路线为{}".format(self.best_path))
                print("全局最优解路线长度列表为{}".format(best_dist_list))
                print("---------------------------------------------------------")
                print("每代最优解距离为：%f，每代最优解长度为%d" % (self.per_pop_best_dis_sum, len(self.per_pop_best_chrom)))
                print("每代最优解为{}".format(self.per_pop_best_chrom))
                print("每代最优解路线为{}".format(self.per_pop_best_path))
                print("每代最优解路线长度列表为{}".format(per_pop_best_dist_list))
                print("**************************************************************************")

            # *******************种群有关参数更新****************************
            # 更新种群
            self.population = pop_new
            # 计数器加1
            self.gen_count += 1
            # -------------------新的一代有关参数更新结束------------------------------------------------

    def plot_dis_sum_diff(self):
        """
        画出每代最优个体的总路程和全局最优个体总路程变化情况
        Returns:

        """
        plt.figure()
        x = [i for i in range(1, self.MAX_GEN + 1)]
        # 每代最优个体的总路程变化情况
        plt.plot(x, self.all_per_pop_best_dist_sum, color='r')
        # 全局最优个体总路程变化情况
        plt.plot(x, self.all_best_dist_sum, color='b')
        plt.legend(['每代最优个体', '全局最优个体'])
        plt.xlabel("迭代次数", fontsize=14)
        plt.ylabel("最优个体总路程", fontsize=14)

        plt.show()

    def plot_chrom_fit_diff(self):
        """
        画出每代最优个体的适应度和全局最优个体适应度变化情况
        Returns:

        """
        plt.figure()
        x = [i for i in range(1, self.MAX_GEN + 1)]
        per_pop_best_fit = [i for i in self.all_per_pop_best_chrom_fit]
        best_fit = [i for i in self.all_best_chrom_fit]
        # 每代最优个体的适应度变化情况
        plt.plot(x, per_pop_best_fit, color='r')
        # 全局最优个体适应度变化情况
        plt.plot(x, best_fit, color='b')
        plt.legend(['每代最优个体', '全局最优个体'])
        plt.xlabel("迭代次数", fontsize=14)
        plt.ylabel("最优个体适应度", fontsize=14)
        plt.show()

    def print_check_best_chrom_routine(self):
        """
        打印最优检查点路线
        Returns:

        """
        print("所有路线长度为：{}".format(self.best_dis_sum))
        best_path, best_dist_list = self.get_check_vertex_distance(self.best_chrom)
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
        print("运输机所有路线长度为：{}".format(self.best_dis_sum))
        best_path, best_dist_list = self.get_china_city_distance(self.best_chrom)
        # 打印全局最优个体的所有路线城市（包括起点和终点）
        for i in range(len(best_path)):
            print("第{}架运输机路线长度为：{}".format(i+1, best_dist_list[i]))
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
    # 主要用来测试
    # 实验1：检查点路线
    ga_obj = GA(salesman_num=3,
                city_num=26,
                start_index=22,
                filename="./data/routine_nodes.txt",
                max_gen=1000,
                distance_weight=1,
                balance_weight=50)

    time_start = time.time()
    ga_obj.check_vertex_ga_process()
    time_end = time.time()
    print('time cost', time_end - time_start, 's')

    # 实验2：运输机航线
    # ga_obj = GA(salesman_num=3,
    #             city_num=34,
    #             start_index=22,
    #             filename="./data/provincial_capital.txt",
    #             max_gen=1000,
    #             distance_weight=1,
    #             balance_weight=3600)  # 设为4000，每条路线路程比较平均
    # ga_obj.china_city_ga_process()



