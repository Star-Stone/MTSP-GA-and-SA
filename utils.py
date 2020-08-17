from math import radians, cos, sin, asin, sqrt
# INF无穷大
INF = 100


def read_nodes_file(filename):
    """
    针对实验 1：检查点路线
    读取顶点文件
    Args:
        filename: 顶点文件名

    Returns:
        vertex_num:顶点数
        edges：顶点文件对应的二维列表形式

    """
    with open(filename, 'r', encoding='utf-8') as file:
        edges = []
        vertex_num = 0
        for line in file:
            # 每行字符串去掉换行符，再以制表符分割，得到['0001', '35', '3']
            edge = line.rstrip('\n').split('\t')
            # 字符串转数字
            edge = [int(x) for x in edge]
            # 节点数为第一、二列中最大值，借助了tmp实现
            tmp = edge[0] if edge[0] > edge[1] else edge[1]
            vertex_num = tmp if vertex_num < tmp else vertex_num
            edges.append(edge)
        return vertex_num, edges


def read_china_cities_coord(filename):
    """
    针对实验 2：运输机航线
    读取中国城市文件
    Args:
        filename: 中国城市文件

    Returns:
        cities_coord：中国城市经纬度
        如：[
                ['北京', '116.407526', '39.90403'],
                ['天津', 117.200983, 39.084158],
                ...
            ]
    """
    cities_coord = [[]]
    # 读取中国34个省会城市的经纬度坐标
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            # 每行字符串去掉换行符，再以制表符分割，得到['北京', '116.407526', '39.90403']
            tmp = line.rstrip('\n').split('\t')
            city = []  # 最后得到city= ['北京', 116.407526, 39.90403]
            for x in range(len(tmp)):
                if x == 0:
                    city.append(tmp[x])
                else:
                    city.append(float(tmp[x]))
            cities_coord.append(city)
    return cities_coord


def geo_distance(city1, city2):
    """
    针对实验 2：运输机航线
    根据公式计算两个城市之间距离（单位公里）
    Args:
        city1: 如['北京', 116.407526, 39.90403]
        city2: 如['天津', 117.200983, 39.084158]

    Returns:
        distance：城市间距离
    """
    lng1, lat1 = city1[1], city1[2]
    lng2, lat2 = city2[1], city2[2]
    # 经度1，纬度1，经度2，纬度2 （十进制度数）, 返回千米单位的距离
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
    distance = round(distance / 1000, 4)
    return distance


class Graph:
    def __init__(self, filename):
        """
        类的初始化
        """
        # vertex_num：总的顶点数；edges_matrix：由[点1, 点2, 两者距离]组成
        self.vertex_num = 0
        self.edges_matrix = 0
        self.filename = filename  # 读取文件名称
        """
        主要是针对实验一：检查点路线
        adj_matrix：邻接矩阵
        par_matrix: 记录路径
        par_matrix维度和adj_matrix维度一样
        """
        self.adj_matrix = []
        self.par_matrix = []

    def init_nodes(self):
        """
        主要是针对实验一：检查点路线
        初始化(与后面使用floyd算法一起用)
        Returns:

        """
        # vertex_num：总的顶点数；edges_matrix：由[点1, 点2, 两者距离]组成
        self.vertex_num, self.edges_matrix = read_nodes_file(self.filename)
        """
        adj_matrix：邻接矩阵
        adj_matrix维度(vertex_num+1)x(vertex_num+1)，目的是矩阵行列直接为对应顶点下标，不用做+1操作变换为原来顶点下标
        adj_matrix初始化时对角线都定义为0，其他定义为INF
        """
        self.adj_matrix = [[(lambda x: 0 if x[0] == x[1] else INF)([i, j]) for
                            j in range(self.vertex_num + 1)] for i in range(self.vertex_num + 1)]
        """
        par_matrix: 记录路径
        par_matrix维度和adj_matrix维度一样
        """
        self.par_matrix = []

        # 无向图的邻接矩阵adj_matrix由edges_matrix赋值
        for u, v, c in self.edges_matrix:
            self.adj_matrix[u][v] = c
            self.adj_matrix[v][u] = c

        # 无向图的par_matrix初始化
        for _ in range(self.vertex_num + 1):
            tmp_list = []
            for i in range(self.vertex_num + 1):
                tmp_list.append(i)
            self.par_matrix.append(tmp_list)

    def short_path_floyd(self):
        """
        主要是针对实验一：检查点路线
        floyd算法
        Returns:

        """
        self.init_nodes()
        for k in range(1, self.vertex_num + 1):
            for i in range(1, self.vertex_num + 1):
                for j in range(1, self.vertex_num + 1):
                    if self.adj_matrix[i][k] + self.adj_matrix[k][j] < self.adj_matrix[i][j]:
                        self.adj_matrix[i][j] = self.adj_matrix[i][k] + self.adj_matrix[k][j]
                        self.par_matrix[i][j] = self.par_matrix[i][k]

    def print_path(self, i, j):
        """
        主要是针对实验一：检查点路线
        打印 i 到 j 两点之间最短路径
        Args:
            i: 起点
            j: 终点

        Returns:
        """
        k = self.par_matrix[i][j]
        # print(i, end='')
        while k != j:
            print(" ——> %d" % k, end='')
            k = self.par_matrix[k][j]
        print(" ——> %d" % j, end="")

    def print_adj_matrix(self):
        """
        主要是针对实验一：检查点路线(方便测试，实际没用到)
        打印邻接矩阵
        Returns:

        """
        for i in range(self.vertex_num + 1):
            for j in range(self.vertex_num + 1):
                print(self.adj_matrix[i][j], end="\t")
            print()

    def save_adj_matrix(self):
        """
        主要是针对实验一：检查点路线(方便测试，实际没用到)
        保存邻接矩阵到文件中
        Returns:

        """
        with open('./data/adj_matrix.txt', 'w+', encoding='utf-8') as file:
            for i in range(self.vertex_num + 1):
                for j in range(self.vertex_num + 1):
                    file.write(str(self.adj_matrix[i][j]) + "\t")
                file.write("\n")


if __name__ == "__main__":
    # 测试实验二：运输机航线
    cities = read_china_cities_coord("./data/provincial_capital.txt")
    print(cities)
