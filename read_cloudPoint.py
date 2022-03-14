import struct


# repc文件头
class LASheader:
    def __init__(self):
        super(LASheader, self).__init__()
    # file_signature = ''  # 文件签名4
    # file_id = 0  # 文件编号
    # major = 0  # 文件主版本号
    # minor = 0  # 文件副版本号
    # hardware_system = ''  # 采集硬件系统名称、版本
    # software_system = ''  # 采集软件系统名称、版本
    # railway_name = ''  # 线路名
    # railway_num = ''  # 线路编号108
    # railway_direction = ''  # 行别
    # mile_direction = ''  # 增、减里程
    # start_mileage = 0  # 开始里程
    # train_no = ''  # 检测车号
    # position = ''  # 位端
    # time = 0  # 开始时间（1970年1月1日0点0分距开始时间的毫秒数）
    # header_size = 0  # 文件头长度
    # point_data_size = ''  # 单个点数据字节数
    # point_number = 0  # 点总数量46
    # x_scale_factor = 0  # X尺度因子
    # y_scale_factor = 0  # Y尺度因子
    # z_scale_factor = 0  # Z尺度因子
    # x_offset = 0  # X偏移值
    # y_offset = 0  # Y偏移值
    # z_offset = 0  # Z偏移值
    # max_x = 0  # 里程真实最大X
    # min_x = 0  # 里程真实最小X
    # max_y = 0  # 真实最大Y
    # min_y = 0  # 真实最小Y258
    # max_z = 0  # 真实最大Z354
    # min_z = 0  # 真实最小Z96
    # rev = ''  # 预留


# 点数据
class LASPoint:
    def __init__(self):
        super(LASPoint, self).__init__()
    # point_source_id = 0  # 点源ID(点序号)
    # x = 0  # X坐标
    # y = 0  # Y坐标
    # z = 0  # Z坐标
    # intensity = 0  # 反射强度
    # return_number = 0  # 反射号(回波)
    # classification = ''  # 分类(不同设施分层管理)
    # key_point = ''  # 是否关键点
    # user_data = ''  # 用户可关联自定义数据
    # color_id = ''  # 颜色ID(根据ID区分颜色)
    # shape_id = ''  # 形状ID(根据ID区分形状)
    # time = 0  # 距头文件Time的毫秒数
    # curvature_radius = 0  # 曲率
    # super = 0  # 超高
    # longitude = 0  # 经度(预留)
    # latitude = 0  # 纬度(预留)
    # height = 0  # 高程(预留)


# repc文件路径
filename = 'C:\\Users\\lenovo\\Desktop\\开阳线_下行_2019-11-08_未知车号_正_K5_K8_1.repc'


def readLASHeader() -> LASheader:
    '''
    读取repc文件头

    :return: repc文件头数据
    '''
    header = LASheader()
    f = open(filename, "rb")
    f.seek(0)
    # 读取文件头的350个字节
    s = f.read(350)
    # 使用unpack将读取到的字节解析为指定类型数据
    header.file_signature, header.file_id, header.major, header.minor, header.hardware_system, header.software_system, header.railway_name, header.railway_num, header.railway_direction, header.mile_direction, header.start_mileage, header.train_no, header.position, header.time, header.header_size, header.point_data_size, header.point_number, \
    header.x_scale_factor, header.y_scale_factor, header.z_scale_factor, header.x_offset, header.y_offset, header.z_offset, header.max_x, header.min_x, header.max_y, header.min_y, header.max_z, header.min_z, header.rev = struct.unpack(
        "<4sH2B30s30s30s10s2Bd20sBQHBL12d100s", s)
    f.close()
    return header


def readLASPoint(index) -> LASPoint:
    '''
    读取一个点

    :param index: 点的索引，从1开始
    :return: 读取的点
    '''
    point = LASPoint()
    f = open(filename, "rb")
    f.seek(350 + 43 * (index - 1))
    # 读取一个数据点大小即43个字节的数据
    s = f.read(43)
    # 使用unpack将读取到的字节解析为指定类型数据
    point.point_source_id, point.x, point.y, point.z, point.intensity, point.return_number, point.classification, point.key_point, point.user_data, point.color_id, point.shape_id, \
    point.time, point.curvature_radius, point.super, point.longitude, point.latitude, point.height = struct.unpack(
        "<L3l2H5BL2h2LH", s)
    f.close()
    return point


def readLASPoints(index, num) -> bytes:
    '''
    读取多个点的数据，但并不做解析

    :param index: 起始点的索引，从0开始
    :param num: 想要读取的点的个数
    :return: 未作解析的字节数据
    '''
    f = open(filename, "rb")
    f.seek(350 + 43 * index)
    s = f.read(num * 43)
    f.close()
    return s


def readRepcPoints(index, num) -> list:
    '''
    读取多个点的数据并解析为list形式返回

    :param index: 起始点的索引，从0开始
    :param num: 想要读取的点的个数
    :return: 读取到的多个点
    '''
    points = []
    f = open(filename, "rb")
    f.seek(350 + 43 * index)
    for i in range(num):
        point = LASPoint()
        # 读取一个数据点大小即43个字节的数据
        s = f.read(43)
        # 使用unpack将读取到的字节解析为指定类型数据
        point.point_source_id, point.x, point.y, point.z, point.intensity, point.return_number, point.classification, point.key_point, point.user_data, point.color_id, point.shape_id, \
        point.time, point.curvature_radius, point.super, point.longitude, point.latitude, point.height = struct.unpack(
            "<L3l2H5BL2h2LH", s)
        points.append(point)
    f.close()
    return points


header = readLASHeader()
point = readLASPoint(501)
s = readLASPoints(500, 600)
points = readRepcPoints(0, 600)
print(points[0].z)
