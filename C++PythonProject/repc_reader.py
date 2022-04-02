import struct
from typing import List

# repc文件文件头结构
class LASheader:
    def __init__(self):
        self.file_signature = ''  # 文件签名
        self.file_id = 0  # 文件编号
        self.major = 0  # 文件主版本号
        self.minor = 0  # 文件副版本号
        self.hardware_system = ''  # 采集硬件系统名称、版本
        self.software_system = ''  # 采集软件系统名称、版本
        self.railway_name = ''  # 线路名
        self.railway_num = ''  # 线路编号
        self.railway_direction = ''  # 行别
        self.mile_direction = ''  # 增、减里程
        self.start_mileage = 0  # 开始里程
        self.train_no = ''  # 检测车号
        self.position = ''  # 位端
        self.time = 0  # 开始时间（1970年1月1日0点0分距开始时间的毫秒数）
        self.header_size = 0  # 文件头长度
        self.point_data_size = ''  # 单个点数据字节数
        self.point_number = 0  # 点总数量
        self.x_scale_factor = 0  # X尺度因子
        self.y_scale_factor = 0  # Y尺度因子
        self.z_scale_factor = 0  # Z尺度因子
        self.x_offset = 0  # X偏移值
        self.y_offset = 0  # Y偏移值
        self.z_offset = 0  # Z偏移值
        self.max_x = 0  # 里程真实最大X
        self.min_x = 0  # 里程真实最小X
        self.max_y = 0  # 真实最大Y
        self.min_y = 0  # 真实最小Y
        self.max_z = 0  # 真实最大Z
        self.min_z = 0  # 真实最小Z
        self.rev = ''  # 预留

# 点数据
class LASPoint:
    def __init__(self):
        self.point_source_id = 0  # 点源ID(点序号)
        self.x = 0  # X坐标
        self.y = 0  # Y坐标
        self.z = 0  # Z坐标

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

###读取repc头文件
def readLASHeader(filename) -> LASheader:
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

def readLASPoint(filename,index) -> LASPoint:
    '''
    读取一个点

    :param index: 点的索引，从1开始
    :return: 读取的点
    '''
    point = LASPoint()
    f = open(filename, "rb")
    f.seek(350 + 43 * (index - 1))
    # 读取一个数据点的id x y z值
    s = f.read(16)
    # 使用unpack将读取到的字节解析为指定类型数据
    point.point_source_id, point.x, point.y, point.z = struct.unpack(
        "<L3l", s)
    f.close()
    return point

def readRepcPoints(filename,index, num) -> List[LASPoint]:  # 从index=0开始 num表示想读取的点的个数
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
        # 读取一个数据点的id x y z值
        s = f.read(16)
        # 使用unpack将读取到的字节解析为指定类型数据
        point.point_source_id, point.x, point.y, point.z = struct.unpack(
            "<L3l", s)
        points.append(point)
        f.seek(27,1)

    f.close()
    return points

#读取所有原始点数据(恢复缩放后的数据)
def readAllData(filename:str):
    header = readLASHeader(filename)
    points = readRepcPoints(filename,0,header.point_number)
    data = []
    for i in range(len(points)):
        point_id = points[i].point_source_id
        x = points[i].x * header.x_scale_factor
        y = points[i].y * header.y_scale_factor
        z = points[i].z * header.z_scale_factor
        data.append([point_id,x,y,z])
    return data


def writeOnePoint(filename:str,point:LASPoint):
    '''
    写入一个指定的点

    Args:
        filename: 文件路径
        point: 待写入的点
    Returns: None

    '''
    f=open(filename,'rb+')
    f.seek(350+43*(point.point_source_id-1))
    #将待写入数据打包
    s=struct.pack("<L3l",point.point_source_id,point.x,point.y,point.z)
    f.write(s)
    f.close()


def writeAllData(filename:str,points:List[LASPoint]):
    '''
    写入处理后的全部点数据

    Args:
        filename: 文件路径
        points: 全部点数据
    Returns: None

    '''
    f=open(filename,'rb+')
    f.seek(350,0)
    for i in points:
        s=struct.pack("<L3l",i.point_source_id,i.x,i.y,i.z)
        f.write(s)
        #跳过后续不需要写入的位置
        f.seek(27,1)
    f.close()


# repc_root = 'test_data/test.repc'
#
# points=[]
# for i in range(1000000):
#     point=LASPoint()
#     point.point_source_id=i+1
#     point.x=1000
#     points.append(point)
#
# writeAllData(repc_root,points)
# data = readAllData(repc_root)
# print(len(data))


# writeOnePoint(repc_root,p)
# p=readLASPoint(repc_root,5)
# print(p.point_source_id,p.x,p.y,p.z)


