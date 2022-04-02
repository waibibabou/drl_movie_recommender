import socket
import threading
import time
from main import test

# 创建udp套接字
udp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# 绑定地址(ip,port)到套接字
udp_server_socket.bind(('', 8888))


class ReceivePath(threading.Thread):
    '''
    获取客户端发送的文件路径并保存到temp.txt中
    '''

    def __init__(self):
        super(ReceivePath, self).__init__()

    def run(self):
        f = open('temp.txt', 'r+')
        while True:
            # 接受数据
            data, addr = udp_server_socket.recvfrom(100)
            data = data.decode('ascii')
            print(f'得到的数据地址为:{data}')
            # 互斥锁
            lock.acquire()
            # 将数据写到文件末尾
            f.seek(0, 2)
            # 将文件地址以及客户端套接字写入临时文件
            f.write(data + ' ' + addr[0] + ' ' + str(addr[1]) + '\n')
            # 将数据立即写入到文件
            f.flush()
            lock.release()


class Calculate(threading.Thread):
    '''
    调用模型计算数据，并在完成后向客户端发送完成提示
    '''

    def __init__(self):
        super(Calculate, self).__init__()

    def run(self):
        f = open('temp.txt', 'r+')

        while True:
            # 互斥锁
            lock.acquire()
            s = f.readline()
            lock.release()

            # 如果读取到文件路径则调用模型计算
            if s != '':
                # 去除读取到的换行符
                s = s[:-1]
                addr = list()
                data, ip, port = s.split(' ')
                addr.append(ip)
                addr.append(int(port))
                # 计算过程
                time.sleep(5)
                #test(data)

                finish_message = 'finish!' + ' ' + data + '\0'
                # 将数据地址以及结束信号发送给客户端
                udp_server_socket.sendto(finish_message.encode('ascii'), tuple(addr))


receiveThread = ReceivePath()
calculateThread = Calculate()
#互斥锁
lock = threading.Lock()
receiveThread.start()
calculateThread.start()
