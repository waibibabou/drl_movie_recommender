import socket
import time


if __name__=='__main__':
    #创建udp套接字
    udp_server_socket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    #绑定地址(ip,port)到套接字
    udp_server_socket.bind(('',8888))
    while True:
        #接受数据
        data,addr=udp_server_socket.recvfrom(100)
        data=data.decode('ascii')
        print(f'得到的数据地址为:{data}')
        time.sleep(3)#模型计算
        finish_message='finish!'+' '+data+'\0'

        #将数据地址以及结束信号发送给客户端
        udp_server_socket.sendto(finish_message.encode('ascii'),addr)






