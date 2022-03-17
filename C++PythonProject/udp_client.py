import socket
import time

if __name__=='__main__':
    udp_client_socket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    data='C:\\Users\\lenovo\\Desktop\\开阳线_下行_2019-11-08_未知车号_正_K5_K8_1.repc'
    udp_client_socket.sendto(data.encode(),('127.0.0.1',8888))

    data_recv,addr=udp_client_socket.recvfrom(1024)
    print(data_recv.decode(),addr)
    udp_client_socket.close()


