from socket import *
from time import ctime


def server():
    HOST = ''
    PORT = 10521
    ADDR = (HOST, PORT)
    server_socket = socket(AF_INET, SOCK_STREAM)
    server_socket.bind(ADDR)
    server_socket.listen(5)
    while True:
        print('Waiting for connecting ......')
        tcpclientsocket, addr = server_socket.accept()
        print('Connected by ', addr)
        while True:
            data = tcpclientsocket.recv(1024)
            if not data:
                break
            print(data)
            data = input('I>')
            tcpclientsocket.send('[%s]%s' % (ctime(), data))
    tcpclientsocket.close()
    server_socket.close()


server()