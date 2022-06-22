import socket
sock = socket.create_connection(('localhost', 50040))
data = sock.recv(1024)
amount_received = len(data)
print('received "%s"' % data)
print('len "%d"' % amount_received)

try:
    
    message = 'START\r\n\r\n'
    message = bytes(message, 'utf-8')
    sock.sendall(message)

    data = sock.recv(1024)
    amount_received = len(data)
    print('received "%s"' % data)
    print('len "%d"' % amount_received)

finally:
    pass

#     sock.close()