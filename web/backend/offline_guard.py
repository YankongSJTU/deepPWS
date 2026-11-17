"""Enforce local-only network operations inside offline app and worker processes."""
import socket,ipaddress

def enable():
    if getattr(socket,'_faceba_local_only',False):return
    original_connect=socket.socket.connect
    original_connect_ex=socket.socket.connect_ex
    original_getaddrinfo=socket.getaddrinfo
    def allowed(address):
        if not isinstance(address,tuple):return True
        host=address[0]
        if host=='localhost':return True
        try:return ipaddress.ip_address(host).is_loopback
        except ValueError:return False
    def connect(self,address):
        if not allowed(address):raise OSError('Offline mode: external network connections are disabled')
        return original_connect(self,address)
    def connect_ex(self,address):
        if not allowed(address):raise OSError('Offline mode: external network connections are disabled')
        return original_connect_ex(self,address)
    def getaddrinfo(host,port,*args,**kwargs):
        if host is not None and not allowed((host,port)):
            raise OSError('Offline mode: external DNS resolution is disabled')
        return original_getaddrinfo(host,port,*args,**kwargs)
    socket.socket.connect=connect;socket.socket.connect_ex=connect_ex;socket.getaddrinfo=getaddrinfo
    socket._faceba_local_only=True
