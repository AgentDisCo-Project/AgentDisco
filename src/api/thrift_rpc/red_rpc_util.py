from poi_rpc.thrift_rpc.thrift_client_pool import ThriftClientPool

# 创建全局连接池实例
client_pool = ThriftClientPool()


def create_thrift_client_http(service_name, client_class, timeout=2000, priority=-1):
    """通过HTTP方式获取服务并创建Thrift客户端"""
    return client_pool.get_client(
        service_name, client_class, use_http=True, timeout=timeout, priority=priority
    )


def create_thrift_client(service_name, client_class, timeout=5000):
    """使用EDS客户端获取服务并创建Thrift客户端"""
    return client_pool.get_client(
        service_name, client_class, use_http=False, timeout=timeout
    )
