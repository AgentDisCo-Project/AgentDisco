import json
import os
import random
import time
from datetime import datetime
import threading
from collections import defaultdict

import eds
import requests
from thrift.protocol.TBinaryProtocol import TBinaryProtocol
from thrift.transport import TTransport
from thrift.transport.TSocket import TSocket


class ThriftClientPool:
    """
    Thrift客户端连接池，管理多个服务实例连接
    """

    def __init__(self, refresh_interval=60):
        self.clients = defaultdict(list)  # 存储服务名称到客户端列表的映射
        self.instances_cache = {}  # 缓存服务实例信息
        self.instances_last_refresh = {}  # 记录上次刷新时间
        self.refresh_interval = refresh_interval  # 刷新间隔(秒)
        self.lock = threading.RLock()  # 线程锁，保证并发安全

    def parse_nodes(self, nodes, priority):
        """按优先级筛选节点"""
        # 过滤 weight <=0 的节点
        nodes = [node for node in nodes if node.get('weight', 1) > 0]

        if priority < 0:
            ret = nodes
        else:
            ret = []
            for node in nodes:
                if node['priority'] <= priority:
                    ret.append(node)
                else:
                    print('skip node for priority %s' % (node))
        ret.sort(key=lambda x: x['priority'], reverse=False)
        return ret

    def get_eds_service_instances(self, service, priority=-1, retry_num=3):
        """通过HTTP获取服务实例列表"""
        host = os.environ.get('EDS_HTTP_HOST', "10.0.215.163:8085")
        url = 'http://%s/endpoints?serviceName=%s' % (host, service)

        for i in range(0, retry_num):
            try:
                resp = requests.get(url, timeout=1000)
                if resp.status_code != 200:
                    continue

                ret = json.loads(resp.text)
                if ret['code'] != 0:
                    print('invalid code %s msg %s' % (ret['code'], ret['msg']))
                    continue

                infos = self.parse_nodes(ret['data']['endpoints'], priority)
                return infos
            except Exception as e:
                print('GET %s except %s' % (url, e))

            time.sleep(1)
        return []

    def get_eds_client_instances(self, service):
        """使用EDS客户端获取服务实例列表"""
        try:
            eds_client = eds.client.EdsClient()
            instances = eds_client.get_instances(service)
            return [{"address": i.address} for i in instances]
        except Exception as e:
            print(f"Error getting instances using EDS client: {e}")
            return []

    def refresh_instances(self, service_name, use_http=True, priority=-1):
        """刷新服务实例列表"""
        current_time = datetime.now()
        last_refresh = self.instances_last_refresh.get(service_name, datetime.min)

        # 判断是否需要刷新缓存
        if (current_time - last_refresh).total_seconds() > self.refresh_interval:
            with self.lock:
                if use_http:
                    instances = self.get_eds_service_instances(service_name, priority)
                else:
                    instances = self.get_eds_client_instances(service_name)

                if instances:
                    self.instances_cache[service_name] = instances
                    self.instances_last_refresh[service_name] = current_time

        return self.instances_cache.get(service_name, [])

    def create_client(self, service_name, client_class, instance, timeout=2000):
        """为指定实例创建Thrift客户端"""
        try:
            address = instance.get('address')
            ip, port = address.split(':')
            port = int(port)

            socket = TSocket(ip, port)
            socket.setTimeout(timeout)
            transport = TTransport.TBufferedTransport(socket)
            protocol = TBinaryProtocol(transport)
            transport.open()

            return {
                "client": client_class.Client(protocol),
                "transport": transport,
                "instance": instance,
                "last_used": datetime.now(),
                "is_healthy": True
            }
        except Exception as e:
            print(f"Failed to create client for {address}: {e}")
            return None

    def get_client(self, service_name, client_class, use_http=True, timeout=2000, priority=-1):
        """
        获取一个可用的Thrift客户端
        使用轮询策略选择客户端
        """
        instances = self.refresh_instances(service_name, use_http, priority)
        if not instances:
            raise EnvironmentError(f"No available instances for service {service_name}")

        # 随机选择一个实例，避免所有客户端同时选择同一个实例
        instance = random.choice(instances)

        # 创建新客户端
        client_info = self.create_client(service_name, client_class, instance, timeout)
        if not client_info:
            # 如果创建失败，尝试另一个实例
            instances.remove(instance)
            if instances:
                instance = random.choice(instances)
                client_info = self.create_client(service_name, client_class, instance, timeout)

        if not client_info:
            raise EnvironmentError(f"Failed to create client for service {service_name}")

        return client_info["client"]

    def release_client(self, client_info):
        """释放客户端连接"""
        try:
            if client_info and "transport" in client_info:
                client_info["transport"].close()
        except Exception as e:
            print(f"Error closing client connection: {e}")


