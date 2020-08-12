# gunicorn_config.py
import os
import logging
import logging.handlers
from logging.handlers import WatchedFileHandler
from logging.handlers import WatchedFileHandler
import multiprocessing

bind = '0.0.0.0:5001'      # 绑定ip和端口号
daemon = True
backlog = 100                # 监听队列
chdir = '/root/albumy'  # gunicorn要切换到的目的工作目录
timeout = 6      # Workers silent for more than this many seconds are killed and restarted.
keepalive = 2   # The number of seconds to wait for requests on a Keep-Alive connection.
graceful_timeout = 20   # Timeout for graceful workers restart.
limit_request_line = 4094   # The maximum size of HTTP request line in bytes.
limit_request_fields = 100  # Limit the number of HTTP headers fields in a request.
limit_request_field_size = 1024     # Limit the allowed size of an HTTP request header field.

# worker_class = 'gevent' # 使用gevent模式，还可以使用sync 模式，默认的是sync模式

workers = 1    # 进程数
threads = 1     # 指定每个进程开启的线程数
<<<<<<< HEAD
loglevel = 'debug' # 日志级别，这个日志级别指的是错误日志的级别，而访问日志的级别无法设置
=======
loglevel = 'debug'  # 日志级别，这个日志级别指的是错误日志的级别，而访问日志的级别无法设置
>>>>>>> 1de80c31eed897935de7a90556d7c69473b6f565
access_log_format = '%(t)s %(p)s %(h)s "%(r)s" %(s)s %(L)s %(b)s %(f)s" "%(a)s"'    # 设置gunicorn访问日志格式，错误日志无法设置

"""
其每个选项的含义如下：
h          remote address
l          '-'
u          currently '-', may be user name in future releases
t          date of the request
r          status line (e.g. ``GET / HTTP/1.1``)
s          status
b          response length or '-'
f          referer
a          user agent
T          request time in seconds
D          request time in microseconds
L          request time in decimal seconds
p          process ID
"""
accesslog = "/root/albumy/gunicorn_access.log"      #访问日志文件
errorlog = "/root/albumy/gunicorn_error.log"        #错误日志文件
