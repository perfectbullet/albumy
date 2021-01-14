import time

import redis
from flask import Flask, request

app = Flask(__name__)
cache = redis.Redis(host='redis', port=6379)


def get_hit_count():
    retries = 5
    while True:
        try:
            return cache.incr('hits')
        except redis.exceptions.ConnectionError as exc:
            if retries == 0:
                raise exc
            retries -= 1
            time.sleep(0.5)


@app.route('/')
def hello():
    count = get_hit_count()
    return 'Hello World! I have been seen {} times.\n'.format(count)


@app.route('/getheaderip')
def getheaderip():
    """ 获取客户的ip """
    ip = request.headers.get('X-Forwarded-For', 'no ip')
    ip2 = request.remote_addr
    return 'X-Forwarded-For is {}, remote_addr is {}'.format(ip, ip2)


app.run()
