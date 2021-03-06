import random
from datetime import datetime

from flask import Flask, jsonify, render_template_string, request

from flask_caching import Cache

app = Flask(__name__)
app.config.from_pyfile("hello.cfg")
cache = Cache(app)
# cache.init_app(app, config={'CACHE_TYPE': 'simple', "CACHE_DEFAULT_TIMEOUT": 900, })
# 使用redis 可以在gunicorn 多个workers的情况下正确的缓存数据，
cache.init_app(app, config={'CACHE_TYPE': 'redis', "CACHE_DEFAULT_TIMEOUT": 900,
                            'CACHE_REDIS_HOST': '47.100.16.146',
                            'CACHE_REDIS_PORT': '6379',
                            'CACHE_REDIS_PASSWORD': '1234asdf',
                            'CACHE_REDIS_DB': '0',
                            })

#: This is an example of a cached view
@app.route("/api/now")
@cache.cached(50)
def current_time():
    return str(datetime.now())


#: This is an example of a cached function
@cache.cached(key_prefix="binary")
def random_binary():
    return [random.randrange(0, 2) for i in range(500)]


@app.route("/api/get/binary")
def get_binary():
    return jsonify({"data": random_binary()})


#: This is an example of a memoized function
@cache.memoize(60)
def _add(a, b):
    return a + b + random.randrange(0, 1000)


@cache.memoize(60)
def _sub(a, b):
    return a - b - random.randrange(0, 1000)


@app.route("/api/add/<int:a>/<int:b>")
def add(a, b):
    return str(_add(a, b))


@app.route("/api/sub/<int:a>/<int:b>")
def sub(a, b):
    return str(_sub(a, b))


@app.route("/api/cache/delete")
def delete_cache():
    cache.delete_memoized("_add", "_sub")
    return "OK"


@app.route("/html")
@app.route("/html/<foo>")
def html(foo=None):
    if foo is not None:
        cache.set("foo", foo)
    return render_template_string(
        "<html><body>foo cache: {{foo}}</body></html>", foo=cache.get("foo")
    )

import hashlib
import base64
import json
@app.route('/api/cache/setimage', methods=['POST'])
def setimage():
    if request.method == 'POST':
        img_name = request.values.get('img_name')
        cache_data_js = cache.get(img_name)
        msg = {'img_name': img_name, }
        if cache_data_js is None:
            img_b64 = request.values.get('img_b64')
            img_md5 = hashlib.md5(base64.b64decode(img_b64)).hexdigest()
            cache_data = {'img_b64': img_b64, 'img_md5': img_md5, 'cache_times': 1}
            cres = cache.set(img_name, json.dumps(cache_data))
            msg['msg'] = 'new cached is {}, cache set {}, img_md5 {}, cache_times {}'.format(img_name, cres, img_md5, cache_data.get('cache_times'))
        else:
            cache_data = json.loads(cache_data_js)
            img_md5 = cache_data.get('img_md5')
            cache_times = cache_data.get('cache_times')
            cache_data['cache_times'] = cache_times + 1
            cres = cache.set(img_name, json.dumps(cache_data))
            msg['msg'] = 'old cached is {}, cache set {}, img_md5 {}, cache_times {}'.format(img_name, cres, img_md5, cache_data.get('cache_times'))
        print(msg)
        return jsonify(msg)


@app.route("/template")
def template():
    dt = str(datetime.now())
    return render_template_string(
        """<html><body>foo cache:
            {% cache 60, "random" %}
                {{ range(1, 42) | random }}
            {% endcache %}
        </body></html>"""
    )

print('all have been load')


if __name__ == "__main__":
    app.run()
