#!/usr/bin/python
# coding=utf-8

import time
import threading
import asyncio


async def hello(t):
    print('Hello world! {}, t={}'.format(threading.currentThread(), t))
    res_asy = await asyncio.sleep(t)
    print('Hello again! {}, res_asy={}, t={}'.format(threading.currentThread(), res_asy, t))


loop = asyncio.get_event_loop()
tasks = [hello(1), hello(2), hello(3), hello(4)]
t1 = time.time()
loop.run_until_complete(asyncio.wait(tasks))
t2 = time.time()
loop.close()
print('T={}'.format(t2 - t1))
