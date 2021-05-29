#!/usr/bin/python
# coding=utf-8

import time
from concurrent.futures import ThreadPoolExecutor
import asyncio
from asyncio import coroutine
import _collections_abc


def write_db(t):
    print(t)
    time.sleep(t)


async def main():
    pool = ThreadPoolExecutor()
    print('start')
    await asyncio.get_event_loop().run_in_executor(pool, write_db, 2)
    print('end')


if __name__ == "__main__":
    asyncio.run(main())


# def blocking_io():
#     # File operations (such as logging) can block the
#     # event loop: run them in a thread pool.
#     with open('/dev/urandom', 'rb') as f:
#         return f.read(100)
#
# def cpu_bound():
#     # CPU-bound operations will block the event loop:
#     # in general it is preferable to run them in a
#     # process pool.
#     return sum(i * i for i in range(10 ** 7)) / (10 ** 7)
#
#
# async def main():
#     loop = asyncio.get_running_loop()
#
#     ## Options:
#
#     # 1. Run in the default loop's executor:
#     result = await loop.run_in_executor(None, blocking_io)
#     result = str(result)
#     print('default thread pool', result)
#
#     # 2. Run in a custom thread pool:
#     with concurrent.futures.ThreadPoolExecutor() as pool:
#         result = await loop.run_in_executor(pool, blocking_io)
#         print('custom thread pool', result)
#
#     # 3. Run in a custom process pool:
#     with concurrent.futures.ProcessPoolExecutor() as pool:
#         result = await loop.run_in_executor(pool, cpu_bound)
#         print('type={}'.format(type(result)))
#         print('custom process pool', result)
#
# asyncio.run(main())