import fcntl
import os
import asyncio

import numpy as np
import cv2


def bytes2np(bytesarray):
    """
    bytes two numpy->array
    :param bytesarray:
    :return:
    """
    nparr = np.frombuffer(bytesarray, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)   # cv2.IMREAD_COLOR in OpenCV 3.1
    return img_np


class asyncfile:
    """
    
    """
    BLOCK_SIZE = 512
    
    def __init__(self, filename, mode, loop=None):
        self.fd = open(filename, mode=mode)
        flag = fcntl.fcntl(self.fd, fcntl.F_GETFL)
        if fcntl.fcntl(self.fd, fcntl.F_SETFL, flag | os.O_NONBLOCK) != 0:
            raise OSError()
        
        if loop is None:
            loop = asyncio.get_event_loop()
        self.loop = loop
        self.rbuffer = bytearray()
    
    def read_step(self, future, n, total):
        
        res = self.fd.read(n)
        
        if res is None:
            self.loop.call_soon(self.read_step, future, n, total)
            return
        if not res:  # EOF
            nparr = np.frombuffer(bytes(self.rbuffer), np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            future.set_result(img_np)
            return
        
        self.rbuffer.extend(res)
        
        if total > 0:
            left = total - len(self.rbuffer)
            if left <= 0:
                # read to the end
                # res, self.rbuffer = self.rbuffer[:n], self.rbuffer[n:]
                nparr = np.frombuffer(bytes(self.rbuffer), np.uint8)
                img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                future.set_result(img_np)
            else:
                left = min(self.BLOCK_SIZE, left)
                self.loop.call_soon(self.read_step, future, left, total)
        else:
            self.loop.call_soon(self.read_step, future, self.BLOCK_SIZE, total)
    
    def read(self, n=-1):
        future = asyncio.Future(loop=self.loop)
        
        if n == 0:
            future.set_result(b'')
            return future
        elif n < 0:
            self.rbuffer.clear()
            self.loop.call_soon(self.read_step, future, self.BLOCK_SIZE, n)
        else:
            self.rbuffer.clear()
            self.loop.call_soon(self.read_step, future, min(self.BLOCK_SIZE, n), n)
        
        return future


async def aimread(file_path):
    af = asyncfile(file_path, mode='rb')
    aimg = await af.read()
    # print('aimg type ={}'.format(type(aimg)))
    return aimg


if __name__ == '__main__':
    tasks = [aimread('25-1028.jpg'), aimread('6-9060_2082_742.jpg')]
    task_results, tt2 = asyncio.get_event_loop().run_until_complete(asyncio.wait(tasks))
    for tk_re in task_results:
        print(tk_re.result().shape)

