#!/usr/bin/python
# coding=utf-8


def consumer():
    data = 0
    while True:
        n = yield data
        if not n:
            return
        print('[CONSUMER] Consuming %s...' % n)
        data += 1


def produce(c):
    print('start send')
    c.send(None)
    n = 0
    while n < 5:
        n = n + 1
        print('before send [PRODUCER] Producing %s...' % n)
        r = c.send(n)
        print('after send [PRODUCER] Consumer return: %s' % r)
    c.close()


def generator_f():
    n = 1
    while n < 50:
        res = yield n
        n = n + 1
        print('res={}, n={}'.format(res, n))

#
c = consumer()
produce(c)

# gn = generator_f()
# t = gn.send(None)
# print('start generator_f and get res={}'.format(t))
# j = 0
# while j < 10:
#     j += 1
#     res_from_gn = gn.send(j*10)
#     print('res_from_gn={}'.format(res_from_gn))
# gn.close()



'''
生产者生产消息后，直接通过yield跳转到消费者开始执行，待消费者执行完毕后，切换回生产者继续生产:
注意到consumer函数是一个generator，把一个consumer传入produce后：
1. 首先调用c.send(None)启动生成器；(这里也会得到返回值)
2. 然后，一旦生产了东西，通过c.send(n)切换到consumer执行；
3. consumer通过yield拿到消息，处理，又通过yield把结果传回；
4. produce拿到consumer处理的结果，继续生产下一条消息；
5. produce决定不生产了，通过c.close()关闭consumer，整个过程结束。
整个流程无锁，由一个线程执行，produce和consumer协作完成任务，所以称为“协程”，而非线程的抢占式多任务。
最后套用Donald Knuth的一句话总结协程的特点： “子程序就是协程的一种特例。”
'''
