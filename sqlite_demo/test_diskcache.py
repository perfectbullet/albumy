import diskcache

#设置文件到内存中。=。  速度还不错
cache = diskcache.Cache("/tmp")


#写入缓存
cache.set(
    key="key",
    value="value1",
    expire="100",
    tag="namespace1"
)


#手动清除超过生存时间的缓存
cache.cull()

#清除命名空间的缓存
cache.evict()