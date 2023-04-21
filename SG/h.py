import humansize

size = 1073741824  # 1 GB 的字节数

# 将字节数转换为人类可读的格式
print(humansize.approximate_size(size, binary=True))  # 输出：1.0 GiB
print(humansize.approximate_size(size, binary=False))  # 输出：1.1 GB
