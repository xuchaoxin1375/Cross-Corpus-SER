import mmap

with open("example.txt", "r+b") as f:
    # 将文件映射到内存中
    mm = mmap.mmap(f.fileno(), 0)

    # 在内存中读取文件内容
    file_content = mm.read()

    # 在内存中修改文件内容
    mm[0:5] = b"Hello"

    # 将修改后的内容写回到文件中
    mm.flush()

    # 解除文件内存映射
    mm.close()