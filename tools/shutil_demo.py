import shutil, os

# folder1 = os.path.join(os.getcwd(), "aaa")
# folder2 = os.path.join(os.getcwd(), "bbb")
# shutil.move(folder1, folder2)
# # 示例二，将src文件移动至dst文件夹下面，如果bbb文件夹不存在，则变成了重命名操作
# file1 = os.path.join(os.getcwd(), "aaa.txt")
# folder2 = os.path.join(os.getcwd(), "bbb")
# shutil.move(file1, folder2)
# # 示例三，将src文件重命名为dst文件(dst文件存在，将会覆盖)
# file1 = os.path.join(os.getcwd(), "aaa.txt")
# file2 = os.path.join(os.getcwd(), "bbb.txt")


import shutil,os

dir_ls = [l[i:i+n] for i in range(0, len(l), n)]

for i in range(159):
    new_dir = str(i) + 'part'
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    for dr in dir_ls[i]:
        if 'part' not in dr:
            print(dr)
            shutil.move(dr, os.path.join(new_dir, dr))





import shutil, os

n = 160
base_dir = os.getcwd()
format = "zip"

for d in os.listdir('.'):
    full_dir = os.path.join(base_dir, d)
    shutil.make_archive(full_dir, format, full_dir)
    shutil.rmtree(full_dir)
