import os
import os.path as osp

'''
路径的操作
'''
os.listdir
'''
返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序。 它不包括 '.' 和'..' 即使它在文件夹中
'''

osp.join(path,path)
'''
拼接路径
'''

osp.isdir()
'''
判断是否存在这个目录,不是判断文件是否存在
'''

osp.exists()
'''
判断文件是否存在
'''

os.path.split()
'''
按照路径将文件名和路径分开
如果给出的是一个目录和文件名，则输出路径和文件名
如果给出的是一个目录名，则输出路径和为空文件名
'''

os.path.splitext
'''
将文件名和扩展名分开
'''

os.path.realpath(__file__)
'''
获取当前py文件的绝对路径
'''

os.path.expanduser（）
'''
把path中包含的"~"和"~user"转换成用户目录
'''



