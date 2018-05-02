import re
# FIXME very bad code for userprog

def split_iter(string):
    return (x.group(0) for x in re.finditer(r"[A-Za-z']+", string))

def test():
    return 'test func'

def data_header(data):

    # get first line
    index = data.find('\n')
    # split count and dim
    return list(map(int,data[:index].split()))

def split_data(data):

    # count = int(data[0].strip())
    # it = split_data(data)
    # skip first line
    # next(it)
    d = data.split('\n')

    # file layout:
    # count dim
    # 1 d1 d2 d3 .....\n
    # 2 d1 d2 d3 .....\n
    # 3 d1 d2 d3 .....\n
    ...
    generator = (list(map(float, line.split()[1:])) for line in d[1:-1])

    return generator
