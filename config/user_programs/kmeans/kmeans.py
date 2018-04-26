import re
# FIXME very bad code for userprog

def split_iter(string):
    return (x.group(0) for x in re.finditer(r"[A-Za-z']+", string))

def test():
    return 'test func'

def data_header(data):
    # return next(split_iter(data)).split()
    d = data.split('\n')
    return d[0].split()

def split_data(data):

    # count = int(data[0].strip())
    # it = split_data(data)
    # skip first line
    # next(it)
    d = data.split('\n')
    generator = (line.split()[1:] for line in d)

    return generator
