import h5py as h5

def getindex(f):
    f = h5.File(f, 'r')['dataMatrix'][:].tolist()
    id = {}
    i = 0
    for x in f:
        id[tuple(x)] = i
        i = i + 1
    return id

def get_variable_index(f):
    id = {}
    i = 0
    for x in f:
        id[tuple(x)] = i
        i = i + 1
    return id
