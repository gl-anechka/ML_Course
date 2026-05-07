def find_modified_max_argmax(L, f):
    l = [f(x) for x in L if type(x) == int]
    return l and (m:=max(l), l.index(m)) or ()