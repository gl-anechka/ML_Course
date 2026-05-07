def check(x, file):
    arr = x.lower().split()

    count = {}
    for s in arr:
        count[s] = count.get(s, 0) + 1
    
    words = sorted(count.items())
    with open(file, 'w') as f:
        for i, j in words:
            f.write(f'{i} {j}\n')