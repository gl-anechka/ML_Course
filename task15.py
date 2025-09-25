from typing import List


def hello(x = None):
    if x == "" or x is None:
        return "Hello!"
    else:
        return f'Hello, {x}!'


def int_to_roman(x):
    values = [(1, 'I'), (4, 'IV'), (5, 'V'), (9, 'IX'), (10, 'X'),
              (40, 'XL'), (50, 'L'), (90, 'XC'), (100, 'C'),
              (400, 'CD'), (500, 'D'), (900, 'CM'), (1000, 'M')]
    values = values[::-1]
    res = []
    for i in range(len(values)):
        count = x // values[i][0]
        if count > 0:
            res.append(values[i][1] * count)
            x -= values[i][0] * count
        if x == 0:
            break
    return ''.join(res)


def longest_common_prefix(x):
    if not x:
        return ''
    y = [s.lstrip() for s in x]
    if all(len(s) == 0 for s in y):
        return ''
    min_st = min(y, key=len)
    for i in range(len(min_st)):
        for s in y:
            if s[i] != min_st[i]:
                return min_st[:i]
    return min_st


def primes():
    yield 2
    n = 3
    while True:
        if all(n % p != 0 for p in range(2, int(n ** 0.5) + 1)):
            yield n
        n += 2


class BankCard:
    def __init__(self, total_sum, balance_limit = None):
        self.total_sum = total_sum
        self.balance_limit = balance_limit

    def __call__(self, sum_spent):
        if sum_spent > self.total_sum:
            raise ValueError(f"Not enough money to spend {sum_spent} dollars.")
        self.total_sum -= sum_spent
        print(f"You spent {sum_spent} dollars.")
        return self
    
    def __str__(self):
        return "To learn the balance call balance."

    @property
    def balance(self):
        if self.balance_limit is not None:
            if self.balance_limit == 0:
                raise ValueError("Balance check limits exceeded.")
            self.balance_limit -= 1
        return self.total_sum
    
    def put(self, sum_put):
        self.total_sum += sum_put
        print(f"You put {sum_put} dollars.")

    def __add__(self, other):        
        if self.balance_limit is None:
            new_limit = other.balance_limit
        elif other.balance_limit is None:
            new_limit = self.balance_limit
        else:
            new_limit = max(self.balance_limit, other.balance_limit)
        return BankCard(self.total_sum + other.total_sum, new_limit)