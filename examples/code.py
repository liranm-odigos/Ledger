class Node:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.nxt = None

def push(node, key, val):
    fresh = Node(key, val)
    fresh.nxt = node
    return fresh

def find(node, key):
    while node is not None:
        if node.key == key:
            return node.val
        node = node.nxt
    return None

OPS = {
    "+": lambda a, b: (a + b),
    "-": lambda a, b: (a - b),
    "*": lambda a, b: (a * b),
    "/": lambda a, b: (a // b) if b else 0,
    "%": lambda a, b: (a % b) if b else 0,
    "&": lambda a, b: (a & b),
    "|": lambda a, b: (a | b),
    "^": lambda a, b: (a ^ b),
    "<": lambda a, b: int(a < b),
    ">": lambda a, b: int(a > b),
}

def fold(expr, env):
    acc = 0
    for raw in expr.split(";"):
        op, lhs, rhs = raw.split(",")
        x = env[lhs] if lhs in env else int(lhs, 16)
        y = env[rhs] if rhs in env else int(rhs, 16)
        acc = OPS[op](acc if acc else x, y)
    return acc & 0xFFFFFFFF
