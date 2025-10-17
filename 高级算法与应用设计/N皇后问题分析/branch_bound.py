from collections import deque

EPOCH = 8


class Node:
    def __init__(self, row, col, md, sd):
        self.row = row
        self.col = col or set()
        self.md = md or set()
        self.sd = sd or set()


for n in range(1, EPOCH + 1):
    cnt = 0
    q = deque([Node(0, None, None, None)])

    while q:
        now: Node = q.popleft()
        row = now.row
        if row == n:
            cnt += 1
            continue
        for i in range(n):
            if i not in now.col and row - i not in now.md and row + i not in now.sd:
                q.append(Node(row + 1, now.col | {i}, now.md | {row - i}, now.sd | {row + i}))
    print(n, cnt)
