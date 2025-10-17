EPOCH = 8


def judge(q: list[int]) -> bool:
    for i in range(len(q)):
        for j in range(i + 1, len(q)):
            if q[i] == q[j] or j - i == abs(q[j] - q[i]):
                return False
    return True


for n in range(1, EPOCH + 1):
    tot = n**n
    res = 0
    for cnt in range(tot):
        pos = []
        while cnt:
            pos.append(cnt % n)
            cnt //= n
        pos += (n - len(pos)) * [0]
        if judge(pos):
            res += 1
    print(n, res)
