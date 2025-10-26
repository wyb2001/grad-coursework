def judge(q: list[int]) -> bool:
    for i in range(len(q)):
        for j in range(i + 1, len(q)):
            if q[i] == q[j] or j - i == abs(q[j] - q[i]):
                return False
    return True


def dfs(n, t) -> None:
    global cnt
    if t >= n:
        if judge(pos):
            cnt += 1
        return
    for i in range(n):
        if not is_used[i]:
            valid = True
            for j in range(t):
                if j-pos[j] == t-i or j+pos[j]==t+i:
                    valid = False
                    break
            if valid:
                is_used[i] = True
                pos[t] = i
                dfs(n, t + 1)
                is_used[i] = False
    return


n=int(input())
cnt = 0
is_used = [False] * n
pos = [0] * n
dfs(n, 0)
print(cnt)
