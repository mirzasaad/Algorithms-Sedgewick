from collections import deque, defaultdict


class DepthFirstOrder():
    def __init__(self, G):
        self.G = G
        self._pre = deque([])
        self._post = deque([])
        self._reverse_post = deque([])
        self._marked = defaultdict(bool)

        for v in G.vertices():
            if not self._marked[v]:
                self.dfs(v)

    def dfs(self, v):
        marked, G, pre, post, reverse_post = self._marked, self.G, self._pre, self._post, self._reverse_post

        marked[v] = True
        pre.append(v)

        for w in G.get_adjacent_vertices(v):
            if not marked[w]:
                marked[w] = True
                self.dfs(w)

        post.append(v)
        reverse_post.appendleft(v)

    def prefix(self):
        return self._pre

    def postfix(self):
        return self._post

    def reverse_postfix(self):
        return self._reverse_post
