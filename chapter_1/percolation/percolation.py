from union_find import WeightedUnionFind
"""
Builds a N*N sized WeightedQuickUnionUF grid to mock create a simple percolation system. Initially all
nodes in the grid are blocked and must be opened. The grid is considered to percolate when there is a
connection from an open node on the top row to an open node on the bottom row.
Whether a node is open or not is kept in an array. All connections are done through a WeightedQuickUnionUF object.
We have a second WeightedQuickUnionUF object for checking fullness so as to not run into the backwash issue.
"""


class Percolation:
    """
    Initialises an N * N WeightedQuickUnionUF object plus two extra nodes for the virtual top and virtual bottom
    nodes. Creates an internal boolean array to keep track of whether a node is considered open or not.
    Also initialises a second N * N WeightedQuickUnionUF object plus one extra node as a second collection to check
    for fullness and avoid the backwash issue.
    """

    def __init__(self, n):
        self.N = n
        self.grid = WeightedUnionFind(n * n + 2)
        self.full = WeightedUnionFind(n * n + 1)
        self.top = self.get_single_index(n, n) + 1
        self.bottom = self.get_single_index(n, n) + 2
        self.open_nodes = [False] * (n * n)

    def get_single_index(self, row, col):
        return (self.N * (row - 1) + col) - 1

    def is_valid_index(self, row, col):
        return 0 < row <= self.N and 0 < col <= self.N

    def is_open(self, row, col):
        return self.open_nodes[self.get_single_index(row, col)]
    """
  Sets a given node coordinates to be open (if it isn't open already). First is sets the appropriate index of the "openNodes" array to be true and then attempts to union with all adjacent open nodes (if any).
  If the node is in the first row then it will union with the virtual top node. If the node is in the last row then it will union with the virtual bottom row.
  This does connections both for the internal "grid" WeightedQuickUnionUF as well as the "full" WeightedQuickUnionUF,
  """

    def open(self, row, col):
        if not self.is_valid_index(row, col):
            raise Exception('Out of index')

        if self.is_open(row, col):
            return

        index = self.get_single_index(row, col)
        self.open_nodes[index] = True

        # Node is in the top row. Union node in `grid` and `full` to the virtual top row.
        if row == 1:
            self.grid.union(self.top, index)
            self.full.union(self.top, index)

        # Node is in the bottom row. Only union the node in `grid` to avoid backwash issue.
        if row == self.N:
            self.grid.union(self.bottom, index)

        # Union with the node above the given node if it is already open
        if self.is_valid_index(row - 1, col) and self.is_open(row - 1, col):
            self.grid.union(self.get_single_index(row - 1, col), index)
            self.full.union(self.get_single_index(row - 1, col), index)

        # Union with the node to the right of the given node if it is already open
        if self.is_valid_index(row, col + 1) and self.is_open(row, col + 1):
            self.grid.union(self.get_single_index(row, col + 1), index)
            self.full.union(self.get_single_index(row, col + 1), index)

        # Union with the node below the given node if it is already open
        if self.is_valid_index(row + 1, col) and self.is_open(row + 1, col):
            self.grid.union(self.get_single_index(row + 1, col), index)
            self.full.union(self.get_single_index(row + 1, col), index)

        # Union with the node to the left of the given node if it is already open
        if self.is_valid_index(row, col - 1) and self.is_open(row, col - 1):
            self.grid.union(self.get_single_index(row, col - 1), index)
            self.full.union(self.get_single_index(row, col - 1), index)

    def percolates(self):
        return self.grid.connected(self.top, self.bottom)

    def get2d(self):
        grid = []
        for i in range(self.N):
            grid.append(self.open_nodes[i * self.N:i * self.N + self.N])
        return grid
