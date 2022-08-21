from typing import List
from prettytable import PrettyTable

def printTable(matrix: List[List[any]], field_names=[]):
    x = PrettyTable()
    x.field_names = field_names
    for row in matrix:
        x.add_row(row)

    print(x)
    