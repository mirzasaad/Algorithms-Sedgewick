# from tabulate import tabulate
from collections import namedtuple
from tabulate import tabulate

from elemination import BaseBallElimination

Board = namedtuple('Board', 'teams board headers team_size')

def divisionOne():
    teams = ['Atlanta', 'Philadelphia', 'New_York', 'Montreal']
    headers = ['Win', 'Losses', 'Remaining'] + teams
    board = [
        ['Atlanta',       83, 71, 8, 0, 1, 6, 1],
        ['Philadelphia',  80, 79, 3, 1, 0, 0, 2],
        ['New_York',      78, 78, 6, 6, 0, 0, 0],
        ['Montreal',      77, 82, 3, 1, 2, 0, 0],
    ]

    return Board(teams, board, headers, 4)


def divisionTwo():
    teams = ['New_York', 'Baltimore', 'Boston', 'Toronto', 'Detroit']
    headers = ['Win', 'Losses', 'Remaining'] + teams
    board = [
        ['New_York', 75, 59, 28,   0, 3, 8, 7, 3],
        ['Baltimore', 71, 63, 28,   3, 0, 2, 7, 7],
        ['Boston', 69, 66, 27,   8, 2, 0, 0, 3],
        ['Toronto', 63, 72, 27,   7, 7, 0, 0, 3],
        ['Detroit', 49, 86, 27,   3, 7, 3, 3, 0],
    ]

    return Board(teams, board, headers, 5)

print(tabulate(divisionOne().board, headers=divisionOne().headers))
division = BaseBallElimination(divisionOne())

for team in division.teams():
  if division.isEliminated(team):
    print(team + " is eliminated by the subset R =>> " + str(division.certificateOfElimination(team)))
  else:
    print(team + " is not eliminated")

print()
print('======================================================================================================')
print('======================================================================================================')
print()

division = BaseBallElimination(divisionTwo())

print(tabulate(divisionTwo().board, headers=divisionTwo().headers))
print()

for team in division.teams():
  if division.isEliminated(team):
    print(team + " is eliminated by the subset R =>> " + str(division.certificateOfElimination(team)))
  else:
    print(team + " is not eliminated")