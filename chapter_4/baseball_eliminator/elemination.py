
from copy import deepcopy
from ford_fulkerson import FordFulkerson, FlowNetwork, FlowEdge
import sys
sys.path.append('../chapter_4')

INFINTY = (1<<63) - 1

class BaseBallElimination():
    def __init__(self, _input):
        self._teamSize = _input.team_size
        self._g = [[0 for i in range(self._teamSize)]
                   for j in range(self._teamSize)]
        self._teamNames = deepcopy(_input.teams)
        self._teamNameToId = {}
        self._wins = [0] * self._teamSize
        self._losses = [0] * self._teamSize
        self._remaining = [0] * self._teamSize

        for i, team in enumerate(_input.board):
            self._teamNameToId[team[0]] = i
            self._wins[i] = team[1]
            self._losses[i] = team[2]
            self._remaining[i] = team[3]

            for j in range(self._teamSize):
                self._g[i][j] = team[4 + j]

    # subset R of teams that eliminates given team; null if not eliminated
    def certificateOfElimination(self, team):
        teamNames, teamSize = self._teamNames, self._teamSize
        wins, remaining, against = self.wins, self.remaining, self.against

        # check trivial elimination
        for i, t in enumerate(teamNames):
            if team == t:
                continue
            if wins(team) + remaining(team) < wins(t):
                return [t]

        totalGames = 0

        source = 'source'
        target = 'target'

        network = FlowNetwork()
        add_edge = network.add_edge

        # build network check lecture for nodes
        for i, team_one in enumerate(teamNames):
            if team_one == team:
                continue
            for j in range(i + 1, teamSize):
                team_two = teamNames[j]
                if team_two == team:
                    continue
                # remaining games b/w team_one and team_two
                remaining_games_x = against(team_one, team_two)

                add_edge(FlowEdge(source, team_one + '-VS-' +
                         team_two, 0, remaining_games_x))
                add_edge(FlowEdge(team_one + '-VS-' +
                         team_two, team_one, 0, INFINTY))
                add_edge(FlowEdge(team_one + '-VS-' +
                         team_two, team_two, 0, INFINTY))
                totalGames += remaining_games_x

            # most possibe wins of given team(params) minus the current team
            most_wins_possible = wins(team) + remaining(team) - wins(team_one)
            add_edge(FlowEdge(team_one, target, 0, most_wins_possible))

        ford = FordFulkerson(network, source, target)

        if ford.max_flow() == totalGames:
            return None

        result = []
        for team in teamNames:
            if ford.inCut(team):
                result.append(team)

        return result

    # number of teams
    def numberOfTeams(self):
        return self._teamSize

    # all teams
    def teams(self):
        return self._teamNames

    # number of wins for given team
    def wins(self, team):
        if not team in self._teamNameToId:
            raise Exception('Invalid team name')
        return self._wins[self._teamNameToId[team]]

    # number of losses for given team
    def losses(self, team):
        if not team in self._teamNameToId:
            raise Exception('Invalid team name')
        return self._losses[self._teamNameToId[team]]

    # number of remaining games for given team
    def remaining(self, team):
        if not team in self._teamNameToId:
            raise Exception('Invalid team name')
        return self._remaining[self._teamNameToId[team]]

    # number of remaining games between team1 and team2
    def against(self, t1, t2):
        if not t1 in self._teamNameToId:
            raise Exception('Invalid team1 name')
        if not t2 in self._teamNameToId:
            raise Exception('Invalid team2 name')
        return self._g[self._teamNameToId[t1]][self._teamNameToId[t2]]

    # is given team eliminated?
    def isEliminated(self, team):
        return self.certificateOfElimination(team) != None

