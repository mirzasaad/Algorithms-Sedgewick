from collections import defaultdict, deque
from Digraph import Digraph
from DirectedCycle import DirectedCycle
from DirectedDegrees import DirectedDegrees
from ShortestAncestralPaths import ShortestAncestralPaths

class WordNet():
    # constructor takes the name of the two input files
    def __init__(self, synsets, hypernyms):
        self.G = Digraph()
        self._sysnets = {}
        self._nouns = defaultdict(list)
        self._size = 0

        self.__init_synsets(synsets)
        self.__build_digraph(hypernyms)

        self._sap = ShortestAncestralPaths(self.G)

        # assert self.is_DAG()
        # assert self.is_graph_rooted()

    def is_DAG(self):
        return not DirectedCycle(self.G).has_cycle()

    def is_graph_rooted(self):
        count, id = 0, None
        for v in self.G.vertices():
            if (len(self.G.get_adjacent_vertices(v)) == 0):
                id, count = v, count + 1
            if count > 1:
                return False

        sinks = DirectedDegrees(self.G).sinks()
        return len(sinks) == 1 and sinks[0] == id

    def __init_synsets(self, synsets):
        with open(synsets, 'r') as f:
            for line in f:
                split = line.split(',')
                id, sysnet = int(split[0]), split[1]
                self._sysnets[id] = sysnet

                for noun in sysnet.split(' '):
                    self._nouns[noun].append(id)
                self._size += 1

    def __build_digraph(self, hypernyms):
        count = 0
        with open(hypernyms, 'r') as f:
            for line in f:
                edges = deque(line.split(','))
                id = edges.popleft()
                for w in edges:
                    self.G.add_edge(int(id), int(w))

                count += 1

    # returns all WordNet nouns
    def nouns(self):
        return self._nouns.keys()

    # is the word a WordNet noun?
    def isNoun(self, word):
        assert word is not None
        return self._nouns.get(word, None) is not None

    # distance between nounA and nounB (defined below)
    def distance(self, noun_a, noun_b):
        assert self.isNoun(noun_a) is not None
        assert self.isNoun(noun_b) is not None

        distance = self._sap.length(self._nouns.get(noun_a), self._nouns.get(noun_b))

        return distance

    # synset (second field of synsets.txt) that is the common ancestor of nounA and nounB in a shortest ancestral path (defined below)
    def sap(self, noun_a, noun_b):
        assert self.isNoun(noun_a) is not None
        assert self.isNoun(noun_b) is not None
 
        ancestor = self._sap.ancestor(self._nouns.get(noun_a), self._nouns.get(noun_b))

        if ancestor == -1:
            raise ValueError('"Nouns do not have common ancestor"')

        return self._sysnets.get(ancestor, -1)
