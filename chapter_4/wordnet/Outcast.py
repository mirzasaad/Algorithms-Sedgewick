class Outcast():
    def __init__(self, wordnet):
        self.wordnet = wordnet

    def outcast(self, nouns=[]):
        N, max_distance, outcast, w = len(
            nouns), float('-inf'), None, self.wordnet
        for i in range(N):
            for j in range(i + 1, N):
                if nouns[i] != nouns[j]:
                    distance = w.distance(nouns[i], nouns[j])
                    if distance > max_distance:
                        max_distance = distance
                        outcast = nouns[i]
        return outcast
