import doctest
from bintrees import RBTree
import json
import module_3_3
"""
Question 1
Red–black BST with no extra memory. Describe how to save the memory 
for storing the color information when implementing a red–black BST.

Answer 1
just modify the BST tree. For black node, do nothing. And for red node, 
exchange its left child and right child. In this case, a node can be 
justified red or black according to if its right child is larger than 
its left child.
"""
class MinimumDistanceWords(object):
    """
        Question 2
    Document search. Design an algorithm that takes a sequence of n 
    document words and a sequence of m query words and find the shortest 
    interval in which the m query words appear in the document in the order given. 
    The length of an interval is the number of words in that interval.

    Answer 2:
    use red black trees, find minimum, 
    keep find the floor of min_value+1 
    till u get a different word interval
    >>> items = '{"it": [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 119, 134], "was": [1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 89], "the": [2, 8, 14, 20, 26, 32, 38, 44, 50, 56, 82, 87, 93, 113], "best": [3], "of": [4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 98, 116], "times": [5, 11], "worst": [9], "age": [15, 21], "wisdom": [17], "foolishness": [23], "epoch": [27, 33], "belief": [29], "incredulity": [35], "season": [39, 45], "light": [41], "darkness": [47], "spring": [51], "hope": [53], "winter": [57], "despair": [59], "we": [60, 65, 70, 77], "had": [61, 66], "everything": [62],"before": [63, 68], "us": [64, 69], "nothing": [67], "were": [71, 78], "all": [72, 79], "going": [73, 80], "direct": [74, 81], "to": [75, 144], "heaven": [76],"other": [83], "way": [84], "in": [85, 112], "short": [86], "period": [88, 95],"so": [90], "far": [91, 122, 123, 137, 138], "like": [92], "present": [94], "that": [96, 126, 141], "some": [97], "its": [99, 104], "noisiest": [100], "authorities": [101], "insisted": [102], "on": [103], "being": [105], "received": [106],"for": [107, 110], "good": [108], "or": [109], "evil": [111], "superlative": [114], "degree": [115], "comparison": [117], "only": [118], "is": [120, 135], "a":[121, 136], "better": [124, 139], "thing": [125], "i": [127, 130, 142, 146], "do": [128], "than": [129, 145], "have": [131, 147], "ever": [132, 148], "done": [133], "rest": [140], "go": [143], "known": [149]}'
    >>> items = json.loads(items)
    >>> query = dict(items.items())
    >>> mdw = MinimumDistanceWords()
    {'distance': 1, 'known': 'far'}
    """

    def find(self, __dict):
        rb = RBTree()

        for word in __dict:
            for times in __dict[word]:
                rb.insert(times, word)
                
        max_item = rb.max_item()
        min_item = rb.min_item()
        min_interval = 1 << 63 - 1
        
        result = None

        while True:
            ceil = rb.ceiling_item(min_item[0] + 1)

            is_same_word = ceil[1] == min_item[1]
            distance = ceil[0] - min_item[0]

            if is_same_word and distance < min_interval:
                min_interval = distance
                result = (min_item, ceil)
            
            min_item = ceil

            if max_item == ceil:
                break
        
        return { "distance": result[1][0] - result[0][0], word: result[0][1] } if result else None


############
### Rb tree has some problem
############
class GeneralizedQueue(object):
    """
    Generalized queue. Design a generalized queue data type that supports all of the 
    following operations in logarithmic time (or better) in the worst case.

    1: Create an empty data structure.
    2: Append an item to the end of the queue.
    3: Remove an item from the front of the queue.
    4: Return the ith item in the queue.
    5: Remove the ith item from the queue.

    Use RedBlack Tree to support all these  operations
    """
    def __init__(self) -> None:
        self._store = module_3_3.RBTree()
        self._index = 0

    def push(self, value):
        self._store.put(self._index, value)
        self._index += 1

    def popLeft(self):
        item = self._store.delete_min()
        self._index -= 1
        return item
    
    def get(self, index):
        node = self._store.get(index)
        return node.value

    def delete(self, index):
        return self._store.delete(index)

if __name__ == '__main__':
    doctest.testmod()

