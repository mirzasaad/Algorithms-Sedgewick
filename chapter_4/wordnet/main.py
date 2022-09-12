import doctest
from WordNet import WordNet
from Outcast import Outcast
import os

class WordNetApi(object):
    """
    >>> my_path = os.path.abspath(os.path.dirname(__file__))
    >>> synsets = os.path.join(my_path, "./synsets.txt")
    >>> hypernyms = os.path.join(my_path, "./hypernyms.txt")
    >>> wordnet = WordNet(synsets, hypernyms)
    >>> assert wordnet.distance("Black_Plague", "black_marlin") == 33
    >>> assert wordnet.distance("American_water_spaniel", "histology") == 27
    >>> assert wordnet.distance("Brown_Swiss", "barrel_roll") == 29

    >>> assert wordnet.sap('physical_entity', 'locus') == 'entity'
    >>> assert wordnet.sap("AIDS", "blood") == 'abstraction abstract_entity'
    >>> assert wordnet.sap("munition", "munitions_industry") == 'entity'
    >>> assert wordnet.sap("Bushido", "samurai") == 'entity'
    >>> assert wordnet.sap("individual", "edible_fruit") == 'physical_entity'
    >>> assert wordnet.sap("municipality", "district") == 'district territory territorial_dominion dominion'
    >>> assert wordnet.sap("apple", "banana") == 'edible_fruit'

    >>> relateness = Outcast(wordnet)
    >>> assert relateness.outcast(['George_Bush', 'Eric_Arthur_Blair']) == 'George_Bush'
    """

    def __init__(self) -> None:
        pass

if __name__ == '__main__':
    doctest.testmod()
