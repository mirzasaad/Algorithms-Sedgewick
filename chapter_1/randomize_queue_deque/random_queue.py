
import random

from chapter_1.module_1_3_double_node_linked_list import DoublyLinkedList
from chapter_1.randomize_queue_deque.iterator import IteratorList

class RandomizedQueue():
  def __init__(self):
    self.queue = DoublyLinkedList()

  def isEmpty(self):
    return self.queue.length == 0

  def size(self):
    return self.queue.length 

  def enqueue(self, data):
    self.queue.append(data)
  
  def dequeue(self):
    sample = self.sample()
    if sample:
      self.queue.remove(sample)

  def __iter__(self):
    return IteratorList(self.queue.head)

  def sample(self):
    if self.size() > 0:
      rand = random.randint(0, self.size() - 1)

      for index, node in enumerate(self.queue):
        if index == rand:
          return node
  
  def __repr__(self):
    return '[{}]'.format([item for item in self.queue])