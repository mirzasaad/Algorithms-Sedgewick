class IteratorList():
  def __init__(self, head):
    self.current = head
  
  def __iter__(self):
    return self

  def __next__(self):
    if not self.current:
      raise StopIteration
    else:
      item = self.current
      self.current = self.current.next
      return item