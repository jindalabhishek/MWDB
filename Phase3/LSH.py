import os
import json
import numpy as np


class LSHash(object):

  def __init__(self, input_dim, k_bit_hash, num_hashtables):
    self.k_bit_hash = k_bit_hash
    self.input_dim = input_dim
    self.storage_config = {'dict': None}
    self.num_hashtables = num_hashtables
    self.uniform_planes = [ np.random.randn(k_bit_hash, input_dim) for _ in range(num_hashtables)]  #initialise the family of hash tables
    self.hash_tables = [dict() for i in range(self.num_hashtables)]


  def _hash_projections(self, planes, input_point):
    input_point = np.array(input_point)  
    projections = np.dot(planes, input_point)
    return "".join(['1' if i > 0 else '0' for i in projections])
  

  def index(self, input_point, extra_data=None):

        if isinstance(input_point, np.ndarray):
            input_point = input_point.tolist()

        if extra_data:
            value = (tuple(input_point), extra_data)
        else:
            value = tuple(input_point)

        for i, table in enumerate(self.hash_tables):
            table.setdefault(self._hash_projections(self.uniform_planes[i], input_point),[]).append(value)


  def query(self, query_point, num_results=None):

    candidates = set()   
    for i, table in enumerate(self.hash_tables):
      binary_hash = self._hash_projections(self.uniform_planes[i], query_point)
      
      candidates.update(table.get(binary_hash,[]))
    with open('hash_tables.json','w') as file:
      for i in range(len(self.hash_tables)):
        json.dump(self.hash_tables[i], file)
    
    candidates = [(ix, np.linalg.norm(query_point-np.asarray(ix))) for ix in candidates]
    candidates.sort(key=lambda x: x[1])

    return candidates[:num_results] if num_results else candidates




pp = LSHash(3,5,2)
inps = np.random.randn(130, 3)
for inp in inps:
  pp.index(inp)

planes = pp.query([0.3,0.6,0.9])

for plane in planes:
  print(plane)

print(len(planes))