import numpy as np
import math
import os
from Util.Utils import get_similarity_matrix
from Util.graph_util import convert_similarity_matrix_to_graph

def update_ASCOS_similarity(processed, adjacency, c, i, j):
  #print("   Updating " + str(i) + ", " + str(j))
  if(i == j): return 1

  weight_i = np.sum(adjacency[i])
  s = 0
  for k in range(len(adjacency)):
    weight_ik = adjacency[i][k]
    if(weight_ik == 0 or k == i or k in processed): 
      s += 0
    else:
      processed.append(k)
      s += update_ASCOS_similarity(processed, adjacency, c, k, j)*c*(weight_ik/weight_i)*(1-math.exp(-weight_ik))
  
  return s

def convergence_test(S, S_new):
  if np.isclose(S,Snew):
    return True
  else:
    return False
   # implemented but not tested

def ASCOS_similarity(adjacency, c, iterations = 10):
  # Initialize S (nxn) with diagnal = 1 
  n_objects = len(adjacency)
  S = np.zeros((n_objects,n_objects))

  iter_i = 1
  while True:
    S_new = S
    for i in range(n_objects):
      for j in range(n_objects):
        #print("Updating " + str(i) + ", " + str(j))
        S_new[i][j] = update_ASCOS_similarity([i], adjacency, c, i, j)
    
    if(iter_i >= iterations):
      return S_new
    S = S_new
    iter_i += 1
    #print("*** Iteration " + str(iter_i) + " ***")

def ASCOS_PageRank(similarity_graph):
  n_subjects = len(similarity_graph)
  PR = np.zeros(n_subjects)
  for i in range(n_subjects):
    for j in range(n_subjects):
      PR[i] += similarity_graph[i][j]
  return PR/n_subjects

sim_matrix_dir = input("Enter similarity matrix name: ")
n = int(input("Enter value n: "))
while(n > 40):
  print("n must be <= 40")
  n = int(input("Enter value n: "))
m = int(input("Enter value m: "))
while(m > 40):
  print("m must be <= 40")
  m = int(input("Enter value m: "))

similarity_matrix = np.array(get_similarity_matrix(sim_matrix_dir))
adjacency_matrix = convert_similarity_matrix_to_graph(similarity_matrix, n)
print("\nSimilarity Graph:")
print(adjacency_matrix)

similarity_graph = ASCOS_similarity(adjacency_matrix, 1)
PageRank = ASCOS_PageRank(similarity_graph)
print("\nPageRank:")
print(PageRank)

sorted_PR = sorted(PageRank, reverse=True)
PR_list = PageRank.tolist()
sorted_PR_subject = []
for i in range(len(sorted_PR)):
  index = PR_list.index(sorted_PR[i])
  sorted_PR_subject.append(index)
  PR_list[index] = 0

sorted_PR_subject = np.array(sorted_PR_subject)
print("Most significant subjects (1-40):")
for i in range(m):
  print(sorted_PR_subject[i]+1)


