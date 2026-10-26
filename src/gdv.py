# ROUTINES FOR COMPUTING THE GENERAL DISCRIMINATION VALUE

'''
The routine cmpGDV(dta,lab) expects
  a matrix of data (rows = data vectors) and
  a vector of corresponding labels.

It returns
  the mean intra-cluster distance,
  the mean inter-cluster distance, and
  the gdv-value
'''

# ***** IMPORTS ****************************************************************

from numpy import unique, concatenate, zeros
from numpy import isnan, isinf, sum, sqrt, array, triu
from scipy.spatial import distance

# ------------------------------------------------------------------------------

def makeGDVData(dta,lab):
  res = []
  labels = unique(lab)
  for L in labels:
    res.append( dta[ lab==L ] )
  return res

# ------------------------------------------------------------------------------

def zScoreSpecial(data):

  # get parameters
  NC = len(data) # nr. of clusters
  ND = data[0].shape[1] # nr. of dimensions

  # copy data --> zData
  zData = []
  for C in range(NC):
    arr = data[C].copy()
    zData.append(arr)

  # compute means and STDs for each dimension, over ALL data
  all = concatenate(zData)
  mu =  zeros(shape=ND, dtype=float)
  sig = zeros(shape=ND, dtype=float)
  for D in range(ND):
    mu[D]  = all[:,D].mean()
    sig[D] = all[:,D].std()

  # z-score the data in each cluster
  for C in range(NC):
    for D in range(ND):
      zData[C][:,D] = ( zData[C][:,D] - mu[D] ) / ( 2 * sig[D] )

  # replace nan and inf by 0
  for C in range(NC):
    nanORinf = isnan(zData[C]) | isinf(zData[C])
    zData[C][ nanORinf ] = 0.0

  return zData

# ------------------------------------------------------------------------------

def computeGDV(data):

  '''
  Returns the Generalized Discrimination Value
  as well as intraMean and interMean

  data is expected to be a list of label-sorted point 'clusters':
  data = [cluster1, cluster2, ...]

  Each cluster is a NumPy matrix,
  and the rows of this matrix
  are n-dimensional data vectors,
  each belonging to the same label.
  '''

  # get parameters
  NC = len(data) # nr. of clusters
  ND = data[0].shape[1] # nr. of dimensions

  # copy data --> zData
  zData = []
  for C in range(NC):
    arr = data[C].copy()
    zData.append(arr)

  # dimension-wise z-scoring
  zData = zScoreSpecial(zData)

  # intra-cluster distances
  dIntra = zeros(shape=NC, dtype=float)
  for C in range(NC):
    NP = zData[C].shape[0]
    dis = distance.cdist(zData[C], zData[C], 'euclidean')
    # dis is symmetric with zero diagonal
    dIntra[C] = sum(dis) / (NP*(NP-1)) # divide by nr. of non-zero el.
  #print('dIntra = ',dIntra)

  # inter-cluster distances
  dInter = zeros(shape=(NC,NC), dtype=float)
  for C1 in range(NC):
    NP1 = zData[C1].shape[0]
    for C2 in range(NC):
      NP2 = zData[C2].shape[0]
      dis = distance.cdist(zData[C1], zData[C2], 'euclidean')
      dInter[C1][C2] = sum(dis) / (NP1*NP2) # divide by nr. of non-zero el.
  #print('dInter =\n',dInter)

  # compute GDV
  pre = 1.0 / sqrt(float(ND))
  intraMean = dIntra.mean()
  interMean = sum( triu(dInter,k=1) ) / (NC*(NC-1)/2) # divide by nr. of non-zero el.
  #print('intraMean=',intraMean,'\ninterMean=',interMean)
  gdv = pre * (intraMean - interMean)

  return pre*intraMean, pre*interMean,gdv

# ------------------------------------------------------------------------------

def cmpGDV(dta,lab):
  gdvData = makeGDVData(dta,lab)
  intraMean,interMean,gdv = computeGDV(gdvData)
  return intraMean,interMean,gdv

# ------------------------------------------------------------------------------
