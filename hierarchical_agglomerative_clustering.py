import pandas as pd


def longest_subsequence(string1, string2):
  ''' distance metric for measuring the distance between two strings,
      longest common subsequence between the two strings, case insensitive
      params: string1: str
      params: string2: str
      return: the length of longest common subsequence between string1 and string2
  '''
  string1 = string1.lower()
  string2 = string2.lower()
  m, n = len(string1), len(string2)
  result = [[0] * (n + 1) for _ in range(m + 1)]
  # dynamic programming algorithm for finding the longest common subsequence
  # between two strings
  for i in range(1, m + 1):
    for j in range(1, n + 1):
      if string1[i - 1] == string2[j - 1]:
        result[i][j] = result[i - 1][j - 1] + 1
      else:
        result[i][j] = max(result[i][j - 1], result[i - 1][j])
  return result[-1][-1]


def hac(df, threshold=4):
  ''' hierarchical agglomerative clustering algorithm while applying the
      longest common subsequence metric for string distance.
      params: df: pandas.dataframe, the dataframe for clustering
      return: None, modify the df in-place
  '''
  clusters = []
  retailers = dict()
  n_cluster = 0
  for _, row in df.iterrows():
    retailer = row['retailer_nm_modified']
    for other in retailers:
      if longest_subsequence(retailer, other) >= threshold:
        # greedily assign the cluster when longest common subsequence
        # is greater than some threshold
        clusters.append(retailers[other])
        break
    else:
      # if no cluster is found, assign a new cluster (use integer encoding)
      n_cluster += 1
      retailers[retailer] = n_cluster
      clusters.append(retailers[retailer])

  # add the assigned cluster_id to the dataframe
  df.insert(0, 'cluster_id', clusters)


def evaluate(df):
  ''' evalute the clustering of the dataframe by calculating the precision
      and recall for each cluster_id
      params: df: pandas.dataframe, the dataframe for evalution
      return: None, modify the dataframe in-place
  '''
  precision, recall = {}, {}

  def assign_precision(cluster_id):
    return precision[int(cluster_id)]

  def assign_recall(cluster_id):
    return recall[int(cluster_id)]

  for cluster in df['cluster_id'].unique():
    # group the true clusters within the predicted cluster
    cluster_group = df[(df['cluster_id'] == cluster)].groupby(['retailer_id'])
    # the closest true label for the predicted cluster
    retailer = cluster_group.size().idxmax()
    # the number of true-positives in the predicted cluster
    tp = max(cluster_group.size())
    # recall = tp / (tp + fn)
    recall[int(cluster)] = tp / len(df[(df['retailer_id'] == retailer)])
    # precision = tp / (tp + fp)
    precision[int(cluster)] = tp / len(df[(df['cluster_id'] == cluster)])

  # add the cluster_id-wise precision and recall to the dataframe
  df.insert(1, 'precision_for_cluster_id', df['cluster_id'].apply(assign_precision))
  df.insert(2, 'recall_for_cluster_id', df['cluster_id'].apply(assign_recall))

if __name__ == '__main__':
  # parse the dataframe
  df = pd.read_csv('retailers.csv')

  # apply the clustering algorithm on the dataframe
  hac(df)

  # postprocess the dataframe by adding the evaluation results
  evaluate(df)

  print(df.columns)
