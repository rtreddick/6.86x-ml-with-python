import json

import numpy as np
import pandas as pd
from pyprojroot import here

import p04_netflix.kmeans as kmeans
import p04_netflix.common as common
import p04_netflix.naive_em as naive_em
import p04_netflix.em as em


# X = np.loadtxt("toy_data.txt")
X = np.loadtxt(here('./p04_netflix/toy_data.txt'))
Ks = [1,2,3,4]
seeds = [0,1,2,3,4]


def run_kmeans(X, Ks, seeds, results_data=None):
    if results_data == None:
        results_data = []
    for K in Ks:
        for seed in seeds:
            init_mixture, init_post = common.init(X, K, seed)
            mixture, post, cost = kmeans.run(X, init_mixture, init_post)
            results_data.append((K, seed, mixture, post, cost))

    return results_data


results_data = run_kmeans(X, Ks, seeds)
results_df = pd.DataFrame(results_data, columns=['K', 'seed', 'mixture', 'post', 'cost'])
print(results_df.groupby(['K'])['cost'].apply(np.min))

# df = results_df
# idx = df.groupby(['K'])['cost'].transform(min) == df['cost']
# min = df[['K', 'seed', 'cost']][idx]
# print(min)



