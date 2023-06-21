import numpy as np
import torch
import pandas as pd, numpy as np
from sklearn.cluster import KMeans

#(tensor([0, 0, 0, 4, 4, 4]), tensor([4, 4, 4, 0, 0, 0]), tensor([1, 2, 3, 1, 2, 3]))
a = np.array([[[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 1, 1, 1, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],

        [[0, 1, 1, 1, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]]
             )

x = torch.randn(3, 3)
print(x)
y = torch.ones(3, 3)
print(torch.where(x))
print(torch.where(y))

b = torch.where(torch.tensor(a))
print("b,", b)

print(torch.nonzero(torch.tensor([[0.6, 0.1, 0.0, 0.0, 0],
                                [0.4, 0.0, 0.0, 0.0, 0],
                                [0.0, 0.0, 1.2, 0.0, 0],
                                [0.0, 0.0, 0.0,-0.4, 0]]
                                 ), as_tuple=True))


import json

df = pd.DataFrame({
    "Sample Name": ["Sample "+str(i) for i in range(6)],
    "Feature1": [6, 5, 1, 2, 3, 4],
    "Feature2": [5, 6, 2, 1, 4, 3],
})

df.loc[[4,5], ["Feature1"]] += df.loc[[4,5], ["Feature1"]] + 30 # Completely arbitrary
print(df)

kms = KMeans(n_clusters=3, random_state=1).fit(df[['Feature1', 'Feature2']])
df['Cluster'] = kms.labels_

features = df.columns.tolist()[1:-1]
print(f"Features: \n{features}")

centroids = kms.cluster_centers_
print(f"Centroids \n{centroids}")

sorted_centroid_features_idx = centroids.argsort(axis=1)[:,::-1]
print(f"Sorted Feature/Dimension Indexes for Each Centroid in Descending Order: \n{sorted_centroid_features_idx}")

print()

sorted_centroid_features_values = np.take_along_axis(centroids, sorted_centroid_features_idx, axis=1)
print(f"Sorted Feature/Dimension Values for Each Centroid in Descending Order: \n{sorted_centroid_features_values}")


first_features_in_centroid_1 = centroids[0][sorted_centroid_features_idx[0]]
print(list(
        zip(
            [features[feature] for feature in sorted_centroid_features_idx[0]],
            first_features_in_centroid_1
        )
    ))