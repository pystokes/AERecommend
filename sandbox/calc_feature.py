import datetime
import random
import numpy as np

# Config of prepared data
n_class = 165
dim_feature = 512 * 13 * 13 # Based on SqueezeNet

# Config of each user data
n_like = 10

# Set sample data
class_centers = np.random.randn(n_class * dim_feature).reshape(n_class, dim_feature)
like_list = np.random.randn(n_like * dim_feature).reshape(n_like, dim_feature)

ids_in_class = {}
for i_class in range(len(class_centers)):
    ids_in_class[i_class] = [f'{i_class}-{id}' for id in range(100)]

"""
BEGINNING
"""
bgn_time = datetime.datetime.now()
# Center of user
user_center = like_list.sum(axis=0) / len(like_list)

# Calculate distances between user center and each class center
distances = []
for i_class in range(len(class_centers)):
    diff = user_center - class_centers[i_class]
    squared_diff = np.square(diff)
    discrete_dist = np.sqrt(squared_diff.sum())
    distances.append(discrete_dist)

# Random selection from nearest class
arg_min = np.argmin(distances)
recommend = random.sample(ids_in_class[arg_min], 10)

print(datetime.datetime.now() - bgn_time)
"""
END
"""

print(recommend)
