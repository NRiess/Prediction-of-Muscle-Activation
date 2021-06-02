import numpy as np
from operator import itemgetter, attrgetter, methodcaller

eval_data = [[0, 1, 1, 3],[0, 1, 3, 0],[0, 1, 2, 3],[0, 3, 1, 0],[2, 1, 0, 0],[2, 2, 0, 1],[2, 1, 1, 3],\
                      [0, 2, 3, 0],[0, 1, 2, 1],[2, 3, 1, 0],[3, 1, 2, 0],[2, 1, 0, 2]]
sort_priority = [0, 2, 1, 3]
descending = False



number_of_zeros = 0
for i in range(len(sort_priority)):
    if sort_priority[i] == 0:
        number_of_zeros += 1
number_of_nonzero_entries = len(sort_priority)-number_of_zeros

idx_feature = np.argsort(np.absolute(sort_priority))
idx_feature = idx_feature[number_of_zeros:len(idx_feature)]
print(idx_feature)
print(descending)




# def number_of_features(number_of_nonzero_entries,eval_data,idx_feature,descending):
#     switcher = {
#         0: eval_data,
#         1: sorted(eval_data, key=itemgetter(2), reverse=descending),
#         2: sorted(eval_data, key=itemgetter(idx_feature[0],idx_feature[1]), reverse=descending),
#         3: sorted(eval_data, key=itemgetter(idx_feature[0],idx_feature[1],idx_feature[2]), reverse=descending),
#         4: sorted(eval_data, key=itemgetter(idx_feature[0],idx_feature[1],idx_feature[2],idx_feature[3]), reverse=descending)
#     }
#     return switcher.get(number_of_nonzero_entries,eval_data,idx_feature,descending, 2)

# eval_data = number_of_features(number_of_nonzero_entries,eval_data,idx_feature,descending)

descending = False
if number_of_nonzero_entries==0:
        result = eval_data
if number_of_nonzero_entries==1:
    result = sorted(eval_data, key=itemgetter(idx_feature[0]),
                    reverse=descending)
if number_of_nonzero_entries==2:
    result = sorted(eval_data, key=itemgetter(idx_feature[0], idx_feature[1]),
                    reverse=descending)
if number_of_nonzero_entries==3:
    result = sorted(eval_data, key=itemgetter(idx_feature[0], idx_feature[1], idx_feature[2]), reverse=descending)
    num0 = idx_feature[0]
    num1 = idx_feature[1]
    num2 = idx_feature[2]
if number_of_nonzero_entries==4:
    result = sorted(eval_data, key=itemgetter(idx_feature[0], idx_feature[1], idx_feature[2], idx_feature[3]),
                    reverse=descending)
eval_data = result
print(eval_data)
