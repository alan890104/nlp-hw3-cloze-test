import numpy as np
from sklearn.metrics import precision_recall_fscore_support

y_true = np.array(['A','B','A','A','B'])
y_pred = np.array(['A','A','A','B','A'])

a = precision_recall_fscore_support(y_true, y_pred, average='macro')
print(a)