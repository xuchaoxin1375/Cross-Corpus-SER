from sklearn.model_selection import train_test_split

import numpy as np

M=np.arange(100).reshape(10,10)
print(M,"@{M}")
X_train,X_test=train_test_split(M,train_size=0.7,shuffle=False)
print(X_train,"@{X_train}")
print(X_test,"@{X_test}")