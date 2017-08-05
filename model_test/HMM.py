import tushare as ts
from hmmlearn import hmm
import numpy as np
from matplotlib import pyplot as plt
import pickle

data=ts.get_k_data(code="600000", start="2016-08-01").ix[:, 1:5]
price = np.array(data.ix[:,1])
data_len=len(price)
fit_len=int(data_len*0.75)
test_len=data_len-fit_len
data=data-data.shift(1)
data=data.fillna(method='bfill')
data_fit=np.array(data.iloc[0:fit_len])
print(fit_len)
print("Begin Training:")
model = hmm.GaussianHMM(n_components=4, covariance_type="diag", verbose=True, n_iter=10000).fit(data_fit)
print("Done!")

with open("hmm_model","wb") as f:
    pickle.dump(model,f)

with open("hmm_model","rb") as f:
    model2=pickle.load(f)
hiden_state = model2.predict(data_fit)
X, Z = model2.sample(data_len)
open=[price[0]]
add=price[0]
for prices in X[1:,1]:
    add+=prices
    open.append(add)
open_pred=np.array(open)


print("hiden_state: \n",hiden_state)
print("=======================")
print("Transition Matrix: \n",model2.transmat_)
print("=======================")
print("mean = \n", model2.means_)
print("=======================")
print("var = \n ",model2.covars_)
print("=======================")

#price.plot()
#plt.scatter(np.arange(len(price)),price,c=hiden_state)
plt.plot(np.arange(data_len), price, label="price")
plt.plot(np.arange(data_len), open_pred, label="predict")
plt.legend()
plt.show()





