import matplotlib.pyplot as plt
import pickle

with open("layer_input.pkl","rb") as f:
	layer_input = pickle.load(f)
with open("pre_activation.pkl","rb") as f:
	pre_activation = pickle.load(f)

#Plot histogram
def plot_histogram(l_in, pre_ac):
	for i, (ax_pre, ax_in) in enumerate(zip(axs[0, :], axs[1, :])):
		[a.clear() for a in [ax_pre, ax_in]]
		ax_pre.set_title('L'+str(i))
		ax_pre.hist(pre_ac[i].ravel(), bins=50,)
		ax_in.hist(l_in[i].ravel(), bins=50,)


f, axs = plt.subplots(2, 5, figsize=(10, 5))
plot_histogram(layer_input, pre_activation)
plt.show()