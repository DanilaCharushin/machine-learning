import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("ggplot")

data = pd.read_csv("data_25_11.csv", sep=",", header="infer")

print(data.head())
print(data.describe(include="all"))

# data["duration"].hist()
# data["duration"].plot.hist()

fig, ax1 = plt.subplots(figsize=(9, 5))

ax1.hist(data["duration"], bins=300, histtype="barstacked", linewidth=2, alpha=0.7, log=True)
# plt.plot(data["duration"].hist(bins=400, normed=1))
# ax1.set_xlabel("Interval")
# ax1.set_ylabel("d1 freq")
#
plt.show()
asdfsadf asdfasdfasd ddf
sdf

cvxcvxv
