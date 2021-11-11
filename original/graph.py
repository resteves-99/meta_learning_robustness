import matplotlib.pyplot as plt

K = [1, 2, 4, 6, 8, 10]
maml_performance = [0.963, 0.982, 0.989, 0.989, 0.991, 0.991]
maml_conf = [0.004, 0.002, 0.002, 0.002, 0.002, 0.001]
proto_performance = [0.973, 0.985, 0.992, 0.992, 0.995, 0.994]
proto_conf = [0.003, 0.002, 0.001, 0.001, 0.001, 0.001]


fix, ax = plt.subplots()
plt.xlabel("Num Shot")
plt.ylabel("Test Accuracy")
# plt.title("MAML Multi Shot Performance")
# ax.errorbar(K, maml_performance, yerr=maml_conf)
plt.title("Protonet Multi Shot Performance")
ax.errorbar(K, proto_performance, yerr=proto_conf)
plt.show()