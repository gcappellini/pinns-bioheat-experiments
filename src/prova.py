import utils as uu

a = uu.mm_observer("cooling", "02")

a = uu.import_testdata()
e = a[:, 0:2]
theta_true = a[:, 2]
# g = uu.import_obsdata()
# uu.plot_comparison(e, )

print(a)