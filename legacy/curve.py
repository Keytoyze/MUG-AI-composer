import os, csv, re

csv_file = open(os.path.join("out", "result.csv"))
next(csv_file)
content = csv.reader(csv_file)
color_map = ["r", "tan", "m", "b", "g", "y"]

old = []
new = []
key = []

backet_map = {}

def f(x):
    if x <= 1.3:
        return 0.474* x
    elif x <= 7:
        return 0.967 * x - 0.641
    else:
        return x - 0.872

for line in content:
    diff, current_key, new_star, old_star = line[1:]
    old_star = float(old_star)
    new_star = float(new_star)
    current_key = int(current_key)

    # current_key = 4
    old_star_round = round(old_star, 1)
    if current_key not in backet_map:
        backet_map[current_key] = {}
    if old_star_round not in backet_map[current_key]:
        backet_map[current_key][old_star_round] = []
    
    if re.search(r"7k\s-\s.*lvl", diff) is None and abs(old_star - new_star) <= 2.5:
        # old.append(old_star)
        # new.append(new_star)
        # key.append(current_key)
        backet_map[current_key][old_star_round].append(new_star)

for current_key, round_to_stars in backet_map.items():
    for old_star_round, new_stars in round_to_stars.items():
        if len(new_stars) != 0:
            new.append(sum(new_stars) / len(new_stars))
            old.append(f(old_star_round))
            key.append(current_key)
max_star = 15
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
plt.xlabel("xxy's algo")
# plt.xlim(0, max_star)
# plt.ylim(0, max_star)
plt.ylabel("ppy's algo")
patches = [mpatches.Patch(color=color_map[i-4], label="{:s}K".format(str(i))) for i in range(4, 10)]
ax=plt.gca()
ax.legend(handles=patches)
diag_line, = plt.plot([0, max_star], [0, max_star], ls="--", c=".3")

from sklearn import linear_model        #表示，可以调用sklearn中的linear_model模块进行线性回归。
import numpy as np

plt.scatter(old, new, s=3, color=[color_map[x - 4] for x in key])
with open("out.csv", "w") as f:
    f.write("x,y\n")
    for i in range(len(old)):
        f.write("%lf,%lf\n" % (old[i], new[i]))

plt.show()