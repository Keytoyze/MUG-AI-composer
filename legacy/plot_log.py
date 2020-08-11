import matplotlib.pyplot as plt
import numpy as np
import re

if __name__ == "__main__":
    acc = []
    top5 = []
    over = []
    lost = []
    cce_loss = []
    str_loss = []
    var = []
    length = 0
    with open("../logs/log_lr_test.log") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue

            line = line.replace(":", " =")
            count = length % 1000
            length += 1

            # x.append(float(re.findall("loss\s=\s\((.*), (.*), (.*)\)", line)[0][0]))
            # y.append(float(re.findall("loss\s=\s\((.*), (.*), (.*)\)", line)[0][2]))
            acc.append(float(re.findall("acc\s=\s(.*?),", line)[0]))
            top5.append(float(re.findall("top5_acc\s=\s(.*?),", line)[0]))
            over.append(float(re.findall("overmap_acc\s=\s(.*?),", line)[0]))

            lost.append(float(re.findall("lostnote_acc\s=\s(.*?),", line)[0]))
            cce_loss.append(float(re.findall("cce_loss\s=\s(.*?),", line)[0]))
            str_loss.append(float(re.findall("strength_loss\s=\s(.*?),", line)[0]))
            var.append(float(re.findall("var\s=\s(.*?),", line)[0]))



    s = 100
    print(top5)


    def format(in_array):
        out_array = []
        pre_x = None
        for i, xi in enumerate(in_array):
            count = i % 1000
            if count != 0:
                x = xi * (count + 1) - pre_x * count
            else:
                x = xi
            pre_x = xi
            # x = xi
            if i % s == 0:
                out_array.append(x)
            else:
                out_array[-1] = (out_array[-1] * (i % s) + x) / (i % s + 1)
        # maxx = max(out_array)
        # minn = min(out_array)
        # out_array = [(x - minn) / (maxx - minn) for x in out_array]
        return out_array


    cce_loss = format(cce_loss)
    str_loss = format(str_loss)
    acc = format(acc)
    top5 = format(top5)
    over = format(over)
    lost = format(lost)
    var = format(var)
    length = len(acc)

    # plt.plot(list(range(length)), cce_loss, color='red', linewidth=2.0, label="cce")
    # plt.plot(list(range(length)), str_loss, color='blue', linewidth=2.0, label="strength")
    # plt.plot(list(range(length)), acc, color='orange', linewidth=2.0, label="acc")
    # plt.plot(list(range(length)), top5, color='g', linewidth=2.0, label="top5")
    # plt.plot(list(range(length)), over, color='y', linewidth=2.0, label="over")
    # plt.plot(list(range(length)), lost, color='black', linewidth=2.0, label="lost")
    plt.plot(list(range(length)), var, color='purple', linewidth=2.0, label="var")
    plt.legend()
    plt.show()
