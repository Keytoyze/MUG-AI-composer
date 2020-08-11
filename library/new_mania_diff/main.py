import algorithm
import sys
import os

length_of_arguments = len(sys.argv) # argv:运行程序时在输入窗口打上的变量
file_path = ""
if length_of_arguments != 1:
    file_path = sys.argv[1]
    calculator = algorithm.star_calculator()  # 对应__init__
    calculator.calculate_SR(file_path)
    # Asp = calculator.calculate_asperity()
    # print(Asp)
    X_collection = calculator.X_collection()
    D = calculator.time_to_note_per_column(calculator.note_starts, calculator.note_ends, X_collection[0], X_collection[1])
    cal_J = calculator.J(X_collection[0], X_collection[1], calculator.note_starts, calculator.note_ends, D, calculator.asperity)
    cal_X = calculator.X(cal_J)

    Y = calculator.calculate_Y(calculator.note_starts, calculator.note_ends, calculator.asperity)

    Z_collection = calculator.Z_collection()
    E = calculator.time_to_note_per_pair(calculator.note_starts, calculator.note_ends, Z_collection[0], Z_collection[1])
    cal_O = calculator.O(Z_collection[0], Z_collection[1], calculator.note_starts, calculator.note_ends, E, calculator.asperity)
    cal_Z = calculator.Z(cal_O)

    A = (0.35*cal_X**1.5 + 0.85*Y**1.5)**(0.667)
    B = 20*cal_Z**0.5/(0.4*cal_X**1.5 + 0.7*Y**1.5)**(0.667)
    SR = 0.16*((0.6*min(B**0.9, B**1.8)+1)*A*(0.88+0.03*calculator.column_count))**1.06
    # SR = round(100*SR)/100
    if len(sys.argv) == 3:
        with open(sys.argv[2], "w") as f:
            f.write(str(SR))
    else:
        print("Rating: " + str(SR) + "     ")
    print()