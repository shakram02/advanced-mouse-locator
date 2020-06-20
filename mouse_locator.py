import tkinter as tk
import numpy as np
from point_decider import PointDecider
import matplotlib.pyplot as plt

pd = PointDecider()
WIN_WIDTH = 300
WIN_HEIGHT = 300


def prepare_nn():
    train_x = []
    train_y = []

    for r in np.random.randint(-100, 100, 50):
        train_x.append(r)

        label = None
        if r < 0:
            label = 0
        else:
            label = 1

        # 0 for negative
        # 1 for positive
        train_y.append(label)

    train_x = np.array([train_x])
    train_y = np.array([train_y])

    pd.train(train_x, train_y)


def draw_pd_boundary():
    inputs = np.arange(-10, 10, 0.1)
    outputs = pd.predict([inputs])
    outputs = np.squeeze(outputs)
    plt.plot(inputs, outputs)
    plt.show()


def click_callback(event):
    x_shift = WIN_WIDTH / 2
    y_shift = WIN_HEIGHT / 2

    x_norm = event.x - x_shift
    y_norm = event.y - y_shift

    p = pd.predict([x_norm, y_norm])
    p = np.round(p)
    [x_pred, y_pred] = np.squeeze(p)

    if x_pred == 0 and y_pred == 0:
        print("Top left")
    elif x_pred == 0 and y_pred == 1:
        print("Bottom left")
    elif x_pred == 1 and y_pred == 1:
        print("Bottom right")
    else:
        print("Top right")


def display_window():
    root = tk.Tk()
    root.title("Advanced Mouse Locator.")
    frame = tk.Frame(root, width=WIN_WIDTH, height=WIN_HEIGHT)
    frame.bind("<Button-1>", click_callback)
    frame.pack()

    root.mainloop()


def main():
    prepare_nn()
    display_window()


if __name__ == "__main__":
    main()
