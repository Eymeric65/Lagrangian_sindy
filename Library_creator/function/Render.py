import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

def Animate_Single_pendulum(L, q_v, t_v):

    x = L* np.sin(q_v[:, 0])
    y = -L * np.cos(q_v[:, 0])


    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, L))
    ax.set_aspect('equal')
    ax.grid()

    trace, = ax.plot([], [], '.-', lw=1, ms=2)
    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def animate(i):
        thisx = [0, x[i]]
        thisy = [0, y[i]]

        history_x = x[:i]
        history_y = y[:i]

        line.set_data(thisx, thisy)
        trace.set_data(history_x, history_y)
        time_text.set_text(time_template % (t_v[i]))
        return line, trace, time_text

    ani = animation.FuncAnimation(
        fig, animate, len(q_v), interval=40, blit=True)
    plt.show()

def Animate_double_pendulum(L1, L2, q_v, t_v):
    Lt = L1 + L2

    x1 = L1 * np.sin(q_v[:, 0])
    y1 = -L1 * np.cos(q_v[:, 0])

    x2 = L2 * np.sin(q_v[:, 2]) + x1
    y2 = -L2 * np.cos(q_v[:, 2]) + y1

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(autoscale_on=False, xlim=(-Lt, Lt), ylim=(-Lt, Lt))
    ax.set_aspect('equal')
    ax.grid()

    trace, = ax.plot([], [], '.-', lw=1, ms=2)
    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]

        history_x = x2[:i]
        history_y = y2[:i]

        line.set_data(thisx, thisy)
        trace.set_data(history_x, history_y)
        time_text.set_text(time_template % (t_v[i]))
        return line, trace, time_text

    ani = animation.FuncAnimation(
        fig, animate, len(q_v), interval=40, blit=True)
    plt.show()