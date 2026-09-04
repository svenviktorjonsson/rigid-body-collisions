import numpy as np
import matplotlib.pyplot as plt
import time

# Disable specific shortcuts
plt.rcParams['keymap.save'] = 'ctrl+s'  # Disable 's' for saving
plt.rcParams['keymap.fullscreen'] = 'ctrl+f'  # Disable 'f' for fullscreen
plt.rcParams['keymap.quit'] = ''  # Disable 'f' for fullscreen

def black_ax(fig):
    fig.patch.set_facecolor('black')
    ax = fig.add_axes([0,0,1,1])
    ax.set_facecolor('black')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.axis("off")
    ax.set_aspect("equal")
    return ax


class Simulation:
    def __init__(self,cdim):
        self.data = np.zeros((100,cdim))

class Interactor:
    def __init__(self,width,height,dpi):
        self.dpi=dpi
        self.fig = plt.figure(figsize=(width/dpi,height/dpi),dpi=dpi)
        self.ax = black_ax(self.fig)
        self.model = None
        self.time_zero = time.perf_counter()

        self.buttons_pressed = {}
        self.keys_pressed = {}

        self.connect_events()
        
    def connect_events(self):
        self.press_cid = self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.move_cid = self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.release_cid = self.fig.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.key_press_cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.key_release_cid = self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)

    def time(self):
        return time.perf_counter()-self.time_zero

    def on_button_press(self, event):
        self.buttons_pressed[event.button] = self.time(), event.xdata, event.ydata

    def on_mouse_move(self, event):
        self.mouse_event = self.time(), event.xdata, event.ydata
        if self.buttons_pressed:
            self.on_mouse_drag()
    
    def on_mouse_drag(self):
        pass

    def on_button_release(self, event):
        self.buttons_pressed.pop(event.button,0)

    def on_key_press(self, event):
        self.keys_pressed[event.key] = (self.time(), *self.mouse_event[1:])

    def on_key_release(self, event):
        self.keys_pressed.pop(event.key,"")

    def setModel(self, model):
        self.model=model

    def start(self):
        plt.show()



def main():
    simulation = Simulation(10)
    
    interactor = Interactor(700,400,200)

    interactor.setModel(simulation)
    
    interactor.start()



if __name__=="__main__":
    main()

