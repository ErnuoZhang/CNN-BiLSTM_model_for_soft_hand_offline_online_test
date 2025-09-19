from matplotlib.widgets import Button, CheckButtons, Slider
import matplotlib
from matplotlib.colors import same_color
import numpy as np
matplotlib.use('Qt5Agg')
# import ipywidgets as widgets

def Custom_Buttom(axes,text, func):
    bt = Button(axes, text, hovercolor="#111113")
    bt.drawon = False
    bt.on_clicked(func)
    bt.color = "#5c6370"
    bt.label.set_color("#abb2bf")
    return bt

def Custom_CheckButtom(axes, text):
    bt = CheckButtons(axes, text)
    return bt
    
def Custom_Slider(axes, text, func):
    bt = Slider(axes, "", valmin=1, valmax=17, valinit=1, valstep=np.arange(1, 17, 1))
    bt.label = bt.ax.text(0.5, .85, text, transform=bt.ax.transAxes,
                        verticalalignment='bottom',
                        horizontalalignment='center')
    bt.valtext.set(alpha=0)                 
    bt.valtext = bt.ax.text(0.5, .15, bt._format(0),
                transform=bt.ax.transAxes,
                verticalalignment='top',
                horizontalalignment='center')
    bt.on_changed(func)

    return bt

# class Custom_BWidget(Button):
#     def __init__(self, ax, label, image=None, color='0.85', hovercolor='0.95'):
#         super().__init__(ax, label, image, color, hovercolor)
    
#     def _motion(self, event):
#         if self.ignore(event):
#             return
#         c = self.hovercolor if event.inaxes == self.ax else self.color
#         if not same_color(c, self.ax.get_facecolor()):
#             self.label.set_color(c)
#             if self.drawon:
#                 self.ax.figure.canvas.draw()