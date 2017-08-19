# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 10:06:09 2017
This is matplotlib examples
@author: xiabingyu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#==============================================================================
# #DATA
# x = np.linspace(-3, 3, 60)
# y = 2*x +1 
# y1 = x**2
# 
# #Declare a figure
# plt.figure(num=3, figsize=(4, 5))
# plt.plot(x, y, "b--", lw=3)
# plt.scatter(x, y1, label="y1",c="r", s=np.random.randint(10,100,30), alpha=0.6)
# plt.xlim(0, 3)
# plt.ylim(-2, 8)
# plt.xticks(np.linspace(-3, 3, 10))
# 
# #AX
# Base function
# ax = plt.gca()   # get current ax
# ax.set_title("matplot test")
# ax.set_xlabel("x")
# ax.set_xlim(0, 3)
# ax.set_ylim(-2, 8)
# ax.set_xticks([0, 1, 2, 3])#,("poor", "not\ bad", r"good", "pretty"))
# ax.spines["right"].set_color('none')
# ax.spines["top"].set_color('none')
# ax.spines["left"].set_position(('data',0))
# ax.spines['bottom'].set_position(('data', 0))
# ax.legend()
# 
# #Note
# x0 = 2
# y0 = 2*x0 + 1
# ax.scatter(x0, y0, c="b", s=100)
# ax.annotate(r'$2x+1=%s$' %y0, xy=(x0, y0), xycoords='data', xytext=(+30, -30),
#              textcoords='offset points', fontsize=16,
#              arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
# 
# ax.text(0.5, 6, r"$\mu \ \sigma_i\ \alpha_j$", 
#         fontdict={'size':16, 'color':'y'})
# plt.show()
#==============================================================================

#==============================================================================
# #Bar 
# plt.figure(num=2, figsize=(6, 4))
# ax = plt.gca()
# n = 12 
# X = np.arange(n)
# Y1 = np.random.uniform(0, 8, n)
# Y2 = np.random.uniform(-8, 0, n)
# 
# ax.bar(X, Y1, facecolor="b")
# ax.bar(X, Y2, facecolor="r")
# ax.set_xticks([])
# ax.set_ylim([-10, 10])
# #ax.spines['left'].set_color('none')
# ax.spines['right'].set_color('none')
# ax.spines['bottom'].set_color('none')
# ax.spines['top'].set_position(('data', 0.))
# for x,y in zip(X, Y1):
#     ax.text(x, y+0.5, '%.2f' %y, ha='center', va="bottom")
# 
# for x,y in zip(X, Y2):
#     ax.text(x, y-0.5, '%.2f' %y, ha='center', va='top')
# 
# plt.show()
#==============================================================================

#==============================================================================
# #Image
# plt.figure(num=1, figsize=(3,3))
# a = np.random.uniform(0., 1., (3,3))
# ax = plt.gca()
# ax.imshow(a, cmap ="bone", interpolation="bilinear", origin="upper")
# plt.show()
#==============================================================================


#==============================================================================
#3D plot
from mpl_toolkits.mplot3d import Axes3D
fig =plt.figure(num=0)
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X,Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
R = np.sin(R)

ax.plot_surface(X, Y, R, rstride=1, cstride=1, cmap='rainbow')
#ax.contourf(X, Y, R, zdir="z", offset=-2, cmap="rainbow")  #3D
plt.show()
#==============================================================================


#==============================================================================
# #Figure in Figure
# fig = plt.figure(num=6, figsize=(8,6))
# x = np.arange(6)
# y = np.random.random(6).cumsum()
# ax1 = fig.add_axes([0.1, 0.1, 0.9, 0.9])
# ax1.plot(x, y, "r--", lw=3.)
# ax2 = fig.add_axes([0.2, 0.6, 0.25, 0.25])
# ax2.scatter(x, y, s=20, alpha=0.8)
# ax3 = fig.add_axes([0.7, 0.2, 0.25, 0.25])
# ax3.bar(x, y)
# ax3.set_title(r"$bar\ example$")
# plt.show()
#==============================================================================


#==============================================================================
# #Secondary_axis
# x = np.arange(0, 10, 0.01)
# y1 = 0.05 * x**2
# y2 = -1 * y1
# 
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# 
# ax1.plot(x, y1, "g-")
# ax2.plot(x, y2, "r--")
# 
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y1', color='g')
# ax2.set_ylabel('Y2', color='r')
# 
# plt.show()
#==============================================================================
 
# #Animation
# from matplotlib import animation
# fig, ax = plt.subplots()
# x = np.arange(0, 2*np.pi, 0.01)
# line, = ax.plot(x, np.sin(x))

# def animation_local(i):
#     line.set_ydata(np.sin(x+i/100.))
#     return line,

# def init():
#     line.set_ydata(np.sin(x))
#     return line,

# ani = animation.FuncAnimation(fig=fig, func=animation_local, frames=100, init_func=init, interval=20, blit=False)
# plt.show()


#==============================================================================
# #DataFrame
# #df = pd.DataFrame(np.random.random((10, 4)), columns=list("ABCD")).cumsum()
# #Some methods
# 
# #df.plot(figsize=(5,8))
# #df.plot.scatter(x="A", y="B", c="r")
# #df.plot.area()
# #plt.show()
# 
#==============================================================================


