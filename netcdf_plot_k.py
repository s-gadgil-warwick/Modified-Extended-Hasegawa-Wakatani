#!/usr/bin/env python

#Script to plot amplitude of Phi at a given k_y value over time and fit to a prescribed model

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.legend_handler import HandlerLine2D
from scipy.optimize import curve_fit
from netCDF4 import Dataset

def fit_func(t, a, b, c, d, e):
	return a*np.exp(b*t)*np.sin(c*t + d)+e

fig = plt.figure(linewidth=2)
ax = fig.add_subplot(1,1,1)

col = 'white'

plt.rc('font', family='serif')
fsize = str('24')

Data = Dataset("1d_variables.nc", "r")
Phi_k_s = Data.variables["phi_k_s"]

x = np.linspace(0.0, Phi_k_s.shape[0]*0.01, Phi_k_s.shape[0])
y = Phi_k_s
y_max = 1.25*np.max(Phi_k_s)
y_min = 1.25*np.min(Phi_k_s)

init = []
init.append(1.1)
init.append(0.1)
init.append(0.6)
init.append(0.0)
init.append(0.0)

init = np.asarray(init)

params, pcov = curve_fit(fit_func, x, y, p0=init, maxfev=120000)

freq = params[2]
growth = params[1]

print("Frequency: %f\n" % freq)
print("Growth: %f\n" % growth)

f = fit_func(x, params[0], params[1], params[2], params[3], params[4])

nt = x[len(x)-1]
plt.xlim((x[0], nt))
plt.ylim((y_min, y_max))

plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)

plt.subplots_adjust(bottom=0.15, left=0.15)

s = str("Curve fit: Freq = $%.3f$, Growth rate = $%.3f$" % (freq, growth))
s1 = str(r"$\varphi(k_y=%.1f\pi,t)$" % (0.3))

plt.xlabel(r'Time$[\omega_{ci}^{-1}]$', fontsize=fsize)
plt.ylabel(s1, fontsize=fsize)

plt.plot(x, y, c = 'r', label=s1, linewidth=4)
plt.plot(x, f, c = 'g', label=s, linewidth=2)

ax = plt.gca()
ax.patch.set_facecolor(col)

legen = plt.legend(loc=2, markerscale=0.75, fontsize='16')
legen.get_frame().set_facecolor(col)
plt.show()

Data.close()
