import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-15, 15, 500)
steepness_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]

plt.figure(figsize=(8, 6))

for s in steepness_values:
    y = np.tanh(s * x)
    plt.plot(x, y, label=f'tanh({s}x)')

plt.title('Tanh Functions with Different Steepness')
plt.xlabel('x')
plt.ylabel('tanh(s * x)')
plt.legend()
plt.grid(True)

plt.savefig('/home/congshan/uav/uav_roundup/navigation_strategy_2/utils')
