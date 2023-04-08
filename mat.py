import matplotlib.pyplot as plt

# Data
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
y1 = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500]
y2 = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105]
y3 = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
y4 = [500, 550, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]

# Plot the data
fig, ax = plt.subplots()
ax.scatter(x, y1, color='blue', label='y1')
ax.scatter(x, y2, color='red', label='y2')
ax.scatter(x, y4, color='green', label='y4')
for i in range(len(x)):
    if y3[i] == 1:
        ax.axvline(x=x[i], color='orange', linestyle='--')
ax.legend(loc='upper left')

# Add axis labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Scatter Plot')

# Show the plot
plt.show()
