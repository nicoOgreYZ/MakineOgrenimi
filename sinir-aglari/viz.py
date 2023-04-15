import matplotlib.pyplot as plt
import numpy as np

# Define the layers and number of neurons
layers = ['Girdi Katmanı', 'Gizli Katman 1', 'Gizli Katman 2', 'Gizli Katman 3', 'Çıktı Katmanı']
neurons = [1, 10, 10, 10, 3]

# Create a new figure and set the figure size
fig, ax = plt.subplots(figsize=(10,11))

# Set the axis limits and remove the ticks and spines
ax.set_xlim([-1.5, len(layers) - 0.5])
ax.set_ylim([-1.5, max(neurons) + 0.5])
ax.set_xticks(np.arange(len(layers)))
ax.set_xticklabels(layers)
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Loop over the layers and plot the neurons
for i, (layer, num_neurons) in enumerate(zip(layers, neurons)):
    # Compute the y-coordinates of the neurons in the layer
    y = np.arange(num_neurons) - (num_neurons - 1) / 2
    # Plot the neurons as circles with a light blue color
    ax.scatter([i] * num_neurons, y, s=150, facecolor='#C0D9E9', edgecolor='none')
    # Add text labels for the number of neurons in the layer
    ax.text(i, max(y) + 0.5, str(num_neurons), ha='center', va='bottom', fontsize=10)
    # Add text labels for the layer names
    ax.text(i, -1, layer, ha='center', va='top', fontsize=10)

# Draw the connections between the neurons
for i in range(len(layers) - 1):
    for j in range(neurons[i]):
        for k in range(neurons[i+1]):
            # Compute the coordinates of the start and end points of the connection
            x = [i, i + 1]
            y = [j - (neurons[i] - 1) / 2, k - (neurons[i+1] - 1) / 2]
            # Set the color of the connection based on the sign of the weight
            color = 'black' if np.random.rand() < 0.9 else 'red'
            # Draw the connection as a line with the appropriate color
            ax.plot(x, y, linewidth=0.5, color=color)

# Add a title and display the plot
plt.title('Sinir Ağı', fontsize=14)
plt.tight_layout()
plt.show()
