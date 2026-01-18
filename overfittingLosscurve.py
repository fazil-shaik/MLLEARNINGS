import matplotlib.pyplot as plt

epochs = range(1, 21)

train_loss = [0.9, 0.7, 0.5, 0.4, 0.35, 0.3, 0.25, 0.22, 0.2, 0.18, 
              0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07]

val_loss = [0.95, 0.8, 0.65, 0.55, 0.5, 0.48, 0.47, 0.48, 0.52, 0.58, 
            0.65, 0.72, 0.8, 0.88, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, 'bo-', label='Training Loss')   # Blue solid line
plt.plot(epochs, val_loss, 'r*-', label='Validation Loss')  # Red dashed/starred line

# Annotate the point of divergence
plt.annotate('Overfitting starts here', xy=(7, 0.47), xytext=(10, 0.8),
             arrowprops=dict(facecolor='purple', shrink=0.05))

# Formatting the graph
plt.title('Overfitting Detection: Training vs Validation Loss', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()
