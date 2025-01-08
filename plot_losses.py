import json
import matplotlib.pyplot as plt

#######################
# Load the losses
#######################

with open('results/losses.json', 'r') as f:
    losses = json.load(f)

training_batch_losses = losses['training_batch_losses']
validation_losses = losses['validation_losses']
val_accuracies = losses['validation_accuracies']
batch_train = losses['batch_train']
batch_val = losses['batch_val']

#######################
# Plot training and validation losses
#######################

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot training and validation losses
ax1.plot(batch_train, training_batch_losses, label='Training Batch Loss')
ax1.plot(batch_val, validation_losses, label='Validation Loss')
ax1.set_title('Training and Validation Losses')
ax1.set_xlabel('Batch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# Plot validation accuracies
ax2.plot(batch_val, val_accuracies, label='Validation Accuracy')
ax2.set_title('Validation Accuracies')
ax2.set_xlabel('Batch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Save the plot
fig.savefig('results/losses_plot.pdf')
