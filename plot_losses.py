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

ax1.set_title('Training and Validation Losses', fontsize=24)
ax1.set_xlabel('Batch', fontsize=22)
ax1.set_ylabel('Loss', fontsize=22)
ax1.legend(fontsize=20)
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.grid(True)

# Plot validation accuracies
ax2.plot(batch_val, val_accuracies, label='Validation Accuracy')

ax2.set_title('Validation Accuracies', fontsize=24)
ax2.set_xlabel('Batch', fontsize=22)
ax2.set_ylabel('Accuracy', fontsize=22)
ax2.legend(fontsize=20)
ax2.tick_params(axis='both', which='major', labelsize=20)
ax2.grid(True)

plt.tight_layout()
plt.show()

# Save the plot
fig.savefig('results/losses_plot.pdf')
