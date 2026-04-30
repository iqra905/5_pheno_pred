import torch
import pandas as pd
import os

# Load the saved model
result_path  = os.path.dirname(os.getcwd()) + "/GenNet_MLP/Results/Experiment_1005_Best/"
model_state_dict = torch.load(result_path + 'trained_model.pth')

# Initialize lists to store layer names and weights
layer_names = list(model_state_dict.keys())
print(layer_names)
weights = list(model_state_dict.values())
print(weights)

# Transpose the weights array to have each row correspond to a sample's weights
weights_transposed = list(zip(*weights))

# Create a DataFrame to store layer names and corresponding weights for each sample
df = pd.DataFrame(weights_transposed, columns=layer_names)

# Save the DataFrame to a CSV file
df.to_csv(result_path + 'trained_csv_weights.csv', index=False)


# # Load the saved model
# result_path  = os.path.dirname(os.getcwd()) + "/GenNet_MLP/Results/Experiment_1005_Best/"
# model_state_dict = torch.load(result_path + 'trained_model.pth')

# # Initialize lists to store layer names and weights
# layer_names = []
# weights = []

# # Extract layer names and weights from the model state dictionary
# for name, param in model_state_dict.items():
#     layer_names.append(name)
#     weights.append(param.numpy())

# # Create a DataFrame to store layer names and weights
# df = pd.DataFrame({'Layer': layer_names, 'Weights': weights})

# # Save the DataFrame to a CSV file
# df.to_csv(result_path + 'trained_csv_weights.csv', index=False)
