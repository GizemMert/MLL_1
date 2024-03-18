
import torch
from Model_Vae_GE_2 import VAE_GE
from torchsummary import summary

from model4 import VariationalAutoencodermodel4
model = VariationalAutoencodermodel4(latent_dim=50)
sample_input = torch.randn(256, 14, 14)

# Printing the model summary and parameters
summary(model, input_size=sample_input.shape[0:])
sample_input = sample_input.unsqueeze(0)
z, _, mu, logvar = model(sample_input)
print("Shape of z:", z.shape)
print("Shape of mu:", mu.shape)
print("Shape of logvar:", logvar.shape)

"""
import torch
from model4 import VariationalAutoencodermodel4


# Function to print the model summary and calculate total trainable parameters
def print_model_summary_and_params(model, input_shape):
    print("Model Summary:\n")

    # Dummy input to pass through the model
    dummy_input = torch.randn(1, *input_shape)

    # Forward pass through the model to get output
    try:
        with torch.no_grad():
            output = model(dummy_input)
    except Exception as e:
        return str(e)

    # Print model
    print(model)

    # Total parameters and trainable parameters for the whole model
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Total parameters and trainable parameters for the decoder part
    decoder_total_params = sum(p.numel() for p in model.decoder.parameters())
    decoder_trainable_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)

    print("\nTotal parameters in the model: {}".format(total_params))
    print("Total trainable parameters in the model: {}".format(total_trainable_params))
    print("\nTotal parameters in the decoder: {}".format(decoder_total_params))
    print("Total trainable parameters in the decoder: {}".format(decoder_trainable_params))


# Create the model
model = VariationalAutoencodermodel4()

# Sample input shape
sample_input_shape = (256, 14, 14)

# Printing the model summary and parameters
print_model_summary_and_params(model, sample_input_shape)


import torch
from torchsummary import summary
from model4 import VariationalAutoencodermodel4

from model import VariationalAutoencodermodel

# Create the model instance
model = VariationalAutoencodermodel()

def calculate_trainable_params(model):
    # Total trainable parameters in the first decoder
    decoder_trainable_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)

    # Total trainable parameters in the image decoder
    img_decoder_trainable_params = sum(p.numel() for p in model.img_decoder.parameters() if p.requires_grad)

    # Sum of parameters in both decoders
    total_trainable_params = decoder_trainable_params + img_decoder_trainable_params

    return total_trainable_params

# Calculate and print the total trainable parameters in both decoders
total_trainable_params_decoders = calculate_trainable_params(model)
print("Total trainable parameters in both decoders:", total_trainable_params_decoders)
"""