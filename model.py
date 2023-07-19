#!/usr/bin/python3.6

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np


# Define a MultiStageModel class that inherits from nn.Module
class MultiStageModel(nn.Module):

    # Initialize the model with parameters for number of stages, layers, feature maps, dimensions and classes
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()

        # Create a TowerModel instance with given parameters
        self.tower_stage = TowerModel(num_layers, num_f_maps, dim, num_classes)

        # For each other stage after the first, create a SingleStageModel instance using deepcopy and add to a ModuleList
        self.single_stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes, 3))
                                     for s in range(num_stages-1)])

    # Define the forward pass of the model, takes an input tensor and a mask tensor
    def forward(self, x, mask):

        # Run the input through the tower stage and return the middle and end outputs
        middle_out, out = self.tower_stage(x, mask)

        # Add the output to list of all outputs
        outputs = out.unsqueeze(0)

        # For each additional stage, run the output through a single stage model and add that output to the list of all outputs
        for s in self.single_stages:
            middle_out, out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        # Return the final middle output and the list of all outputs
        return middle_out, outputs



# Define a TowerModel class that inherits from nn.Module
class TowerModel(nn.Module):

    # Initialize the model with parameters for number of layers, feature maps, dimensions and classes
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(TowerModel, self).__init__()

        # Create two SingleStageModel instances for two stages of the tower, using given parameters
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes, 3)
        self.stage2 = SingleStageModel(num_layers, num_f_maps, dim, num_classes, 5)

    # Define the forward pass of the model, takes an input tensor and a mask tensor
    def forward(self, x, mask):

        # Run the input through the first stage of the tower
        out1, final_out1 = self.stage1(x, mask)

        # Run the input through the second stage of the tower
        out2, final_out2 = self.stage2(x, mask)

        # Return the sum of the outputs of both stages and the sum of the final outputs of both stages
        return out1 + out2, final_out1 + final_out2


# Define a SingleStageModel class that inherits from nn.Module
class SingleStageModel(nn.Module):

    # Initialize the model with parameters for number of layers, feature maps, dimensions, classes and kernel size
    def __init__(self, num_layers, num_f_maps, dim, num_classes, kernel_size):
        super(SingleStageModel, self).__init__()

        # Create a Conv1d layer with a 1x1 kernel to reduce dimensionality
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)

        # Use ModuleList and copy.deepcopy to create a list of DilatedResidualLayers with given parameters
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps, kernel_size))
                                     for i in range(num_layers)])

        # Create a final Conv1d layer with a 1x1 kernel for output classification
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    # Define the forward pass of the model, takes an input tensor and a mask tensor
    def forward(self, x, mask):

        # Run the input through the 1x1 convolutional layer
        out = self.conv_1x1(x)

        # Iterate over all residual layers and run input through each of them with given mask
        for layer in self.layers:
            out = layer(out, mask)

        # Multiply output of last convolutional layer by the mask and return both the output of this layer and the final output
        final_out = self.conv_out(out) * mask[:, 0:1, :]
        return out, final_out



# Define a DilatedResidualLayer class that inherits from nn.Module
class DilatedResidualLayer(nn.Module):

    # Initialize the layer with parameters for dilation, input channels, output channels and kernel size
    def __init__(self, dilation, in_channels, out_channels, kernel_size):
        super(DilatedResidualLayer, self).__init__()

        # Calculate padding required using formula for dilated convolutional layers
        padding = int(dilation + dilation * (kernel_size - 3) / 2)

        # Create a dilated Conv1d layer with given parameters
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)

        # Create a 1x1 Conv1d layer with same number of output channels as dilated layer
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)

        # Create a dropout layer
        self.dropout = nn.Dropout()

    # Define the forward pass of the layer, takes an input tensor and a mask tensor
    def forward(self, x, mask):

        # Pass the input through the dilated convolutional layer with ReLU activation
        out = F.relu(self.conv_dilated(x))

        # Use a 1x1 convolutional layer on output and apply dropout
        out = self.conv_1x1(out)
        out = self.dropout(out)

        # Add the original input to the output and multiply by the mask tensor
        return (x + out) * mask[:, 0:1, :]



# Define a Trainer class that trains and predicts with a MultiStageModel
class Trainer:

    # Initialize the trainer with necessary parameters and create a MultiStageModel object
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes):
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes)
        
        # Define loss functions
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        
        # Store number of classes
        self.num_classes = num_classes

    # Define a custom confidence loss function used in training
    def confidence_loss(self, pred, confidence_mask, device):
        batch_size = pred.size(0)
        
        # Apply softmax on prediction to get probabilities
        pred = F.log_softmax(pred, dim=1)
        
        loss = 0
        for b in range(batch_size):
            # Get number of frames in video
            num_frame = confidence_mask[b].shape[2]
            
            # Convert confidence mask to tensor and move to device
            m_mask = torch.from_numpy(confidence_mask[b]).type(torch.float).to(device)
            
            # Calculate left and right difference using log probability and mask values
            left = pred[b, :, 1:] - pred[b, :, :-1]
            left = torch.clamp(left[:, :num_frame] * m_mask[0], min=0)
            left = torch.sum(left) / torch.sum(m_mask[0])
            loss += left

            right = (pred[b, :, :-1] - pred[b, :, 1:])
            right = torch.clamp(right[:, :num_frame] * m_mask[1], min=0)
            right = torch.sum(right) / torch.sum(m_mask[1])
            loss += right

        return loss

    # Define a method to train the model
    def train(self, save_dir, batch_gen, writer, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Set start epoch for pseudo label generation
        start_epochs = 30
        print('start epoch of single supervision is:', start_epochs)
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            # Train on all batches
            while batch_gen.has_next():
                batch_input, batch_target, mask, batch_confidence = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                
                # Generate predictions and middle feature map
                middle_pred, predictions = self.model(batch_input, mask)

                # Generate pseudo labels after training specified number of epochs
                if epoch < start_epochs:
                    batch_boundary = batch_gen.get_single_random(batch_size, batch_input.size(-1))
                else:
                    batch_boundary = batch_gen.get_boundary(batch_size, middle_pred.detach())
                batch_boundary = batch_boundary.to(device)

                loss = 0
                
                # Calculate loss for each stage prediction
                for p in predictions:
                    # Compute cross-entropy loss between predicted and true labels
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_boundary.view(-1))
                    
                    # Compute MSE loss on normalized log probabilities of adjacent frames
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16) * mask[:, :, 1:])
                    
                    # Compute confidence loss using the custom confidence_loss function
                    loss += 0.075 * self.confidence_loss(p, batch_confidence, device)

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                # Calculate accuracy on current batch
                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            
            # Save model and optimizer state at each epoch, write loss and accuracy values to tensorboard
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            writer.add_scalar('trainLoss', epoch_loss / len(batch_gen.list_of_examples), epoch + 1)
            writer.add_scalar('trainAcc', float(correct)/total, epoch + 1)
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct)/total))

    # Define a method to predict labels for given videos
    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                # Load video features and convert to tensor
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                
                # Generate predictions for video and convert to action labels
                _, predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    index = list(actions_dict.values()).index(predicted[i].item())
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[index]]*sample_rate))
                
                # Write recognition results to file
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()

