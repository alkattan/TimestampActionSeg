#!/usr/bin/python3.6

import torch
import numpy as np
import random


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate):
        # Initializing the batch generator with given parameters
        self.list_of_examples = list()  # List to store all examples
        self.index = 0  # Current index in the list of examples
        self.num_classes = num_classes  # Total number of classes in the dataset
        self.actions_dict = actions_dict  # Dictionary mapping action names to class indices
        self.gt_path = gt_path  # Path to the ground truth labels
        self.features_path = features_path  # Path to the input features
        self.sample_rate = sample_rate  # Rate at which frames are sampled from the videos
        self.gt = {}  # Dictionary to store the ground truth labels for each video
        self.confidence_mask = {}  # Confidence mask for each video

        # Extract the dataset name from the ground truth path and load the annotation file
        dataset_name = gt_path.split('/')[2]
        self.random_index = np.load(gt_path + dataset_name + "_annotation_all.npy", allow_pickle=True).item()

    def reset(self):
        # Resetting the batch generator
        self.index = 0
        random.shuffle(self.list_of_examples)  # Randomly shuffle the examples for the next epoch

    def has_next(self):
        # Check if there are any more examples to process
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        # Load the list of videos to process
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)  # Shuffle the list of videos
        self.generate_confidence_mask()  # Generate the confidence masks for each video


    def generate_confidence_mask(self):
        # Iterate over all videos in the list
        for vid in self.list_of_examples:
            # Open the ground truth file for the video
            file_ptr = open(self.gt_path + vid, 'r')
            # Read the file content and split it into lines, excluding the last line which is empty
            content = file_ptr.read().split('\n')[:-1]
            # Initialize an array to hold the class labels of the actions
            classes = np.zeros(len(content))
            # Iterate over all actions in the content
            for i in range(len(classes)):
                # Map the action name to its class index using the actions dictionary
                classes[i] = self.actions_dict[content[i]]
            # Downsample the action labels array by selecting every nth element, where n is the sample rate
            classes = classes[::self.sample_rate]
            # Store the downsampled action labels in the ground truth dictionary
            self.gt[vid] = classes
            # Get the number of frames in the video
            num_frames = classes.shape[0]
            
            # Get the indices of the randomly selected frames for the video
            random_idx = self.random_index[vid]
            
            # Initialize the left and right confidence masks
            left_mask = np.zeros([self.num_classes, num_frames - 1])
            right_mask = np.zeros([self.num_classes, num_frames - 1])
            # Iterate over all pairs of consecutive randomly selected frames
            for j in range(len(random_idx) - 1):
                # Set the elements of the left mask between the indices of the current and next frames to 1 for the action class of the current frame
                left_mask[int(classes[random_idx[j]]), random_idx[j]:random_idx[j + 1]] = 1
                # Set the elements of the right mask between the indices of the current and next frames to 1 for the action class of the next frame
                right_mask[int(classes[random_idx[j + 1]]), random_idx[j]:random_idx[j + 1]] = 1
            
            # Store the left and right masks in the confidence mask dictionary for the video
            self.confidence_mask[vid] = np.array([left_mask, right_mask])


    def next_batch(self, batch_size):
        """
        This function generates a batch of examples for training the model. Each example consists of a tensor of input features, a tensor of target action labels, a tensor of masks indicating the presence of an action class, and a list of confidence masks indicating the boundaries of actions. The inputs and targets are downsampled based on the sample rate, and the tensors are padded to the maximum sequence length in the batch. The target tensor is initialized with -100 as this is the default ignore index in the PyTorch loss functions.
        """
        # Select a subset (batch) from the list of examples based on the current index and the batch size
        batch = self.list_of_examples[self.index:self.index + batch_size]
        # Increment the index by the batch size for the next batch
        self.index += batch_size

        # Initialize empty lists to store the inputs, targets and confidence masks for the batch
        batch_input = []
        batch_target = []
        batch_confidence = []
        # Iterate over each video in the batch
        for vid in batch:
            # Load the features of the video from a numpy file
            features = np.load(self.features_path + vid.split('.')[0] + '.npy')
            # Downsample the features by selecting every nth column, where n is the sample rate, and append them to the batch input list
            batch_input.append(features[:, ::self.sample_rate])
            # Append the ground truth action labels of the video to the batch target list
            batch_target.append(self.gt[vid])
            # Append the confidence mask of the video to the batch confidence list
            batch_confidence.append(self.confidence_mask[vid])

        # Get the lengths of the target sequences in the batch
        length_of_sequences = list(map(len, batch_target))
        # Initialize an input tensor with zeros, of shape [batch size, number of features, maximum sequence length]
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        # Initialize a target tensor with -100, of shape [batch size, maximum sequence length]
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        # Initialize a mask tensor with zeros, of shape [batch size, number of classes, maximum sequence length]
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        # Iterate over each example in the batch
        for i in range(len(batch_input)):
            # Copy the input features of the example to the corresponding slice of the input tensor
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            # Copy the target action labels of the example to the corresponding slice of the target tensor
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            # Set the corresponding slice of the mask tensor to ones
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        # Return the input tensor, target tensor, mask tensor and confidence list for the batch
        return batch_input_tensor, batch_target_tensor, mask, batch_confidence


    def get_single_random(self, batch_size, max_frames):
        # Select a subset (batch) from the list of examples based on the current index and the batch size
        batch = self.list_of_examples[self.index - batch_size:self.index]
        # Initialize a target tensor with -100, of shape [batch size, max frames]
        # -100 is commonly used in PyTorch as an ignore index in the loss calculation
        boundary_target_tensor = torch.ones(len(batch), max_frames, dtype=torch.long) * (-100)
        # Iterate over each video in the batch
        for b, vid in enumerate(batch):
            # Get the random index for the video
            single_frame = self.random_index[vid]
            # Get the ground truth action labels for the video
            gt = self.gt[vid]
            # Convert the random index and ground truth action labels to PyTorch tensors
            frame_idx_tensor = torch.from_numpy(np.array(single_frame))
            gt_tensor = torch.from_numpy(gt.astype(int))
            # Assign the ground truth action labels to the positions of the random index in the target tensor
            boundary_target_tensor[b, frame_idx_tensor] = gt_tensor[frame_idx_tensor]

        # Return the target tensor
        return boundary_target_tensor


    def get_boundary(self, batch_size, pred):
        # This function is to generate pseudo labels
        """
        The function is used to generate pseudo labels for a given batch of videos. 
        These labels are generated based on the features predicted by the model, 
        and the objective is to find action boundaries in each video. 
        The action boundaries are detected by moving forward and then backward 
        through the video frames and finding the boundaries that minimize a certain 
        score. The score is calculated based on the difference between the predicted 
        features and the mean feature value in a segment. 
        The pseudo labels are then used for further learning and prediction.
        """

        # Retrieve the batch of examples based on the current index and batch size
        batch = self.list_of_examples[self.index - batch_size:self.index]
        
        # Get the number of videos and maximum frames from the prediction tensor
        num_video, _, max_frames = pred.size()
        
        # Initialize the boundary target tensor with all elements as -100
        boundary_target_tensor = torch.ones(num_video, max_frames, dtype=torch.long) * (-100)


        # Loop through each video in the batch
        for b, vid in enumerate(batch):
            # Retrieve the random indices and ground truth for the video
            single_idx = self.random_index[vid]
            vid_gt = self.gt[vid]
            
            # Retrieve the predicted features for the video
            features = pred[b]
            
            # Initialize the boundary target for this video with all elements as -100
            boundary_target = np.ones(vid_gt.shape) * (-100)
            
            # Frames before the first random index frame are given the same label
            boundary_target[:single_idx[0]] = vid_gt[single_idx[0]]
            
            # Initialize the left boundary
            left_bound = [0]

            # Forward to find action boundaries
            for i in range(len(single_idx) - 1):
                # Define the start and end frame for this iteration
                start = single_idx[i]
                end = single_idx[i + 1] + 1
                # Initialize left score tensor with zeros
                left_score = torch.zeros(end - start - 1, dtype=torch.float)

                # Loop through each frame in the range and compute the action boundary scores
                for t in range(start + 1, end):
                    center_left = torch.mean(features[:, left_bound[-1]:t], dim=1)
                    diff_left = features[:, start:t] - center_left.reshape(-1, 1)
                    score_left = torch.mean(torch.norm(diff_left, dim=0))

                    center_right = torch.mean(features[:, t:end], dim=1)
                    diff_right = features[:, t:end] - center_right.reshape(-1, 1)
                    score_right = torch.mean(torch.norm(diff_right, dim=0))

                    left_score[t-start-1] = ((t-start) * score_left + (end - t) * score_right)/(end - start)

                # Append the frame with minimum score to the left boundaries
                cur_bound = torch.argmin(left_score) + start + 1
                left_bound.append(cur_bound.item())

            # Backward to find action boundaries
            right_bound = [vid_gt.shape[0]]
            for i in range(len(single_idx) - 1, 0, -1):
                # Define the start and end frame for this iteration
                start = single_idx[i - 1]
                end = single_idx[i] + 1

                # Initialize right score tensor with zeros
                right_score = torch.zeros(end - start - 1, dtype=torch.float)

                # Loop through each frame in the range and compute the action boundary scores
                for t in range(end - 1, start, -1):
                    center_left = torch.mean(features[:, start:t], dim=1)
                    diff_left = features[:, start:t] - center_left.reshape(-1, 1)
                    score_left = torch.mean(torch.norm(diff_left, dim=0))

                    center_right = torch.mean(features[:, t:right_bound[-1]], dim=1)
                    diff_right = features[:, t:end] - center_right.reshape(-1, 1)
                    score_right = torch.mean(torch.norm(diff_right, dim=0))

                    right_score[t-start-1] = ((t-start) * score_left + (end - t) * score_right)/(end - start)

                # Append the frame with minimum score to the right boundaries
                cur_bound = torch.argmin(right_score) + start + 1
                right_bound.append(cur_bound.item())

            # Average two action boundaries for same segment and generate pseudo labels
            left_bound = left_bound[1:]
            right_bound = right_bound[1:]
            num_bound = len(left_bound)
            for i in range(num_bound):
                # Calculate the middle boundary
                temp_left = left_bound[i]
                temp_right = right_bound[num_bound - i - 1]
                middle_bound = int((temp_left + temp_right)/2)
                # Update the boundary target with the ground truth labels
                boundary_target[single_idx[i]:middle_bound] = vid_gt[single_idx[i]]
                boundary_target[middle_bound:single_idx[i + 1] + 1] = vid_gt[single_idx[i + 1]]

            # Frames after the last random index frame are given the same label
            boundary_target[single_idx[-1]:] = vid_gt[single_idx[-1]]  # frames after last single frame has same label

            # Update the boundary target tensor with the computed boundary target
            boundary_target_tensor[b, :vid_gt.shape[0]] = torch.from_numpy(boundary_target)

        # Return the boundary target tensor
        return boundary_target_tensor
