
from torch.utils.data import Dataset
import torch
import copy
from utils import Accuracy
from Server.ServerBase import Server
from Client.ClientFedMD import ClientFedMD
from tqdm import tqdm
import numpy as np
from utils import average_weights
from mem_utils import MemReporter
import time
from sampling import LocalDataset, LocalDataloaders, partition_data
import gc

class ServerFedMD(Server):
    def __init__(self, args, global_model,Loader_train,Loaders_local_test,Loader_global_test, pub_test,logger,device):
        super().__init__(args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device)
        dict_pub = [np.random.randint(low=0,high=10000,size = 1000)]
        self.public_data = LocalDataloaders(pub_test,dict_pub,args.batch_size,ShuffleorNot = False,frac=1)[0]
    
    def Create_Clints(self):
        for idx in range(self.args.num_clients):
            self.LocalModels.append(ClientFedMD(self.args, copy.deepcopy(self.global_model),self.Loaders_train[idx], self.Loaders_local_test[idx], loader_pub = self.public_data, idx=idx, logger=self.logger, code_length = self.args.code_len, num_classes = self.args.num_classes, device=self.device))
            
            
    def train(self):
        reporter = MemReporter()
        start_time = time.time()
        train_loss = []
        global_weights = self.global_model.state_dict()  # Get the state dictionary of the global model
        for epoch in tqdm(range(self.args.num_epochs)):  # Iterate over the number of epochs
            Knowledges = []  # Initialize a list to store knowledge from clients
            test_accuracy = 0  # Initialize test accuracy
            local_weights, local_losses = [], []  # Initialize lists to store local weights and losses
            print(f'\n | Global Training Round : {epoch+1} |\n')  # Print the current global training round
            m = max(int(self.args.sampling_rate * self.args.num_clients), 1)  # Calculate the number of clients to sample
            idxs_users = np.random.choice(range(self.args.num_clients), m, replace=False)  # Randomly select clients
            for idx in idxs_users:  # Iterate over the selected clients
                if self.args.upload_model == True:  # Check if the model should be uploaded
                    self.LocalModels[idx].load_model(global_weights)  # Load the global model weights into the local model
                if epoch < 1:    
                    w, loss = self.LocalModels[idx].update_weights(global_round=epoch)  # Update weights for the first epoch
                    local_losses.append(copy.deepcopy(loss))  # Append the loss to the local losses list
                    local_weights.append(copy.deepcopy(w))  # Append the weights to the local weights list
                    acc = self.LocalModels[idx].test_accuracy() # Test the accuracy of the local model
                    test_accuracy += acc  # Accumulate the test accuracy
                else:
                    w, loss = self.LocalModels[idx].update_weights_MD(global_round=epoch, knowledges=global_soft_prediciton, lam=0.1, temp=self.args.temp)  # Update weights using knowledge distillation
                    local_losses.append(copy.deepcopy(loss))  # Append the loss to the local losses list
                    local_weights.append(copy.deepcopy(w))  # Append the weights to the local weights list
                    acc = self.LocalModels[idx].test_accuracy()  # Test the accuracy of the local model
                    test_accuracy += acc  # Accumulate the test accuracy
                knowledges = self.LocalModels[idx].generate_knowledge(temp=self.args.temp)  # Generate knowledge from the local model
                Knowledges.append(torch.stack(knowledges))  # Append the knowledge to the Knowledges list
            global_soft_prediciton = []  # Initialize a list to store global soft predictions
            batch_pub = Knowledges[0].shape[0]  # Get the batch size of the public data
            for i in range(batch_pub):  # Iterate over the batch size
                num = Knowledges[0].shape[1]  # Get the number of classes, the sample number of each batch
                soft_label = torch.zeros(num, self.args.num_classes)  # Initialize a tensor to store soft labels
                for idx in idxs_users:  # Iterate over the selected clients
                    soft_label += Knowledges[idx][i]  # Accumulate the soft labels from the clients
                soft_label = soft_label / len(idxs_users)  # Average the soft labels
                global_soft_prediciton.append(soft_label)  # Append the soft label to the global soft predictions list
            del Knowledges  # Delete the Knowledges list to free memory
            gc.collect()  # Collect garbage to free memory

            # Update global weights
            global_weights = average_weights(local_weights)  # Average the local weights to get the new global weights
            self.global_model.load_state_dict(global_weights)  # Load the new global weights into the global model
            loss_avg = sum(local_losses) / len(local_losses)  # Calculate the average loss
            train_loss.append(loss_avg)  # Append the average loss to the train loss list
            print("average loss:  ", loss_avg)  # Print the average loss
            print('average local test accuracy:', test_accuracy / self.args.num_clients)  # Print the average local test accuracy
            print('global test accuracy: ', self.global_test_accuracy())  # Print the global test accuracy
            
        print('Training is completed.')
        end_time = time.time()
        print('running time: {} s '.format(end_time - start_time))
        reporter.report()