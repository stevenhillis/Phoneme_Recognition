import ast
import math
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.functional as F
from Levenshtein import distance

from torch.utils.data import Dataset, DataLoader
# pip install python-Levenshtein

from Net import Net
from Test_Dataset import Test_Dataset
from Train_Dataset import Train_Dataset
from ctcdecode import CTCBeamDecoder
from operator import itemgetter


class Recognizer_LSTM:
    def __init__(self):
        # self.batch_size = 1
        self.batch_size = 32
        self.num_workers = 8
        self.beam_size = 100
        self.hidden_size = 256
        self.train_data_params = {'batch_size': self.batch_size,
                                  'shuffle': True,
                                  'num_workers': self.num_workers,
                                  'pin_memory': True,
                                  'collate_fn': my_train_collate}
        self.val_data_params = {'batch_size': self.batch_size,
                                'shuffle': False,
                                'num_workers': self.num_workers,
                                'pin_memory': True,
                                'collate_fn': my_train_collate}
        self.test_data_params = {'batch_size': 1,
                                 'shuffle': False,
                                 'num_workers': self.num_workers,
                                 'pin_memory': True,
                                 'collate_fn': my_test_collate}

        training_gen_start_time = time.time()
        print('Creating the training dataset.')
        self.training_dataset = Train_Dataset("wsj0_train.npy", "wsj0_train_merged_labels.npy")
        print('Creating the training dataset took {:0.2f} seconds'.format(time.time() - training_gen_start_time))
        print('Num training batches per epoch is ' + repr(math.ceil(len(self.training_dataset) / self.batch_size)) + '.')

        print('Creating the validation dataset.')
        self.validation_dataset = Train_Dataset("wsj0_dev.npy", "wsj0_dev_merged_labels.npy")

        print('Creating the test dataset.')
        self.test_dataset = Test_Dataset("transformed_test_data.npy")

        self.net = Net(self.hidden_size)
        self.criterion = nn.CTCLoss(reduction='none')
        with open('list_of_phonemes.txt', 'r') as f:
            self.vocab = ast.literal_eval(f.read())
        with open('list_of_single_phonemes.txt', 'r') as f:
            self.single_phonemes = ast.literal_eval(f.read())
        self.phoneme_map = {}
        for index, p in enumerate(self.single_phonemes):
            self.phoneme_map[p] = self.vocab[index]
        self.decoder = CTCBeamDecoder(self.single_phonemes, beam_width=100, log_probs_input=True, blank_id=0)
        self.net.apply(self.init_weights)

    def train(self, epochs, gpu, lr=0.001, weight_decay=0):
        device = torch.device('cuda' if gpu else 'cpu')
        self.net = self.net.to(device)

        training_generator = DataLoader(self.training_dataset, **self.train_data_params)
        validation_generator = DataLoader(self.validation_dataset, **self.val_data_params)

        optimizer = torch.optim.Adam(self.net.parameters(), lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)

        print('Beginning training.')
        for epoch in range(epochs):
            print('Device is ' + repr(device))
            start = time.time()
            count = 0
            cumulative_train_loss = 0.0
            for batch in training_generator:
                self.net.zero_grad()
                frames, labels, frame_sizes, label_sizes, unsort_index = batch
                # print("Frames shape: " + repr(frames.size()))
                # # print(repr(frames))
                # print("Labels shape: " + repr(labels.size()))
                # print("Frame sizes shape: " + repr(frame_sizes.size()))
                # print("Frame sizes: " + repr(frame_sizes))
                # print("Label sizes shape: " + repr(label_sizes.size()))
                # print("Label sizes: " + repr(label_sizes))

                self.net.train()
                if (count % 50 == 0 and count > 0):
                    print(
                        "So far, training on {:} batches has taken {:.2f} minutes. Average training loss is {:.2f}"
                        .format(count, (time.time() - start) / 60, cumulative_train_loss / count))
                frames, labels, frame_sizes, label_sizes = frames.to(device), labels.to(device), \
                                                          frame_sizes.to(device), label_sizes.to(device)

                frames = rnn.pack_padded_sequence(frames, frame_sizes, batch_first=True)
                output = self.net(frames)
                output = output.transpose(0,1)
                loss = self.criterion(output, labels, frame_sizes, label_sizes).sum() / self.batch_size
                cumulative_train_loss += loss.item()

                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.25)
                optimizer.step()
                optimizer.zero_grad()
                count += 1

            print("After epoch ", repr(epoch))
            print("Training loss: {:.2f}".format(cumulative_train_loss / count))

            self.net.eval()
            cumulative_val_loss = 0.0
            cumulative_edit_distance = 0.0
            val_start = time.time()
            val_count = 0
            with torch.set_grad_enabled(False):
                for val_frames, val_labels, val_frame_sizes, val_label_sizes, unsort_index in validation_generator:
                    val_frames, val_labels, val_frame_sizes, val_label_sizes = val_frames.to(device), \
                                                                              val_labels.to(device), val_frame_sizes.to(device), val_label_sizes.to(device)
                    val_frames = rnn.pack_padded_sequence(val_frames, val_frame_sizes, batch_first=True)
                    val_output = self.net(val_frames)
                    val_output = val_output.transpose(0, 1)
                    val_loss = self.criterion(val_output, val_labels, val_frame_sizes, val_label_sizes).sum() / self.batch_size
                    cumulative_val_loss += val_loss.item()

                    val_output = val_output.transpose(1, 0).cpu().detach()
                    # print("Val output shape: " + repr(val_output.size()))
                    # print("Val output type: " + repr(val_output.dtype))
                    # print("Val frame sizes: " + repr(val_frame_sizes.size()) + ", " + repr(val_frame_sizes))
                    # print("Vocab size: " + repr(len(self.vocab)))
                    # print(val_output)
                    val_frame_sizes = val_frame_sizes.cpu().detach()
                    self.decoder = CTCBeamDecoder(self.single_phonemes, beam_width=100, log_probs_input=True, blank_id=0)
                    val_prediction, scores, time_steps, decoded_lengths = self.decoder.decode(val_output, val_frame_sizes)
                    decoded_lengths = decoded_lengths[:, 0]
                    val_prediction = val_prediction[:,0,:]
                    val_prediction = [val_prediction[batch_num,:index.item()] for batch_num, index in enumerate(decoded_lengths)]
                    val_prediction = [item for sublist in val_prediction for item in sublist]
                    val_prediction = self.convert_to_single_characters(val_prediction)
                    val_labels = val_labels.cpu().numpy().tolist()
                    val_labels = self.convert_to_single_characters_from_numpy(val_labels)
                    val_edit_distance = distance(val_prediction, val_labels) / self.batch_size
                    cumulative_edit_distance += val_edit_distance
                    val_count += 1
            print("Validation took {:.2f} minutes.".format((time.time() - val_start)/60))
            print("Validation loss: {:.2f}".format(cumulative_val_loss / val_count))
            print("Validation edit distance: {:.2f}".format(cumulative_edit_distance / val_count))
            # print("")
            scheduler.step(cumulative_val_loss / val_count)

            stop = time.time()
            print("This epoch took {:.2f} minutes.".format((stop - start) / 60))
            backup_file = "model_{:}_valEdit_{:.2f}.pt".format(epoch, (cumulative_edit_distance / val_count))
            torch.save(self.net.state_dict(), backup_file)

        self.net = self.net.cpu()
        print("Finished training.")

    def test(self, gpu, model_path):
        device = torch.device('cuda' if gpu else 'cpu')

        self.net = Net(self.hidden_size)
        self.net.load_state_dict(torch.load(model_path))
        self.net = self.net.to(device)
        self.net.eval()

        print('Creating the testing generator.')
        test_generator = DataLoader(self.test_dataset, **self.test_data_params)
        print('Beginning testing.')
        count = 0
        out_line = 0
        start = time.time()
        out_file = open("output.csv", "w")
        out_file.write("Id,Predicted\n")
        out_file.close()

        with torch.set_grad_enabled(False):
            for test_frames, test_frame_lengths, unpermute_index in test_generator:
                if (count % 100 == 0 and count > 0):
                    print("So far, testing on {:} examples has taken {:.2f} minutes.".format(count,
                                                                                            (time.time() - start) / 60))

                test_frames, test_frame_lengths = test_frames.to(device), test_frame_lengths.to(device)
                test_frames = rnn.pack_padded_sequence(test_frames, test_frame_lengths, batch_first=True)
                test_output = self.net(test_frames)
                test_output = test_output.cpu().detach()
                test_frame_lengths = test_frame_lengths.cpu().detach()
                self.decoder = CTCBeamDecoder(self.single_phonemes, beam_width=100, log_probs_input=True, blank_id=0)
                test_predictions, _, _, decoded_lengths = self.decoder.decode(test_output, test_frame_lengths)
                decoded_lengths = decoded_lengths[:, 0]
                test_predictions = test_predictions[:, 0, :]
                print("Decoded lengths before reorder: ")
                print(decoded_lengths)
                print("Unpermute index: ")
                print(unpermute_index.data.numpy())
                decoded_lengths = decoded_lengths[unpermute_index.data.numpy()]
                print("Decoded lengths afer reorder: ")
                print(decoded_lengths)
                test_predictions = test_predictions[unpermute_index.data.numpy()]
                test_predictions = [test_predictions[batch_num, :index.item()] for batch_num, index in
                                    enumerate(decoded_lengths)]
                print("Test prediction length: ")
                print(len(test_predictions[0]))
                print("Confirming we have just one batch: ")
                print(len(test_predictions))
                test_predictions = self.convert_to_single_characters_for_test(test_predictions)
                out_file = open("output.csv", "a+")
                for test_prediction in test_predictions:
                    out = repr(out_line) + "," + test_prediction + "\n"
                    out_file.write(out)
                    out_line += 1
                out_file.close()
                count += 1

        print('Finished testing.')
        self.net = self.net.cpu()

    def convert_to_single_characters(self, predictions):
        return ''.join([self.single_phonemes[p.item()] for p in predictions])

    def convert_to_single_characters_from_numpy(self, predictions):
        return ''.join([self.single_phonemes[p] for p in predictions])

    def convert_to_single_characters_for_test(self, predictions):
        return [''.join([self.single_phonemes[p] for p in prediction]) for prediction in predictions]


    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
        elif type(m) == nn.LSTM:
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    upper = 1/np.sqrt(self.hidden_size)
                    nn.init.uniform_(param, -upper, upper)



def my_train_collate(batch):
    frames = [instance[0] for instance in batch]
    labels = np.array([instance[1].numpy() for instance in batch])

    frame_lengths = torch.LongTensor([len(frame) for frame in frames])
    padded_frames = rnn.pad_sequence(frames, batch_first=True)
    frame_lengths, permute_index = frame_lengths.sort(0, descending=True)
    sorted_padded_frames = padded_frames[permute_index.data.numpy()]
    # print(sorted_padded_frames)

    sorted_labels = labels[permute_index.data.numpy()]
    sorted_frequency_lengths = torch.LongTensor([len(label) for label in sorted_labels])
    sorted_labels = [torch.LongTensor(labels) for labels in sorted_labels]
    sorted_labels = torch.cat(sorted_labels)

    _, unpermute_index = permute_index.sort(0)

    return (sorted_padded_frames, sorted_labels, frame_lengths, sorted_frequency_lengths, unpermute_index)

def my_test_collate(frames):
    frame_lengths = torch.LongTensor([len(frame) for frame in frames])
    padded_frames = rnn.pad_sequence(frames, batch_first=True)

    frame_lengths, permute_index = frame_lengths.sort(0, descending=True)
    sorted_padded_frames = padded_frames[permute_index.data.numpy()]
    _, unpermute_index = permute_index.sort(0)

    return sorted_padded_frames, frame_lengths, unpermute_index
