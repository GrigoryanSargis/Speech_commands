import os
from random import shuffle
import glob
import torchaudio
import torch
from torch.utils.data import DataLoader
from config import config 


class Preprocessing:
    def __init__(self):
        print('preprocessing instance creation started')

        self.dir_name = config['data_dir']
        self.input_len = config['input_len']  

    def create_iterators(self):
        test_files = self.get_files_from_txt('testing_list.txt')
        val_files = self.get_files_from_txt('validation_list.txt')
        filenames = glob.glob(os.path.join(self.dir_name, '*/**.wav'), recursive=True)
        filenames = [filename for filename in filenames if 'background_noise' not in filename]
        train_files = list(set(filenames) - set(val_files) - set(test_files))
        shuffle(train_files)

        # Get the commands and some prints
        self.commands = self.get_commands()
        self.num_classes = len(self.commands)
        print('len(train_data):', len(train_files))
        print('len(test_data):', len(test_files))
        print('len(val_data):', len(val_files))
        print('commands:', self.commands)
        print('number of commands:', len(self.commands))

        # Create DataLoader objects
        self.train_loader = self.make_data_loader(train_files, shuffle=True)
        self.val_loader = self.make_data_loader(val_files, shuffle=False)
        self.test_loader = self.make_data_loader(test_files, shuffle=False)

    def get_files_from_txt(self, which_txt):
        """
        Reads the specified text file (testing_list.txt or validation_list.txt)
        and returns a list of file paths.
        """
        assert which_txt in ['testing_list.txt', 'validation_list.txt'], 'wrong argument'
        txt_path = os.path.join(self.dir_name, which_txt)
        with open(txt_path, 'r') as f:
            paths = [os.path.join(self.dir_name, line.strip()) for line in f.readlines()]
        shuffle(paths)
        return paths

    def get_commands(self):
        dirs = glob.glob(os.path.join(self.dir_name, "*", ""))
        commands = [os.path.split(os.path.split(dir)[0])[1] for dir in dirs if 'background' not in dir]
        return commands

    def make_data_loader(self, file_list, shuffle):
        """
        Creates a DataLoader from a list of file paths.
        """
        def collate_fn(batch):
            """
            Custom collate function to handle padding and batching.
            """
            waveforms, labels = zip(*[self.process_file(f) for f in batch])
            waveforms = torch.stack(waveforms)
            labels = torch.tensor(labels)
            return waveforms, labels

        return DataLoader(
            file_list,
            batch_size=config['train_params']['batch_size'],
            shuffle=shuffle,
            collate_fn=collate_fn
        )

    def process_file(self, file_path):
        """
        Processes a single file, returning the padded waveform and label.
        """
        waveform, _ = torchaudio.load(file_path)
        waveform = self.add_paddings(waveform)
        label = self.get_label(file_path)
        return waveform, label

    def get_label(self, file_path):
        """
        Extracts the label (folder name) from a file path.
        """
        command = os.path.split(os.path.dirname(file_path))[1]
        label_to_idx = {command: i for i, command in enumerate(self.commands)}
        return label_to_idx[command]

    def add_paddings(self, waveform):
        """
        Pads the waveform to the desired length.
        """
        length = waveform.shape[1]
        if length < self.input_len:
            padding = self.input_len - length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        return waveform[:, :self.input_len]
