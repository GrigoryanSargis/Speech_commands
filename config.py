# You only need to customize model_params if needed, dont change anything else
config = {
    'data_dir': 'speech_commands_v0.02',
    'model_name': '1',
    'sample_rate': 8000,
    'input_len': 16000,
    'NUM_PARALLEL_CALLS': 32,
    'feature': {
        'window_size_ms': 0.025,
        'window_stride': 0.01,
        'fft_length': 512,
        'mfcc_lower_edge_hertz': 0.0,
        'mfcc_upper_edge_hertz': 4000.0,  
        'mfcc_num_mel_bins': 64
    },
    'train_params': {
        'batch_size': 128,
        'epochs': 25,
        'steps_per_epoch': None,
        'latest_checkpoint_step': 50,
        'summary_step': 50, #also summary step
        'max_checkpoints_to_keep': 5,
    },
    'model_params': {
        # here can go any params you need for your model
        'dropout': 0.2,
        'data_normalization': True,
    }

}
