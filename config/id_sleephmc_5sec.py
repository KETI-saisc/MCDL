params = {
    # Train
    "n_epochs": 300,
    "learning_rate": 1e-4,
    "adam_beta_1": 0.9,
    "adam_beta_2": 0.999,
    "adam_epsilon": 1e-8,
    "clip_grad_value": 5.0,
    "evaluate_span": 50,
    "checkpoint_span": 50,

    # Early-stopping
    "no_improve_epochs": 200,

    # Model
    "model": "model-mod-8",
    "n_rnn_layers": 1,
    "n_rnn_units": 128,
    "sampling_rate": 100.0,############################################################################################
    "input_size": 500,################################################################################################# for 5sec
    # "n_classes": 154,################################################################################################## for id
    "n_classes": 5,########################################################################################################################## for id embedding
    "for_sleephmc": False,############################################################################################# for id
    "l2_weight_decay": 1e-3,

    # Dataset
    "dataset": "id_sleephmc",########################################################################################## for id
    # "data_dir": "./data/sleephmc/recordings/ECG+EEG(5sec_for_id)", ################################################## for 5sec original
    # "data_dir": "./data/sleephmc/recordings/test_ECG+EEG(5sec_for_id)", ############################################## for 5sec
    # "data_dir": "./data/sleephmc/recordings/Train(5sec_for_id)", ############################################## for 5sec
    # "data_dir": "./data/sleephmc/recordings/Test(5sec_for_id)_bandpass", ############################################## for 5sec
    "data_dir": "./data/sleephmc/recordings/Test(5sec_for_id)_bandpass_onlysteering", ############################################## for 5sec
    # "data_dir": "./data/sleephmc/recordings/Ori(5sec_for_id)", ############################################## for 5sec ori
    #"data_dir_snore_test": "./data/Snoring_Data/Snoring_Data/Snoring_Dataset/total_npz_snore_test",

    "n_folds": 5, ########## original 20
    # "n_subjects": 20, ## no use

    # Data Augmentation
    # "augment_seq": True,
    # "augment_signal_full": True,
    # "weighted_cross_ent": True,

    "augment_seq": True,
    "augment_signal_full": True,
    "weighted_cross_ent": False,######################################################################################## for id
}

train = params.copy()
train.update({
    "seq_length": 20,
    "batch_size": 15,
})

predict = params.copy()
predict.update({
    "batch_size": 1,
    "seq_length": 1,
})
