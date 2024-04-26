import os
import re

import numpy as np
from sklearn.preprocessing import StandardScaler ###### for standardrization

def get_subject_files(dataset, files, sid):
    """Get a list of files storing each subject data."""

    # Pattern of the subject files from different datasets
    if "mass" in dataset:
        reg_exp = f".*-00{str(sid+1).zfill(2)} PSG.npz"
        # reg_exp = "SS3_00{}\.npz$".format(str(sid+1).zfill(2))
    elif "sleepedf" in dataset:
        reg_exp = f"S[C|T][4|7]{str(sid).zfill(2)}[a-zA-Z0-9]+\.npz$"
        # reg_exp = "[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(str(sid).zfill(2))
    elif "isruc" in dataset:
        reg_exp = f"subject{sid+1}.npz"

    # elif "sleephmc" in dataset:
    #     # reg_exp = f"[1|0][_]{str(sid)}+\.npz"  ###original
    #     reg_exp = f"SN[1|0][0-9]{str(sid)}+\.npz"  #############################################################################

    elif "id_sleephmc" in dataset:
        # reg_exp = f"[1|0][_]{str(sid)}+\.npz"  ###original
        reg_exp = f"SN[1|0][0-9][0-9]{str(sid)}+\.npz"  ################################################################ for id

    elif "sleepmat" in dataset:
        # reg_exp = f"[1|0][_]{str(sid)}+\.npz"  ###original
        reg_exp = f"Piezo_9Axis_[2|3]11[0-9][0-9]{str(sid)}+\.npz"  ############################################sleepmat 데이터 가져오기 위해

    else:
        raise Exception("Invalid datasets.")

    # Get the subject files based on ID
    subject_files = []
    for i, f in enumerate(files):
        pattern = re.compile(reg_exp)
        if pattern.search(f):
            subject_files.append(f)
            print("^^^^^^^^^^^^subject_files^^")###############################
            print(subject_files)###############################################

    return subject_files

#################################################################################################
## predict 할때 폴더안에 모든 데이터를 테스트 하기위해서
def get_subject_files_sleephmc(dataset, files, sid): ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! for predict
    """Get a list of files storing each subject data."""

    # Pattern of the subject files from different datasets
    if "mass" in dataset:
        reg_exp = f".*-00{str(sid+1).zfill(2)} PSG.npz"
        # reg_exp = "SS3_00{}\.npz$".format(str(sid+1).zfill(2))
    elif "sleepedf" in dataset:
        reg_exp = f"S[C|T][4|7]{str(sid).zfill(2)}[a-zA-Z0-9]+\.npz$"
        # reg_exp = "[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(str(sid).zfill(2))
    elif "isruc" in dataset:
        reg_exp = f"subject{sid+1}.npz"

    elif "id_sleephmc" in dataset:
        # reg_exp = f"SN[1|0][0-9][0-9][0-9]+\.npz"  ################################################################ for id
        reg_exp = f"SN[1|0][0-9][0-9]+\.npz"  ################################################################ for id

    else:
        raise Exception("Invalid datasets.")

    # Get the subject files based on ID
    subject_files = []
    for i, f in enumerate(files):
        pattern = re.compile(reg_exp)
        if pattern.search(f):
            subject_files.append(f)
            # print(subject_files)  ####################original

    return subject_files

####################################################################################################

## predict 할때 폴더안에 모든 데이터를 테스트 하기위해서
def get_subject_files_sleepmat(dataset, files, sid): ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! for predict
    """Get a list of files storing each subject data."""

    # Pattern of the subject files from different datasets
    if "mass" in dataset:
        reg_exp = f".*-00{str(sid+1).zfill(2)} PSG.npz"
        # reg_exp = "SS3_00{}\.npz$".format(str(sid+1).zfill(2))
    elif "sleepedf" in dataset:
        reg_exp = f"S[C|T][4|7]{str(sid).zfill(2)}[a-zA-Z0-9]+\.npz$"
        # reg_exp = "[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(str(sid).zfill(2))
    elif "isruc" in dataset:
        reg_exp = f"subject{sid+1}.npz"

    elif "sleephmc" in dataset:
        reg_exp = f"SN[1|0][0-9][0-9]_reshape+\.npz"  #############################################################################

    elif "sleepmat" in dataset:
        # reg_exp = f"[1|0][_]{str(sid)}+\.npz"  ###original
        reg_exp = f"Piezo_9Axis_2[1|2][1|0][0-9][0-9][0-9]+\.npz"  ############################################sleepmat 모든 데이터 가져오기 위해

    else:
        raise Exception("Invalid datasets.")

    # Get the subject files based on ID
    subject_files = []
    for i, f in enumerate(files):
        pattern = re.compile(reg_exp)
        if pattern.search(f):
            subject_files.append(f)
            #print("^^^^^^^^^^^^subject_files^^")  ###############################
            #print(subject_files)  ###############################################

    return subject_files

####################################################################################################

def load_data(subject_files):
    """Load data from subject files."""

    signals = []
    labels = []
    sampling_rate = None
    for sf in subject_files:
        with np.load(sf) as f:
            x = f['x']
            y = f['y']
            fs = f['fs']

            if sampling_rate is None:
                sampling_rate = fs
            elif sampling_rate != fs:
                raise Exception("Mismatch sampling rate.")

            # Reshape the data to match the input of the model - conv2d
            x = np.squeeze(x)
            x = x[:, :, np.newaxis, np.newaxis]
            #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@X shape:", x.shape)###############################################################original

            # Casting
            x = x.astype(np.float32)
            y = y.astype(np.int32)

            signals.append(x)
            labels.append(y)

    return signals, labels, sampling_rate

#############################################################################################################

######################################################################################################################## for id
def load_data_ecg(subject_files):
    """Load data from subject files."""

    signals = []
    labels = []
    sampling_rate = None
    for sf in subject_files:
        with np.load(sf) as f:
            x = f['x2'] # ecg
            # print(x)

            ####################################################################################### standarization
            s = StandardScaler()
            s.fit(x)
            x = s.transform(x)
            #########################################################################################

            y = f['y2'] # subject no.
            fs = f['fs']

            if sampling_rate is None:
                sampling_rate = fs
            elif sampling_rate != fs:
                raise Exception("Mismatch sampling rate.")

            # Reshape the data to match the input of the model - conv2d
            # x = np.squeeze(x)
            x = x[:, :, np.newaxis, np.newaxis]
            #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@X shape:", x.shape)############original

            # Casting
            x = x.astype(np.float32)
            y = y.astype(np.int32)
            # print(x) ## for debug
            # print(y) ## for debug

            signals.append(x)
            labels.append(y)

    return signals, labels, sampling_rate
######################################################################################################################## for id

#############################################################################################################
def load_data_bcg(subject_files):
    """Load data from subject files."""

    signals = []
    labels = []
    sampling_rate = None
    for sf in subject_files:
        with np.load(sf) as f:

            x1 = f['bcg_1']  ############################################################## Channel 1
            x2 = f['bcg_2']  ############################################################## Channel 2
            x3 = f['bcg_3']  ############################################################## Channel 3
            x4 = f['bcg_4']  ############################################################## Channel 4
            x5 = f['bcg_5']  ############################################################## Channel 5

            x_1 = x1 + x2 + x3 + x4 + x5

            ####################################################################################### standarization
            s = StandardScaler()
            s.fit(x_1)
            x_1 = s.transform(x_1)
            #########################################################################################

            y = f['sleep_stage']
            fs = f['fs']

            if sampling_rate is None:
                sampling_rate = fs
            elif sampling_rate != fs:
                raise Exception("Mismatch sampling rate.")

            # Reshape the data to match the input of the model - conv2d
            x_1 = np.squeeze(x_1)  ######################################################### Channel 1
            # x_2 = np.squeeze(x_2)####################################################### Channel 2

            x_1 = x_1[:, :, np.newaxis, np.newaxis]  ####################################### Channel 1
            # x_2 = x_2[:, :, np.newaxis, np.newaxis]##################################### Channel 2

            temp_x = np.ones((len(x_1), 3000, 1, 1))  ##################################
            temp_x[:, :, 0, 0] = x_1[:, :, 0, 0]  ########### add Channel 1
            # temp_x[:, :, 1, 0] =  x_2[:, :, 0, 0] ########### add Channel 2

            x = temp_x

            #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@X shape:", x.shape)############################################################original

            # Casting
            x = x.astype(np.float32)
            y = y.astype(np.int32)

            signals.append(x)
            labels.append(y)

    return signals, labels, sampling_rate


#############################################################################################################

def load_data2(subject_files):
    """Load data from subject files."""

    signals = []
    labels = []
    sampling_rate = None
    for sf in subject_files:
        with np.load(sf) as f:
            #             x = f['x'] ############################################################### Channel 1
            #             x2 = f['x2']############################################################## Channel 2
            #             y = f['y']
            #             fs = f['fs']

            x1 = f['bcg_1']  ############################################################## Channel 1
            x2 = f['bcg_2']  ############################################################## Channel 2
            x3 = f['bcg_3']  ############################################################## Channel 3
            x4 = f['bcg_4']  ############################################################## Channel 4
            x5 = f['bcg_5']  ############################################################## Channel 5

            x6 = f['ppg_1']  ############################################################## Channel 6
            x7 = f['fsr_1']  ############################################################## Channel 7

            x8 = f['acc_x']  ############################################################## Channel 8
            x9 = f['acc_y']  ############################################################## Channel 9
            x10 = f['acc_z']  ############################################################## Channel 10
            x11 = f['gyr_x']  ############################################################## Channel 11
            x12 = f['gyr_y']  ############################################################## Channel 12
            x13 = f['gyr_z']  ############################################################## Channel 13

            x_1 = x1 + x2 + x3 + x4 + x5 + x6
            x_2 = x8 + x9 + x10 + x11 + x12 + x13

            ####################################################################################### standarization
            s = StandardScaler()
            s.fit(x_1)
            x_1 = s.transform(x_1)

            t = StandardScaler()
            t.fit(x_2)
            x_2 = t.transform(x_2)
            #########################################################################################

            y = f['sleep_stage']
            fs = f['fs']

            if sampling_rate is None:
                sampling_rate = fs
            elif sampling_rate != fs:
                raise Exception("Mismatch sampling rate.")

            # Reshape the data to match the input of the model - conv2d
            x_1 = np.squeeze(x_1)  ######################################################### Channel 1
            x_2 = np.squeeze(x_2)  ####################################################### Channel 2

            x_1 = x_1[:, :, np.newaxis, np.newaxis]  ####################################### Channel 1
            x_2 = x_2[:, :, np.newaxis, np.newaxis]  ##################################### Channel 2

            temp_x = np.ones((len(x_1), 3000, 2, 1))  ##################################
            temp_x[:, :, 0, 0] = x_1[:, :, 0, 0]  ########### add Channel 1
            temp_x[:, :, 1, 0] = x_2[:, :, 0, 0]  ########### add Channel 2

            x = temp_x

            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@X shape:", x.shape)

            # Casting
            x = x.astype(np.float32)
            y = y.astype(np.int32)

            signals.append(x)
            labels.append(y)

    return signals, labels, sampling_rate


#############################################################################################################

def load_data_multi13(subject_files):
    """Load data from subject files."""

    signals = []
    labels = []
    sampling_rate = None
    for sf in subject_files:
        with np.load(sf) as f:
            x1 = f['bcg_1']  ############################################################## Channel 1
            x2 = f['bcg_2']  ############################################################## Channel 2
            x3 = f['bcg_3']  ############################################################## Channel 3
            x4 = f['bcg_4']  ############################################################## Channel 4
            x5 = f['bcg_5']  ############################################################## Channel 5

            x6 = f['ppg_1']  ############################################################## Channel 6
            x7 = f['fsr_1']  ############################################################## Channel 7

            x8 = f['acc_x']  ############################################################## Channel 8
            x9 = f['acc_y']  ############################################################## Channel 9
            x10 = f['acc_z']  ############################################################## Channel 10
            x11 = f['gyr_x']  ############################################################## Channel 11
            x12 = f['gyr_y']  ############################################################## Channel 12
            x13 = f['gyr_z']  ############################################################## Channel 13

            y = f['sleep_stage']
            fs = f['fs']

            if sampling_rate is None:
                sampling_rate = fs
            elif sampling_rate != fs:
                raise Exception("Mismatch sampling rate.")

            # Reshape the data to match the input of the model - conv2d
            x1 = np.squeeze(x1)  ####################################################### Channel 1
            x2 = np.squeeze(x2)  ####################################################### Channel 2
            x3 = np.squeeze(x3)  ####################################################### Channel 3
            x4 = np.squeeze(x4)  ####################################################### Channel 4
            x5 = np.squeeze(x5)  ####################################################### Channel 5

            x6 = np.squeeze(x6)  ####################################################### Channel 6
            x7 = np.squeeze(x7)  ####################################################### Channel 7

            x8 = np.squeeze(x8)  ####################################################### Channel 8
            x9 = np.squeeze(x9)  ####################################################### Channel 9
            x10 = np.squeeze(x10)  ##################################################### Channel 10
            x11 = np.squeeze(x11)  ##################################################### Channel 11
            x12 = np.squeeze(x12)  ##################################################### Channel 12
            x13 = np.squeeze(x13)  ##################################################### Channel 13

            x1 = x1[:, :, np.newaxis, np.newaxis]  ##################################### Channel 1
            x2 = x2[:, :, np.newaxis, np.newaxis]  ##################################### Channel 2
            x3 = x3[:, :, np.newaxis, np.newaxis]  ##################################### Channel 3
            x4 = x4[:, :, np.newaxis, np.newaxis]  ##################################### Channel 4
            x5 = x5[:, :, np.newaxis, np.newaxis]  ##################################### Channel 5

            x6 = x6[:, :, np.newaxis, np.newaxis]  ##################################### Channel 6
            x7 = x7[:, :, np.newaxis, np.newaxis]  ##################################### Channel 7

            x8 = x8[:, :, np.newaxis, np.newaxis]  ##################################### Channel 8
            x9 = x9[:, :, np.newaxis, np.newaxis]  ##################################### Channel 9
            x10 = x10[:, :, np.newaxis, np.newaxis]  ################################### Channel 10
            x11 = x11[:, :, np.newaxis, np.newaxis]  ################################### Channel 11
            x12 = x12[:, :, np.newaxis, np.newaxis]  ################################### Channel 12
            x13 = x13[:, :, np.newaxis, np.newaxis]  ################################### Channel 13

            temp_x = np.ones((len(x1), 3000, 13, 1))  ################################## Multi 13 channel
            temp_x[:, :, 0, 0] = x1[:, :, 0, 0]  ########### add Channel 1
            temp_x[:, :, 1, 0] = x2[:, :, 0, 0]  ########### add Channel 2
            temp_x[:, :, 2, 0] = x3[:, :, 0, 0]  ########### add Channel 3
            temp_x[:, :, 3, 0] = x4[:, :, 0, 0]  ########### add Channel 4
            temp_x[:, :, 4, 0] = x5[:, :, 0, 0]  ########### add Channel 5

            temp_x[:, :, 5, 0] = x6[:, :, 0, 0]  ########### add Channel 6
            temp_x[:, :, 6, 0] = x7[:, :, 0, 0]  ########### add Channel 7

            temp_x[:, :, 7, 0] = x8[:, :, 0, 0]  ########### add Channel 8
            temp_x[:, :, 8, 0] = x9[:, :, 0, 0]  ########### add Channel 9
            temp_x[:, :, 9, 0] = x10[:, :, 0, 0]  ########### add Channel 10
            temp_x[:, :, 10, 0] = x11[:, :, 0, 0]  ########### add Channel 11
            temp_x[:, :, 11, 0] = x12[:, :, 0, 0]  ########### add Channel 12
            temp_x[:, :, 12, 0] = x13[:, :, 0, 0]  ########### add Channel 13

            x = temp_x

            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@X shape:", x.shape)

            # Casting
            x = x.astype(np.float32)
            y = y.astype(np.int32)

            signals.append(x)
            labels.append(y)

    return signals, labels, sampling_rate


#############################################################################################################

def load_data3(subject_files):
    """Load data from subject files."""

    signals = []
    labels = []
    sampling_rate = None
    for sf in subject_files:
        with np.load(sf) as f:
            #             x = f['x'] ############################################################### Channel 1
            #             x2 = f['x2']############################################################## Channel 2
            #             y = f['y']
            #             fs = f['fs']

            x1 = f['bcg_1']  ############################################################## Channel 1
            x2 = f['bcg_2']  ############################################################## Channel 2
            x3 = f['bcg_3']  ############################################################## Channel 3
            x4 = f['bcg_4']  ############################################################## Channel 4
            x5 = f['bcg_5']  ############################################################## Channel 5

            x6 = f['ppg_1']  ############################################################## Channel 6
            x7 = f['fsr_1']  ############################################################## Channel 7

            x8 = f['acc_x']  ############################################################## Channel 8
            x9 = f['acc_y']  ############################################################## Channel 9
            x10 = f['acc_z']  ############################################################## Channel 10
            x11 = f['gyr_x']  ############################################################## Channel 11
            x12 = f['gyr_y']  ############################################################## Channel 12
            x13 = f['gyr_z']  ############################################################## Channel 13

            x_1 = x1 + x2 + x3 + x4 + x5
            x_2 = x7  # + x6
            x_3 = x8 + x9 + x10 + x11 + x12 + x13

            ####################################################################################### standarization
            s = StandardScaler()
            s.fit(x_1)
            x_1 = s.transform(x_1)

            t = StandardScaler()
            t.fit(x_2)
            x_2 = t.transform(x_2)

            w = StandardScaler()
            w.fit(x_3)
            x_3 = t.transform(x_3)
            #########################################################################################

            y = f['sleep_stage']
            fs = f['fs']

            if sampling_rate is None:
                sampling_rate = fs
            elif sampling_rate != fs:
                raise Exception("Mismatch sampling rate.")

            # Reshape the data to match the input of the model - conv2d
            x_1 = np.squeeze(x_1)  ####################################################### Channel 1
            x_2 = np.squeeze(x_2)  ####################################################### Channel 2
            x_3 = np.squeeze(x_3)  ####################################################### Channel 3

            x_1 = x_1[:, :, np.newaxis, np.newaxis]  ##################################### Channel 1
            x_2 = x_2[:, :, np.newaxis, np.newaxis]  ##################################### Channel 2
            x_3 = x_3[:, :, np.newaxis, np.newaxis]  ##################################### Channel 3

            temp_x = np.ones((len(x_1), 3000, 3, 1))  ##################################
            temp_x[:, :, 0, 0] = x_1[:, :, 0, 0]  ########### add Channel 1
            temp_x[:, :, 1, 0] = x_2[:, :, 0, 0]  ########### add Channel 2
            temp_x[:, :, 2, 0] = x_3[:, :, 0, 0]  ########### add Channel 3

            x = temp_x

            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@X shape:", x.shape)

            # Casting
            x = x.astype(np.float32)
            y = y.astype(np.int32)

            signals.append(x)
            labels.append(y)

    return signals, labels, sampling_rate


#############################################################################################################

#############################################################################################################

def load_data4(subject_files):
    """Load data from subject files."""

    signals = []
    labels = []
    sampling_rate = None
    for sf in subject_files:
        with np.load(sf) as f:
            #             x = f['x'] ############################################################### Channel 1
            #             x2 = f['x2']############################################################## Channel 2
            #             y = f['y']
            #             fs = f['fs']

            x1 = f['bcg_1']  ############################################################## Channel 1
            x2 = f['bcg_2']  ############################################################## Channel 2
            x3 = f['bcg_3']  ############################################################## Channel 3
            x4 = f['bcg_4']  ############################################################## Channel 4
            x5 = f['bcg_5']  ############################################################## Channel 5

            x6 = f['ppg_1']  ############################################################## Channel 6
            x7 = f['fsr_1']  ############################################################## Channel 7

            x8 = f['acc_x']  ############################################################## Channel 8
            x9 = f['acc_y']  ############################################################## Channel 9
            x10 = f['acc_z']  ############################################################## Channel 10
            x11 = f['gyr_x']  ############################################################## Channel 11
            x12 = f['gyr_y']  ############################################################## Channel 12
            x13 = f['gyr_z']  ############################################################## Channel 13

            x_1 = x1 + x2 + x3 + x4 + x5
            x_2 = x6  # + x7
            x_3 = x8 + x9 + x10 + x11 + x12 + x13
            x_4 = x7

            ####################################################################################### standarization
            s = StandardScaler()
            s.fit(x_1)
            x_1 = s.transform(x_1)

            t = StandardScaler()
            t.fit(x_2)
            x_2 = t.transform(x_2)

            w = StandardScaler()
            w.fit(x_3)
            x_3 = t.transform(x_3)

            q = StandardScaler()
            q.fit(x_4)
            x_4 = q.transform(x_4)

            #########################################################################################

            y = f['sleep_stage']
            fs = f['fs']

            if sampling_rate is None:
                sampling_rate = fs
            elif sampling_rate != fs:
                raise Exception("Mismatch sampling rate.")

            # Reshape the data to match the input of the model - conv2d
            x_1 = np.squeeze(x_1)  ####################################################### Channel 1
            x_2 = np.squeeze(x_2)  ####################################################### Channel 2
            x_3 = np.squeeze(x_3)  ####################################################### Channel 3
            x_4 = np.squeeze(x_4)  ####################################################### Channel 4

            x_1 = x_1[:, :, np.newaxis, np.newaxis]  ##################################### Channel 1
            x_2 = x_2[:, :, np.newaxis, np.newaxis]  ##################################### Channel 2
            x_3 = x_3[:, :, np.newaxis, np.newaxis]  ##################################### Channel 3
            x_4 = x_4[:, :, np.newaxis, np.newaxis]  ##################################### Channel 4

            temp_x = np.ones((len(x_1), 3000, 4, 1))  ################################## 4 channel
            temp_x[:, :, 0, 0] = x_1[:, :, 0, 0]  ########### add Channel 1
            temp_x[:, :, 1, 0] = x_2[:, :, 0, 0]  ########### add Channel 2
            temp_x[:, :, 2, 0] = x_3[:, :, 0, 0]  ########### add Channel 3
            temp_x[:, :, 3, 0] = x_4[:, :, 0, 0]  ########### add Channel 4

            x = temp_x

            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@X shape:", x.shape)

            # Casting
            x = x.astype(np.float32)
            y = y.astype(np.int32)

            signals.append(x)
            labels.append(y)

    return signals, labels, sampling_rate

#############################################################################################################