##!!! using two(2) features!!!
## 5 sec

import argparse
import glob
import math
import ntpath
import os
import shutil
import pyedflib
import numpy as np
import pandas as pd
from scipy import signal###########################################################
from sleepstage import stage_dict
from logger import get_logger


# Have to manually define based on the dataset
ann2label = {
    "Sleep stage W": 0,
    "Sleep stage N1": 1,
    "Sleep stage N2": 2,
    "Sleep stage N3": 3, "Sleep stage N4": 3, # Follow AASM Manual
    "Sleep stage R": 4,
    "Sleep stage ?": 6,
    "Movement time": 5
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/sleephmc/recordings", ## need to be input
                        help="File path to the Sleep-HMC dataset.")
    parser.add_argument("--output_dir", type=str, default="./data/sleephmc/recordings/test_ECG+EEG(5sec_for_id)", ## need to be input
                        help="Directory where to save outputs.")
    parser.add_argument("--select_ch1", type=str, default="EEG F4-M1",######################################(1st)feature
                        help="Name of the 1st channel in the dataset.")
    parser.add_argument("--select_ch2", type=str, default="ECG",###########################################(2nd)feature
                        help="Name of the 2nd channel in the dataset.")
    parser.add_argument("--log_file", type=str, default="info_ch_extract_hmc.log",
                        help="Log file.")
    args = parser.parse_args()

    # Output dir
    if not os.path.exists(args.output_dir): ## output 폴더 없으면
        os.makedirs(args.output_dir) ## 생성
    else:
        shutil.rmtree(args.output_dir) ## 기존 output 폴더 있으면, 내용 삭제
        os.makedirs(args.output_dir) ## 생성

    args.log_file = os.path.join(args.output_dir, args.log_file) # 로그파일 위치

    # Create logger
    logger = get_logger(args.log_file, level="info") # 로그파일 생성

    # Select channel
    select_ch1 = args.select_ch1
    print("Channel 1 : ", select_ch1)
    select_ch2 = args.select_ch2
    print("Channel 2 : ", select_ch2)

    # Read raw and annotation from EDF files
    psg_fnames = glob.glob(os.path.join(args.data_dir, "SN[0-9][0-9][0-9].edf"))############################ 파일명 확인
    ann_fnames = glob.glob(os.path.join(args.data_dir, "SN[0-9][0-9][0-9]_sleepscoring.edf"))############### 파일명 확인
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    for i in range(len(psg_fnames)):

        logger.info("Loading ...")
        logger.info("Signal file: {}".format(psg_fnames[i]))
        logger.info("Annotation file: {}".format(ann_fnames[i]))

        psg_f = pyedflib.EdfReader(psg_fnames[i])
        ann_f = pyedflib.EdfReader(ann_fnames[i])

        assert psg_f.getStartdatetime() == ann_f.getStartdatetime()
        start_datetime = psg_f.getStartdatetime()
        logger.info("Start datetime: {}".format(str(start_datetime)))

        file_duration = psg_f.getFileDuration()
        logger.info("File duration: {} sec".format(file_duration))

        epoch_duration = psg_f.datarecord_duration
        # epoch_duration = 10 ############################################################################################ for 10 sec

        if psg_f.datarecord_duration == 60: # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
            epoch_duration = epoch_duration / 2
            # epoch_duration = epoch_duration / 6 ######################################################################## for 10 sec
            logger.info("Epoch duration: {} sec (changed from 60 sec)".format(epoch_duration))
        else:
            logger.info("!Epoch duration!: {} sec( 30 sec or 60 sec )".format(epoch_duration))

        # Extract signal from the selected channel
        ch_names = psg_f.getSignalLabels()
        ch_samples = psg_f.getNSamples()
        select_ch_idx1 = -1
        select_ch_idx2 = -1

        for s in range(psg_f.signals_in_file):
            print(ch_names[s])
            #print(select_ch1)
            #print(select_ch2)
            if ch_names[s] == select_ch1:
                select_ch_idx1 = s
                print("Channel1 found: ", select_ch_idx1)
                #break
            if ch_names[s] == select_ch2:
                select_ch_idx2 = s
                print("Channel2 found: ", select_ch_idx2)
                #break

        if select_ch_idx1 == -1:
            raise Exception("Channel1 not found.")
        if select_ch_idx2 == -1:
            raise Exception("Channel2 not found.")

        sampling_rate = psg_f.getSampleFrequency(select_ch_idx1)
        n_epoch_samples = int(epoch_duration * sampling_rate) ## 에포크별(10s) 샘플수

        signals = psg_f.readSignal(select_ch_idx1).reshape(-1, n_epoch_samples) ## epoch별로 자름
        logger.info("Select channel: {}".format(select_ch1))
        logger.info("Select channel samples: {}".format(ch_samples[select_ch_idx1]))
        logger.info("Original @ Sample rate: {}".format(sampling_rate))

        signals2 = psg_f.readSignal(select_ch_idx2).reshape(-1, n_epoch_samples)## epoch별로 자름
        logger.info("Select channel: {}".format(select_ch2))
        logger.info("Select channel samples: {}".format(ch_samples[select_ch_idx2]))
        logger.info("Sample rate: {}".format(sampling_rate))




        # Sanity check
        n_epochs = psg_f.datarecords_in_file
        if psg_f.datarecord_duration == 60: # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
            n_epochs = n_epochs * 2
        assert len(signals) == n_epochs, f"signal: {signals.shape} != {n_epochs}"

        # Generate labels from onset and duration annotation
        labels = []
        labels_id = [] ################################################################################################# for identification
        total_duration = 0
        ann_onsets, ann_durations, ann_stages = ann_f.readAnnotations()

        assert (len(ann_onsets) == len(ann_durations) == len(ann_stages))


        #####delte light off lable

        temp_ann_stages = ann_stages
        temp_ann_onsets = ann_onsets
        temp_ann_durations = ann_durations
        Delete_labels = []

        for a in range(len(ann_stages)):
            # print(a)
            if not (
                    ann_stages[a] == ("Sleep stage W") or
                    ann_stages[a] == ("Sleep stage N1") or
                    ann_stages[a] == ("Sleep stage N2") or
                    ann_stages[a] == ("Sleep stage N3") or
                    ann_stages[a] == ("Sleep stage N4") or
                    ann_stages[a] == ("Sleep stage R") or
                    ann_stages[a] == ("Sleep stage ?") or
                    ann_stages[a] == ("Sleep stage Movement time")
            ):
                Delete_labels.append(a)
                print("Delete need!!!!!!!!", a)
                print(ann_stages[a])
        print(Delete_labels, "Delete_lable!!!!!!!!!!!!!!!!!!!!!!!!!")
        ann_stages = np.delete(temp_ann_stages,Delete_labels)
        ann_onsets = np.delete(temp_ann_onsets, Delete_labels)
        ann_durations = np.delete(temp_ann_durations, Delete_labels)
        print(len(temp_ann_stages), "Original length!!!!!!!!!!!!!!!!!!!!")
        print(len(ann_stages), "Deleted length!!!!!!!!!!!!!!!!!!!!")

        #####


        for a in range(len(ann_stages)):
            onset_sec = int(ann_onsets[a])
            duration_sec = int(ann_durations[a])
            ann_str = "".join(ann_stages[a])

            # Sanity check
            assert onset_sec == total_duration

            # Get label value
            label = ann2label[ann_str]
            label_id = int(i) ######################################################################################## for id

            # Compute # of epoch for this stage
            if duration_sec % epoch_duration != 0:
                logger.info(f"Something wrong: {duration_sec} {epoch_duration}")
                raise Exception(f"Something wrong: {duration_sec} {epoch_duration}")
            duration_epoch = int(duration_sec / epoch_duration)

            # Generate sleep stage labels
            label_epoch = np.ones(duration_epoch, dtype=np.int) * label
            labels.append(label_epoch)

            label_epoch_id = np.ones(duration_epoch, dtype=np.int) * label_id########################################### for id
            labels_id.append(label_epoch_id) ########################################################################### for id


            total_duration += duration_sec

            logger.info("Include onset:{}, duration:{}, label:{} ({})".format(
                onset_sec, duration_sec, label, ann_str
            ))
        labels = np.hstack(labels)
        labels_id = np.hstack(labels_id) ############################################################################### for id

        # Remove annotations that are longer than the recorded signals
        labels = labels[:len(signals)]
        labels_id = labels_id[:len(signals)] ########################################################################### for id

        # Get epochs and their corresponding labels
        x = signals.astype(np.float32)
        x2 = signals2.astype(np.float32)################################################################################

        y = labels.astype(np.int32)
        y2 = labels_id.astype(np.int32) ################################################################################ for id

        # Select only sleep periods
        w_edge_mins = 30
        nw_idx = np.where(y != stage_dict["W"])[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1
        select_idx = np.arange(start_idx, end_idx+1)
        logger.info("Data before selection: {}, {}".format(x.shape, y.shape))
        x = x[select_idx]
        x2 = x2[select_idx]#############################################################################################
        y = y[select_idx]
        y2 = y2[select_idx] ############################################################################################for identification
        logger.info("Data after selection: {}, {}".format(x.shape, y.shape))

        assert ( (x.shape[0]) % 30) ==0#################################################################################
        # assert ((x.shape[0]) % 10) == 0  ###############################################################################for 10 sec

        print(( (x.shape[0]) % 30), " ==> 30seccond * x")
        # print(((x.shape[0]) % 10), " ==> 10seccond * x")################################################################for 10 sec
        print(x.shape[0])

        # Remove movement and unknown
        move_idx = np.where(y == stage_dict["MOVE"])[0]
        unk_idx = np.where(y == stage_dict["UNK"])[0]
        if len(move_idx) > 0 or len(unk_idx) > 0:
            remove_idx = np.union1d(move_idx, unk_idx)
            logger.info("Remove irrelavant stages")
            logger.info("  Movement: ({}) {}".format(len(move_idx), move_idx))
            logger.info("  Unknown: ({}) {}".format(len(unk_idx), unk_idx))
            logger.info("  Remove: ({}) {}".format(len(remove_idx), remove_idx))
            logger.info("  Data before removal: {}, {}".format(x.shape, y.shape))
            select_idx = np.setdiff1d(np.arange(len(x)), remove_idx)
            x = x[select_idx]
            x2 = x2[select_idx]  ################################################################################################
            y = y[select_idx]
            y2 = y2[select_idx] ########################################################################################for identification

            logger.info("  Data after removal: {}, {}".format(x.shape, y.shape))

#########################################################################################################################
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~shampe")
        print(x.shape)

        x_temp = np.transpose(x)
        print(x_temp)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~transposed shampe")
        print(x_temp.shape)

        x_temp_resample = signal.resample(x_temp, 100) ##########################################################################################################for 10 sec, resampling
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~resampled shampe")
        print(x_temp_resample.shape)

        x_temp_resample_trans = np.transpose(x_temp_resample)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~resampled trans shape")
        print(x_temp_resample_trans.shape)

        #x = x_temp_resample_trans

        # x = np.reshape(x_temp_resample_trans,(-1,3000)) ##########################################################################################################for 10 sec
        # x = np.reshape(x_temp_resample_trans,(-1,1000))  ##########################################################################################################for 10 sec
        x = np.reshape(x_temp_resample_trans, (-1, 500))  ###################################################################################################for 5 sec

        ########################################################################################################################
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~x2 shape")
        print(x2.shape)

        x2_temp = np.transpose(x2)
        print(x2_temp)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~transposed shape")
        print(x2_temp.shape)

        x2_temp_resample = signal.resample(x2_temp, 100)##########################################################################################################for 10 sec, resampling
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~resampled shape")
        print(x2_temp_resample.shape)

        x2_temp_resample_trans = np.transpose(x2_temp_resample)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~resampled trans shape")
        print(x2_temp_resample_trans.shape)

        # x2 = x2_temp_resample_trans

        # x2 = np.reshape(x2_temp_resample_trans, (-1, 3000)) ##########################################################################################################for 10 sec
        # x2 = np.reshape(x2_temp_resample_trans, (-1, 1000)) ##########################################################################################################for 10 sec
        x2 = np.reshape(x2_temp_resample_trans, (-1, 500))  ##########################################################################################################for 5 sec
        ########################################################################################################################


########################################################################################################################
        y_temp =[]
        y_temp_id =[]#################################################################################################### for id


        '''
        for idx in range(0, (int)(len(y)), 30):
            #     assert
            #print((int)(np.mean(y[idx:idx + 30])))
            y_temp.append( (int)(np.mean(y[idx:idx + 30])) )
            y_temp_id.append( (int)(np.mean(y2[idx:idx + 30])) ) ####################################################### for id
        '''

        '''
        ######################################################################################################################################## for 10sec
        for idx in range(0, (int)(len(y)), 10):
            #     assert
            #print((int)(np.mean(y[idx:idx + 30])))
            y_temp.append( (int)(np.mean(y[idx:idx + 10])) )
            y_temp_id.append( (int)(np.mean(y2[idx:idx + 10])) ) ####################################################### for id
        ######################################################################################################################################## for 10sec
        '''

        ######################################################################################################################################## for 5sec
        for idx in range(0, (int)(len(y)), 5):
            #     assert
            # print((int)(np.mean(y[idx:idx + 30])))
            y_temp.append((int)(np.mean(y[idx:idx + 5])))
            y_temp_id.append((int)(np.mean(y2[idx:idx + 5])))  ####################################################### for id
        ######################################################################################################################################## for 5sec


        y = y_temp
        y2 = y_temp_id ################################################################################################# for id

########################################################################################################################

        sampling_rate = 100 ######################################################################################
        print("!!!sampling frequency is changed to :", sampling_rate)

        # epoch_duration = 30#######################################################################################
        # epoch_duration = 10  ##################################################################################################################for 10sec
        epoch_duration = 5  ##################################################################################################################for 5sec

        print("!!!epoch_duration is changed to :", epoch_duration)


        # Save
        filename = ntpath.basename(psg_fnames[i]).replace(".edf", ".npz")
        save_dict = {
            "x": x,  ################################################################### Channel 1
            "x2": x2,################################################################### Channel 2
            "y": y,  ################################################################### label
            "y2" : y2, ################################################################################################# for id
            "fs": sampling_rate, ####################################################### Sampling rate
            "ch_label1": select_ch1, #################################################### Channel ID 1
            "ch_label2": select_ch2,#################################################### Channel ID 2
            "start_datetime": start_datetime,
            "file_duration": file_duration,
            "epoch_duration": epoch_duration,########################################### 30 or 60 sec
            "n_all_epochs": n_epochs, ################################################## all epochs
            "n_epochs": len(x), ######################################################## epochs
        }
        ##np.savez(os.path.join(args.output_dir, filename), **save_dict) ## original
        ##print(filename) ## original

        logger.info("\n=======================================\n")

        id_num_interval = int(len(x)/10) # devide it every per 10%
        for id_num in range(0, 10) : ## not to use 10th data for testing
            filename = ntpath.basename(psg_fnames[i]).replace(".edf", "{}.npz".format(id_num))
            logger.info("  interval : {}, id_num : {}, filename : {} ".format(id_num_interval, id_num, filename))
            save_dict = {
                "x": x[ id_num*id_num_interval : (id_num+1)*id_num_interval],   ################################################################### Channel 1
                "x2": x2[ id_num*id_num_interval : (id_num+1)*id_num_interval], ################################################################### Channel 2
                "y": y[ id_num*id_num_interval : (id_num+1)*id_num_interval],   ################################################################### label
                "y2": y2[ id_num*id_num_interval : (id_num+1)*id_num_interval], ################################################################################################# for id
                "fs": sampling_rate,  ####################################################### Sampling rate
                "ch_label1": select_ch1,  #################################################### Channel ID 1
                "ch_label2": select_ch2,  #################################################### Channel ID 2
                "start_datetime": start_datetime,
                "file_duration": file_duration,
                "epoch_duration": epoch_duration,  ########################################### 30 or 60 sec
                "n_all_epochs": n_epochs,  ################################################## all epochs
                "n_epochs": len(x[ id_num*id_num_interval : (id_num+1)*id_num_interval]),  ######################################################## epochs
            }
            np.savez(os.path.join(args.output_dir, filename), **save_dict) ########################################### need to be activated when using training
            print(filename)                                                ########################################### need to be activated when using training

            logger.info("\n============Dividing TRAINING DATA==========================\n")


        filename_test = ntpath.basename(psg_fnames[i]).replace(".edf", ".npz")
        logger.info(" filename : {} ".format(filename))
        save_dict_test = {
            "x": x[-250 : -200],
            ################################################################### Channel 1
            "x2": x2[-250 : -200],
            ################################################################### Channel 2
            "y": y[-250 : -200],
            ################################################################### label
            "y2": y2[-250 : -200],
            ################################################################################################# for id
            "fs": sampling_rate,  ####################################################### Sampling rate
            "ch_label1": select_ch1,  #################################################### Channel ID 1
            "ch_label2": select_ch2,  #################################################### Channel ID 2
            "start_datetime": start_datetime,
            "file_duration": file_duration,
            "epoch_duration": epoch_duration,  ########################################### 30 or 60 sec
            "n_all_epochs": n_epochs,  ################################################## all epochs
            # "n_epochs": len(x[id_num * id_num_interval: (id_num + 1) * id_num_interval]),
            ######################################################## epochs
        }
        # np.savez(os.path.join(args.output_dir, filename_test), **save_dict_test) ####################################### need to be activated when using testing
        # print(filename_test)                                                     ####################################### need to be activated when using testing

        # logger.info("\n===============TEST DATA ====================\n")


if __name__ == "__main__":
    main()

