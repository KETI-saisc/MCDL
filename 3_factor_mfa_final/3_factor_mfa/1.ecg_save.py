import pandas as pd
import serial
import natsort
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import time
import datetime
import os
import csv

global sam_freq, save_sec, iterated
sam_freq = 100  # Hz
save_sec = 5    # 100 Hz * 5 sec (per one line)
iterated = 6   # 100 Hz * 5 sec * 6 lines (TOTAL DATA = 30 sec)
#iterated = 360   # 100 Hz * 5 sec * 360 lines (TOTAL DATA = 30 min)
#iterated = 1080   # 100 Hz * 5 sec * 3 lines (TOTAL DATA = 15 sec)

txt_dir = 'data/ecg/txt/'
csv_dir = 'data/ecg/csv/'
npz_dir = 'data/ecg/npz/'
npz_hpf_dir = 'data/ecg/npz_hpf/'
txt_hpf_dir = 'data/ecg/txt_hpf/'
npz_split_dir = 'data/ecg/npz_split/'
ECG_id = input('\n enter user id ==>  ')
ecg_ser = 'ttyUSB0'


def ecg_get_data_init(tty):
    ser = serial.Serial(
            port='/dev/' + tty,
            baudrate = 115200,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=1
           )
    return ser


def save_ecg2txt(save_dir, ECG_id):

    save_dir = save_dir + ECG_id + '/'

    try:
        os.makedirs(save_dir)
    except OSError:
        if not os.path.isdir(save_dir):
            raise

    ecg_list_save = []
    number = 0

    while 1:
        line = ser.readline().decode('unicode_escape')

        if line == '\n':
            pass

        else:
            data = line.strip().split(',')
            ecg_list_save += data
            #print (data)
            if len(ecg_list_save) == (sam_freq * save_sec):
                f_name = "ecg_data_" + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) + ".txt"
                f = open(save_dir + f_name, 'w')
                for idx, val in enumerate(ecg_list_save):
                    f.write(str(val) + ",")
                f.close()

                ecg_list_save = []
                number += 1
                print("["+str(number)+'/'+str(iterated)+"] Saved ECG data as", f_name)

                if number == iterated:
                    break

            else:
                pass



# save ECG data as [.csv] files
def save_ecg2csv(save_dir, ECG_id):

    number = 0
    save_dir = save_dir + ECG_id + '/'
    data_ECG = pd.DataFrame(columns=['ECG'])

    try:
        os.makedirs(save_dir)
    except OSError:
        if not os.path.isdir(save_dir):
            raise

    # delete past csv files
    for file in os.listdir(save_dir):
        if file.endswith('.csv'):
            os.remove(save_dir + file)

    global timebuf
    timebuf = 0

    while True:

        line = ser.readline().decode('unicode_escape')

        if line == '\n':
            pass

        else:
            if timebuf == 0:
                pass
            else:
                print (datetime.datetime.now() - timebuf)

            data = line.strip().split(',')
            data_ECG = pd.concat([data_ECG, pd.DataFrame([data], columns=data_ECG.columns)], ignore_index=True)

            timebuf = datetime.datetime.now()

            if len(data_ECG) == (sam_freq * save_sec):
                time_str = time.strftime("%H%M%S")
                filename = str(ECG_id) + '-' + time_str + ".csv"
                data_ECG_t = data_ECG.transpose()

                # fixed 23.05.30 by jiseong (Before)
                # data_ECG_t.to_csv(save_dir + filename , sep=',', index=False, header=None)
                # fixed 23.05.30 by jiseong (After)
                data_ECG_t.to_csv(save_dir + filename , sep=',', index=False)

                number += 1
                data_ECG = pd.DataFrame(columns=['ECG'])
                print("["+str(number)+'/'+str(iterated)+"] Saved ECG data as", filename)

            if number == iterated:
                break


def trans_txt2npz(target_dir, save_dir, ECG_id):

    target_dir = target_dir + ECG_id + '/'
    save_dir = save_dir + ECG_id + '/'

    try:
      os.makedirs(save_dir)
    except OSError:
        if not os.path.isdir(save_dir):
            raise

    df = os.listdir(target_dir)
    csv_files = natsort.natsorted(df)

    ecg_data = np.empty((0,sam_freq*save_sec), int)

    for idx, file in enumerate(csv_files):
        print (file)
        f = open(target_dir + file, 'r', encoding='UTF8')
        data = f.read()
        ecg_list = data.split(",")
        ecg_list = [x for x in ecg_list if x != '']

        print (len(ecg_list))

        # x2
        ecg_np = np.array(list(map(float, ecg_list)))
        ecg_data = np.append(ecg_data, np.array([ecg_np]), axis=0)

        # y2
        label_data = np.full((iterated,), int(ECG_id.replace("s",""))-1)

        # fs
        fs = np.array([[100]], dtype=np.int32)

        save_dict = {
            "x2": ecg_data,                             # piezo channel 1
            "y2": label_data,                             # piezo channel 2
            "fs": fs,                             # piezo channel 3
        }

        filename = str(ECG_id) + "_ECG_full.npz"
        np.savez(save_dir + filename, **save_dict)
        print("saved as", filename)

# data recombination (50 EA [.csv]) to (1 EA [.npz])
def trans_csv2npz(target_dir, save_dir, ECG_id):

    target_dir = target_dir + ECG_id + '/'
    save_dir = save_dir + ECG_id + '/'

    try:
      os.makedirs(save_dir)
    except OSError:
        if not os.path.isdir(save_dir):
            raise

    df = os.listdir(target_dir)
    csv_files = natsort.natsorted(df)

    data = pd.DataFrame()

    for idx, file in enumerate(csv_files):
        PATH = target_dir + file
        add = pd.read_csv(PATH, header = None)
        data = pd.concat([data, add])
        print('[',str(idx+1)+'/'+str(iterated),'] appending file', file, 'to NPZ file..')


    x2 = data.to_numpy()
    y2 = np.full((iterated,), int(ECG_id.replace("s",""))-1)

    # fixed 23.05.30 by jiseong (Before)
    # data.to_csv(save_dir + ECG_id + '-total.csv', sep=',', index=False, header=None)
    # fixed 23.05.30 by jiseong (After)
    data.to_csv(save_dir + ECG_id + '-total.csv', sep=',')
    np.savez(save_dir + ECG_id + '-total.npz' , x2=x2, y2=y2)


def split_npzfile(input_file, output_dir, num_files, num_data):

    output_dir = output_dir + ECG_id + '/'

    try:
        os.makedirs(output_dir)
    except OSError:
        if not os.path.isdir(output_dir):
            raise

#    data = np.load(input_file + ECG_id + '/'+ str(ECG_id)+'-total.npz')['x2']
#    reshaped_data = data.reshape((-1, num_data, 500))[:, :num_data, :]  # Select first num_data columns
#
#    for i in range(num_files):
#        output_file = f"{output_dir}/SN00" + ECG_id[-1] + str(i) + ".npz"
#        np.savez(output_file, x2=reshaped_data[i])

    data_x = np.load(input_file + ECG_id + '/'+ str(ECG_id)+'_ECG_full.npz')['x2']
    #data_y = np.load(input_file + ECG_id + '/'+ str(ECG_id)+'_ECG_full.npz')['y2']
    reshaped_data_x = data_x.reshape((-1, num_data, 500))[:, :num_data, :]  # Select first num_data columns
    #reshaped_data_y = data_y.reshape((-1, num_data, 1))[:, :num_data, :]  # Select first num_data columns
    reshaped_data_y = np.full((num_data,), int(ECG_id.replace("s",""))-1)

    for i in range(num_files):
        output_file = f"{output_dir}/SN00" + ECG_id[-1] + str(i) + ".npz"
        np.savez(output_file, x2=reshaped_data_x[i], y2=reshaped_data_y, fs=np.array([[100]], dtype=np.int32))

def readtxt_HPF(target_dir, save_dir, ECG_id, fs, cutoff):

    save_dir = save_dir + ECG_id + '/'
    target_dir = target_dir + ECG_id + '/'

    try:
        os.makedirs(save_dir)
    except OSError:
        if not os.path.isdir(save_dir):
            raise


    df = os.listdir(target_dir)
    txt_files = natsort.natsorted(df)

    for idx, file in enumerate(txt_files):
        data = np.genfromtxt(target_dir + file, delimiter=',')
        b, a = signal.butter(3, cutoff / (fs / 2), btype='high', analog=False)
        filtered_data = signal.lfilter(b, a, data)
        filtered_data_ = filtered_data[~np.isnan(filtered_data)]
        filtered_data_list = filtered_data_.tolist()

        print(len(filtered_data_list))

        filtered_data_str = str(filtered_data_list).replace("[","")
        filtered_data_str = filtered_data_str.replace("]","")
        filtered_data_str = filtered_data_str.replace(" ","")
        filename = file.replace(".txt","_hpf.txt")

        with open(save_dir + filename, "w") as file:
            file.write(filtered_data_str)  # 문자열을 파일에 작성

        print ("saved as", filename)


        #filename = file.replace(".txt","_hpf.txt")
        #np.savetxt(save_dir + filename, filtered_data_, delimiter=',', fmt='%d')

        #print ("saved as", filename)

        # Drawing Graph
        #if (idx == 0) or (idx == 1):
        #    t = np.arange(len(data)) / fs
        #    plt.figure(figsize=(10, 4))
        #    plt.plot(t, data, label='Original Data')
        #    plt.plot(t, filtered_data, label='Filtered Data')
        #    plt.xlabel('Time (s)')
        #    plt.ylabel('Amplitude')
        #    plt.title('Filtered ECG Data')
        #    plt.legend()
        #    plt.show()


# MAIN #
ser = ecg_get_data_init(ecg_ser)
save_ecg2txt(txt_dir, ECG_id)
readtxt_HPF(txt_dir, txt_hpf_dir, ECG_id, sam_freq, cutoff=0.5)
#trans_txt2npz(txt_dir, npz_dir, ECG_id)
#trans_txt2npz(txt_hpf_dir, npz_hpf_dir, ECG_id)
#split_npzfile(npz_dir, npz_split_dir, 10, 108)

#save_ecg2csv(csv_dir, ECG_id)
#trans_csv2npz(csv_dir, npz_dir, ECG_id)
