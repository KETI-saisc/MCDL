# -*- coding:utf-8 -*-

import argparse
import glob
import importlib
import os


#os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async' ################################
#print(os.getenv('TF_GPU_ALLOCATOR'))

import numpy as np
import shutil
import sklearn.metrics as skmetrics
import tensorflow as tf

from data import load_data,load_data_ecg, load_data_bcg, get_subject_files, get_subject_files_sleepmat, get_subject_files_sleephmc
from model import TinySleepNet
from minibatching import (iterate_minibatches,
                          iterate_batch_seq_minibatches,
                          iterate_batch_multiple_seq_minibatches)
from utils import (get_balance_class_oversample,
                   print_n_samples_each_class,
                   save_seq_ids,
                   load_seq_ids)
from logger import get_logger

import time

def compute_performance(cm):
    """Computer performance metrics from confusion matrix.

    It computers performance metrics from confusion matrix.
    It returns:
        - Total number of samples
        - Number of samples in each class
        - Accuracy
        - Macro-F1 score
        - Per-class precision
        - Per-class recall
        - Per-class f1-score
    """

    tp = np.diagonal(cm).astype(np.float)
    tpfp = np.sum(cm, axis=0).astype(np.float) # sum of each col
    tpfn = np.sum(cm, axis=1).astype(np.float) # sum of each row
    acc = np.sum(tp) / np.sum(cm)
    precision = tp / tpfp
    recall = tp / tpfn
    f1 = (2 * precision * recall) / (precision + recall)
    mf1 = np.mean(f1)

    total = np.sum(cm)
    n_each_class = tpfn

    return total, n_each_class, acc, mf1, precision, recall, f1


def predict(
    config_file,
    model_dir,
    output_dir,
    log_file,
    use_best=True,
    official_test_path="./data/sleephmc/5ppl_official_test",########################modified
):
    print(time.strftime('-3 :' + '%H-%M-%S'))
    print(official_test_path,"~~~~~!!!!!!!!!!!!!!!!!!!!!!!!!!")##############check
    spec = importlib.util.spec_from_file_location("*", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    config = config.predict

    # Create output directory for the specified fold_idx
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create logger
    logger = get_logger(log_file, level="info")

    print(time.strftime('-2 :' + '%H-%M-%S'))

    #subject_files = glob.glob(os.path.join(config["data_dir"], "*.npz"))######## original
    subject_files = glob.glob(os.path.join(official_test_path, "*.npz")) ######### modified
    # print(subject_files)

    # Load subject IDs
    fname = "{}.txt".format(config["dataset"])
    seq_sids = load_seq_ids(fname)
    #logger.info("Load generated SIDs from {}".format(fname))#######################################################################################################original
    #logger.info("SIDs ({}): {}".format(len(seq_sids), seq_sids))###################################################################################################original

    # Split training and test sets
    fold_pids = np.array_split(seq_sids, config["n_folds"])

    # Add dummy class weights
    config["class_weights"] = np.ones(config["n_classes"], dtype=np.float32)
    print(config["class_weights"])

    trues = []
    preds = []
    preds_soft = []  ####################################################################### modified
    F1_Score = -1
    F1_Score_sleep = -1

    s_trues = []
    s_preds = []

    print(time.strftime('-1 :' + '%H-%M-%S'))
    for fold_idx in range(config["n_folds"]):

        #logger.info("------ Fold {}/{} ------".format(fold_idx+1, config["n_folds"]))##############################################################################original
        logger.info("------ Processing...{}/{} -------".format(fold_idx + 1, config["n_folds"]))#####################################################################original

        #test_sids = fold_pids[fold_idx]
        #logger.info("Test SIDs: ({}) {}".format(len(test_sids), test_sids))

        model = TinySleepNet(
            config=config,
            output_dir=os.path.join(model_dir, str(fold_idx)),
            # use_rnn=True, ## original
            use_rnn=False, ###########################################################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! rnn 미사용 ## for id
            testing=True,
            use_best=use_best,
        )


        # # Get corresponding files
        # s_trues = []
        # s_preds = []

        #for sid in test_sids:
        #logger.info("Subject ID: {}".format(fold_idx))##############################################################################################################original

        print(time.strftime('0 :' + '%H-%M-%S'))
        #test_files = get_subject_files(
        # test_files = get_subject_files_sleepmat( ### 폴더내 모든 npz 파일 로드
        test_files=get_subject_files_sleephmc(  ### 폴더내 모든 npz 파일 로드

            dataset=config["dataset"],
            files=subject_files,
            sid=fold_idx, # no_use
        )
        print("test files~~:")
        print(test_files)


        print(time.strftime('1 :' + '%H-%M-%S'))

        #for vf in test_files: logger.info("Load files {} ...".format(vf))#############################################################################################original
        for vf in test_files: logger.info("Load files {}/{} ----------------".format((vf), len(test_files) ))###########################################################original

        # test_x, test_y, _ = load_data(test_files)##################################original ##########model에 맞게 수정필요
        # test_x, test_y, _ = load_data2(test_files) ## 수정 -> 2개의 특징 이용할때
        # test_x, test_y, _ = load_data3(test_files) ## 수정 -> 3개의 특징 이용할때
        # test_x, test_y, _ = load_data4(test_files) ## 수정 -> 4개의 특징 이용할때
        # test_x, test_y, _ = load_data_bcg(test_files)  ## 수정 -> bcg만 특징 이용할때
        test_x, test_y, _ = load_data_ecg(test_files)  ## 수정 -> ecg만 특징 이용할때####################################### for id
        # test_x, test_y, _ = load_data_multi13(test_files)  ## 수정 -> 13개의 특징 이용할때

        print(time.strftime('2 :' + '%H-%M-%S'))



        ## Print test set
        #logger.info("Test set (n_night_sleeps={})".format(len(test_y)))###############################################################################################original
        #for _x in test_x: logger.info(_x.shape)#######################################################################################################################original
        #print_n_samples_each_class(np.hstack(test_y))#################################################################################################################original

        for night_idx, night_data in enumerate(zip(test_x, test_y)):
            # Create minibatches for testing
            night_x, night_y = night_data
            test_minibatch_fn = iterate_batch_multiple_seq_minibatches(
                [night_x],
                [night_y],
                batch_size=config["batch_size"],
                seq_length=config["seq_length"],
                shuffle_idx=None,
                augment_seq=False,
            )
            print(time.strftime('2-1 :' + '%H-%M-%S'))
            if (config.get('augment_signal') is not None) and config['augment_signal']: ### False
                # Evaluate
                test_outs = model.evaluate_aug(test_minibatch_fn)

            else: ### True
                # Evaluate
                print(time.strftime('2-1-1 :' + '%H-%M-%S'))
                test_outs = model.evaluate(test_minibatch_fn)
                # print("!!!!!!~~~~")
                # print(model.logits)
                ##print("!!!!!!")
                ##print(test_outs)
                # prob_outs = model.predict(test_minibatch_fn)
                # print("!!!!!!")
                # print(prob_outs)

            print(time.strftime('2-2 :' + '%H-%M-%S'))

            s_trues.extend(test_outs["test/trues"])
            s_preds.extend(test_outs["test/preds"])
            trues.extend(test_outs["test/trues"])
            preds.extend(test_outs["test/preds"])
            preds_soft.extend(test_outs["test/softmax"])  ###################################

            # Save labels and predictions (each night of each subject)
            save_dict = {
                "y_true": test_outs["test/trues"],
                "y_pred": test_outs["test/preds"],
                "total_soft": preds_soft,####################################################
            }
            fname = os.path.basename(test_files[night_idx]).split(".")[0]
            save_path = os.path.join(
                output_dir,
                "pred_{}.npz".format(fname)
            )
            #np.savez(save_path, **save_dict)#######################################################################################################################original
            #logger.info("Saved outputs to {}".format(save_path))###################################################################################################original

        print(time.strftime('3 :' + '%H-%M-%S'))

        # print("s_trues :", s_trues)
        # print("s_preds :", s_preds)
        s_acc = skmetrics.accuracy_score(y_true=s_trues, y_pred=s_preds)
        s_f1_score = skmetrics.f1_score(y_true=s_trues, y_pred=s_preds, average="macro")

        # s_cm = skmetrics.confusion_matrix(y_true=s_trues, y_pred=s_preds, labels=[0,1,2,3,4]) ## original ############################################################
        s_cm = skmetrics.confusion_matrix(y_true=s_trues, y_pred=s_preds, labels= range(config["n_classes"]))#####################for id

        save_dict = {
            # "y_true": test_outs["test/trues"], # original
            "y_true": s_trues,
            # "y_pred": test_outs["test/preds"], # original
            "y_pred": s_preds,
            "total_soft": preds_soft,  ####################################################
            "F1_Score": s_f1_score * 100,
            "Accuracy": s_acc * 100,
            "Confusion_Matrix(row:g_true, Col:SM)": s_cm,
        }
        fname = os.path.basename(test_files[night_idx]).split(".")[0]
        save_path = os.path.join(
            output_dir,
            "predb_{}.npz".format(fname + "_foldID_" + str(fold_idx))
            ###################################################################
        )
        np.savez(save_path, **save_dict)

#######################################save_b_NPZ######################################################################
        # if( F1_Score < s_f1_score) :
        #     F1_Score = s_f1_score
        #     # Save labels and predictions (each night of each subject)
        #     save_dict = {
        #         # "y_true": test_outs["test/trues"], # original
        #         "y_true": s_trues,
        #         # "y_pred": test_outs["test/preds"], # original
        #         "y_pred": s_preds,
        #         "total_soft": preds_soft,  ####################################################
        #         "F1_Score" : s_f1_score*100,
        #         "Accuracy" : s_acc*100,
        #         "Confusion_Matrix(row:g_true, Col:SM)" : s_cm,
        #     }
        #     fname = os.path.basename(test_files[night_idx]).split(".")[0]
        #     save_path = os.path.join(
        #         output_dir,
        #         "predb_{}.npz".format(fname +"_foldID_"+ str(fold_idx)) ###################################################################
        #     )
        #     np.savez(save_path, **save_dict)
        #     if(s_f1_score>0.7) :
        #         logger.info("*****************************")

######################################################################################################################################################################original
###
        # logger.info("n={}, acc={:.1f}, mf1={:.1f}".format(
        #     len(s_preds),
        #     s_acc*100.0,
        #     s_f1_score*100.0,
        #     ##preds_soft, ##################### 시간 오래걸려 주석처리
        # ))
        #
        # logger.info(">> Confusion Matrix")
        # logger.info(s_cm)
        # ################################################################################################################
        # save_path_total_cm = os.path.join(
        #     output_dir,
        #     "Confusion_total_{}.npz".format(fold_idx)
        # )
        # #np.savez(save_path_total_cm, s_cm)############################################################################################################################
###
######################################################################################################################################################################original
        tf.reset_default_graph()
        s_trues = []  ############################################################### initialize
        s_preds = []  ############################################################### initialize
        preds_soft = []  ############################################################ initialize

        ###############################################################################################################
        save_dict_total = {
            "y_true": trues,
            "y_pred": preds,
        }
        save_path_total = os.path.join(
            output_dir,
            "pred_total.npz"
        )
        #np.savez(save_path_total, **save_dict_total)##################################################################################################################original
        ####################################################
        tf.reset_default_graph()

        logger.info("----------------------------------")
        logger.info("")
	

	

    ########################################################################################################

    ####################################################################################################################################################################original
    # acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
    # f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
    # cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0,1,2,3,4])
    #
    # logger.info("")
    # logger.info("=== Overall ===")
    # print_n_samples_each_class(trues)
    # logger.info("n={}, acc={:.1f}, mf1={:.1f}".format(
    #     len(preds),
    #     acc*100.0,
    #     f1_score*100.0,
    # ))
    #
    # logger.info(">> Confusion Matrix")
    # logger.info(cm)
    #
    # metrics = compute_performance(cm=cm)
    # logger.info("Total: {}".format(metrics[0]))
    # logger.info("Number of samples from each class: {}".format(metrics[1]))
    # logger.info("Accuracy: {:.1f}".format(metrics[2]*100.0))
    # logger.info("Macro F1-Score: {:.1f}".format(metrics[3]*100.0))
    # logger.info("Per-class Precision: " + " ".join(["{:.1f}".format(m*100.0) for m in metrics[4]]))
    # logger.info("Per-class Recall: " + " ".join(["{:.1f}".format(m*100.0) for m in metrics[5]]))
    # logger.info("Per-class F1-Score: " + " ".join(["{:.1f}".format(m*100.0) for m in metrics[6]]))
    #
    # # Save labels and predictions (all)
    # save_dict = {
    #     "y_true": trues,
    #     "y_pred": preds,
    #     "seq_sids": seq_sids,
    #     "config": config,
    # }
    # save_path = os.path.join(
    #     output_dir,
    #     "{}.npz".format(config["dataset"])
    # )
    # np.savez(save_path, **save_dict)
    # logger.info("Saved summary to {}".format(save_path))
    ####################################################################################################################################################################original


if __name__ == "__main__":

    print(time.strftime('-5 :' + '%H-%M-%S'))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="./out_sleepedf/finetune")
    parser.add_argument("--output_dir", type=str, default="./output/predict")
    parser.add_argument("--log_file", type=str, default="./output/output.log")
    parser.add_argument("--use-best", dest="use_best", action="store_true")
    parser.add_argument("--no-use-best", dest="use_best", action="store_false")
    parser.add_argument("--official_test_path", type=str, default="./data/sleephmc/5ppl_official_test") #######
    parser.set_defaults(use_best=False)
    args = parser.parse_args()

    print(time.strftime('-4 :' + '%H-%M-%S'))
    while(1):
                    			
	    predict(
		config_file=args.config_file,
		model_dir=args.model_dir,
		output_dir=args.output_dir,
		log_file=args.log_file,
		use_best=args.use_best,
		official_test_path=args.official_test_path,#######################################################
	    )

