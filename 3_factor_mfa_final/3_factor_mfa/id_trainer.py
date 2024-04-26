import argparse
import importlib
import os
import tensorflow as tf

from id_train import train  ## import id_train !!!!


def run(db, gpu, from_fold, to_fold, suffix='', random_seed=42):
    # Set GPU visible to Tensorflow
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Config file
    config_file = os.path.join('config', f'{db}.py')
    spec = importlib.util.spec_from_file_location("*", config_file) # config 파일 불러오기
    config = importlib.util.module_from_spec(spec) #
    spec.loader.exec_module(config)

    # Output directory
    output_dir = f'out_{db}{suffix}' # 출력 폴더명

    assert from_fold <= to_fold
    assert to_fold < config.params['n_folds']

    # Training
    for fold_idx in range(from_fold, to_fold+1):
        train(
            config_file=config_file,
            fold_idx=fold_idx,
            output_dir=os.path.join(output_dir, 'train'), # output train폴더
            log_file=os.path.join(output_dir, f'train_{gpu}.log'), # gpu 번로로 로그 생성
            restart=True,
            random_seed=random_seed+fold_idx, # train 섞기 위해 특정 램덤변수 생성
        )

        # Reset tensorflow graph
        tf.reset_default_graph()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--from_fold", type=int, required=True)
    parser.add_argument("--to_fold", type=int, required=True)
    parser.add_argument("--suffix", type=str, default='')
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    run(
        db=args.db,
        gpu=args.gpu,
        from_fold=args.from_fold,
        to_fold=args.to_fold,
        suffix=args.suffix,
        random_seed=args.random_seed,
    )
