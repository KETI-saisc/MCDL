import csv
import os

ECG_id = input('\n enter user id ==>  ')

def csv_range_to_txt(input_folder, output_folder):
    # 1부터 500까지의 파일 변환

    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    print(csv_files)
    for csv_file in csv_files:
        # CSV 파일 경로
        csv_file_path = os.path.join(input_folder, csv_file)

        # 텍스트 파일 경로
        txt_file = os.path.splitext(csv_file)[0] + '.txt'
        txt_file_path = os.path.join(output_folder, txt_file)

        os.makedirs(output_folder, exist_ok=True)

        with open(csv_file_path, 'r') as file:
            csv_data = csv.reader(file)
            with open(txt_file_path, 'w') as txt_file:
                for row in csv_data:
                    txt_file.write(','.join(row) + '\n')
                    print(txt_file)
# 예시로 'input' 폴더에 있는 1부터 500까지의 CSV 파일들을 'output' 폴더에 텍스트 파일로 변환하여 저장합니다.
csv_range_to_txt('csv/' + ECG_id + '/', 'txt/' + ECG_id + '/')
