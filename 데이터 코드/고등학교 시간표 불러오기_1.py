from urllib.request import urlopen
import pandas as pd
import json
import data_function as dataf
import os

df = pd.read_csv("데이터/고등학교기본정보.csv")
print(df.head())

school_list = df['행정표준코드']
print(school_list)
all_school_data = []
count = 1 
error_data=[]


start_index = 175
end_index = 320  

for index, row in df.iterrows():
    if start_index <= index <= end_index:
        school_type = row['학교종류명']
        if school_type == "고등학교":
            region_code = row['시도교육청코드']
            school_code = row['행정표준코드']
            school_name = row['학교명']
            year = 2022 
            sem = 1      

            first_file_path = dataf.download_timetable(region_code, school_code, school_name, year, sem)
            if first_file_path != None:
                print(f"{school_name} 교과목 정리 및 카운트 실행")
                school_data = dataf.save_count_data(first_file_path)
                all_school_data += school_data
                os.remove(first_file_path)

                output_file_path = f'데이터/{year}년도_{sem}학기_학교별 개설 교과목_4.json'
                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    json.dump(all_school_data, output_file, ensure_ascii=False, indent=4)

                print(f"{count}개의 학교 데이터가 병합되어", output_file_path, "에 저장되었습니다.")
                count += 1
            else:
                error_data.append(school_code)
                print(error_data)
                print("문제 발생 학교를 제외")





