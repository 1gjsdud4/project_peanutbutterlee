import json
import os

# 병합할 JSON 파일들이 있는 디렉토리
directory = 'C:/Users/POWER/Desktop/project_peanutbutterlee/데이터/고교개설 교과몸'

# 병합될 데이터를 저장할 리스트
merged_data = []

file_path = ['C:/Users/POWER/Desktop/project_peanutbutterlee/데이터/2023년도_1학기_all_school_data.json','C:/Users/POWER/Desktop/project_peanutbutterlee/데이터/2023년도_1학기_all_school_data_2.json','C:/Users/POWER/Desktop/project_peanutbutterlee/데이터/2023년도_1학기_all_school_data_3.json','C:/Users/POWER/Desktop/project_peanutbutterlee/데이터/2023년도_1학기_all_school_data_4.json']
for file_path in file_path:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        merged_data.extend(data)

print(len(merged_data))
# 모든 데이터를 하나의 JSON 파일로 병합하여 저장
output_file_path = 'C:/Users/POWER/Desktop/project_peanutbutterlee/데이터/2023년도_1학기_학교별 개설 교과목.json'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(merged_data, output_file, ensure_ascii=False, indent=4)
