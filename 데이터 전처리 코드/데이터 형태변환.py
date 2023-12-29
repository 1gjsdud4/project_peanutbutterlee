import csv
import json

# 입력 파일과 출력 파일 경로 지정
input_file_path = '데이터/클러스터일 최종_결과.json'
output_file_path = '데이터/클러스터링 최종_결과_summary.csv'

# JSON 파일 읽어오기
with open(input_file_path, 'r', encoding='utf-8') as input_file:
    data = json.load(input_file)

# CSV 파일에 저장할 필드 선택
fields_to_save = [
    "ATPT_OFCDC_SC_CODE",
    "ATPT_OFCDC_SC_NM",
    "SD_SCHUL_CODE",
    "AY",
    "Latitude",
    "Longitude",
    "고등학교구분명",
    "location_Cluster",
    "sub_Cluster",
    "final_Cluster"
]

# 결과를 CSV 파일에 쓰기
with open(output_file_path, 'w', encoding='utf-8', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fields_to_save)
    
    # CSV 파일의 헤더 작성
    writer.writeheader()
    
    # 각 문서에서 필요한 필드만 선택하여 CSV 파일에 작성
    for doc in data:
        row_data = {field: doc.get(field, '') for field in fields_to_save}
        writer.writerow(row_data)

print(f'Summary CSV file created at: {output_file_path}')
