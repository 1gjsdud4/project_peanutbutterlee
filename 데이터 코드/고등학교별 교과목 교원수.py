import csv
import json

# CSV 파일 경로
csv_file_path = "C:/Users/POWER/Desktop/project_peanutbutterlee/데이터/2023년도_표시과목별 교원 현황(교과별)(고)_서울특별시교육청.csv"

# 각 학교의 데이터를 담을 딕셔너리 초기화
school_data = {}

# CSV 파일 읽기
with open(csv_file_path, mode="r", encoding="utf-8") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    
    # 각 행에 대해 처리
    for row in csv_reader:
        school_name = row["학교명"]
        subject = row["교과명"]
        teacher_count = int(row["교원수(계)"])

        # 학교가 딕셔너리에 없으면 추가
        if school_name not in school_data:
            school_data[school_name] = {"교원": {}}

        # 교원 데이터 추가 또는 누적
        if subject not in school_data[school_name]["교원"]:
            school_data[school_name]["교원"][subject] = teacher_count
        else:
            school_data[school_name]["교원"][subject] += teacher_count

# JSON 형태로 저장
output_json_path = "C:/Users/POWER/Desktop/project_peanutbutterlee/데이터/2023_학교별 교원 현황.json"
with open(output_json_path, mode="w", encoding="utf-8") as json_file:
    json.dump(school_data, json_file, ensure_ascii=False, indent=2)

print("데이터가 성공적으로 저장되었습니다.")
