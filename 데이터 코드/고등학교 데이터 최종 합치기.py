import json
import pandas as pd

# 학교별 교원 데이터
with open('C:/Users/POWER/Desktop/project_peanutbutterlee/데이터/2023_학교별 교원 현황.json', 'r', encoding='utf-8') as file:
    teachers_data = json.load(file)

# 학교별 개설 교과목 데이터
with open('C:/Users/POWER/Desktop/project_peanutbutterlee/데이터/2023년도_1학기_학교별 개설 교과목_clean_위치_학년통합.json', 'r', encoding='utf-8') as file:
    subjects_data = json.load(file)

# 데이터를 합치기 위한 함수
def merge_teachers_and_subjects(teachers_data, subjects_data):
    merged_data = []

    for subject_info in subjects_data:
        school_name = subject_info["SCHUL_NM"]
        # 학교 이름을 키로 사용하여 데이터를 불러오기
        if school_name in teachers_data:
            # NaN이 있는 행을 건너뛰기
            if any(pd.isna(value) for value in subject_info.values()):
                continue
            
            # 학교 이름을 키로 사용하여 새로운 딕셔너리 생성
            merged_school_data = {
                "ATPT_OFCDC_SC_CODE": subject_info.get("ATPT_OFCDC_SC_CODE", ""),
                "ATPT_OFCDC_SC_NM": subject_info.get("ATPT_OFCDC_SC_NM", ""),
                "SD_SCHUL_CODE": subject_info.get("SD_SCHUL_CODE", ""),
                "SCHUL_NM": subject_info.get("SCHUL_NM", ""),
                "SEM": subject_info.get("SEM", ""),
                "AY": subject_info.get("AY", ""),
                "Latitude": subject_info.get("Latitude", ""),
                "Longitude": subject_info.get("Longitude", ""),
                "고등학교구분명": subject_info.get("고등학교구분명", ""),
                "all_SUBJECT": subject_info.get("all_SUBJECT", {}),
                "교원": teachers_data[school_name]["교원"]  # 교원 데이터를 추가
            }

            # 합쳐진 데이터를 리스트에 추가
            merged_data.append(merged_school_data)

    return merged_data

# 데이터 합치기
merged_data = merge_teachers_and_subjects(teachers_data, subjects_data)

# 결과를 JSON 파일로 저장
with open('C:/Users/POWER/Desktop/project_peanutbutterlee/데이터/2023년도_1학기_학교별 개설교과목_교원 수.json', 'w', encoding='utf-8') as file:
    json.dump(merged_data, file, ensure_ascii=False, indent=2)

print("데이터가 성공적으로 합쳐졌고, 2023년도_1학기_학교별 개설교과목_교원 수.json 파일에 저장되었습니다.")
