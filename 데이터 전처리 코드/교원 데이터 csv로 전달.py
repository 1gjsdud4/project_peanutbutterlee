import json
import pandas as pd

# JSON 파일 경로
json_file_path = '데이터/2023년도_1학기_학교별 개설교과목_교원 수.json'
# CSV 파일 경로
csv_file_path = '데이터/고등학교 교과목 종류.CSV'

# JSON 파일 읽기
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    school_data = json.load(json_file)

# CSV 파일 읽기
subject_data = pd.read_csv(csv_file_path)

# 중복된 값 제거
subject_data = subject_data.drop_duplicates(subset=['고등교과목'])

# 고등 교과목 데이터에서 과목 목록 가져오기
subjects = subject_data['고등교과목']

# 새로운 DataFrame 생성
result_data = pd.DataFrame(columns=['학교'] + list(subjects))
print(result_data)

# 학교 데이터 처리
for school_entry in school_data:
    school_name = school_entry["SCHUL_NM"]
    
    # 각 교과목별 개설된 수업시수 초기화
    subject_count = {subject: 0 for subject in subjects}
    
    # 교과목 데이터를 확인하여 수업시수 누적
    for h_subject, h_subject_count_value in school_entry["all_SUBJECT"].items():
        for sub in subjects:
            if h_subject == sub:
                # 해당 교과목의 수업시수 누적
                subject_count[sub] = h_subject_count_value
    
    # 결과 DataFrame에 추가
    result_data.loc[len(result_data)] = [school_name] + list(subject_count.values())
    print(f"{school_name} 처리 완료")

# CSV 파일로 저장
result_data.to_csv('데이터/결과데이터.csv', index=False, encoding='utf-8-sig')

print("작업 완료. 결과 파일을 확인하세요.")
