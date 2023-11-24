import json
import re

json_file_path = "C:/Users/POWER/Desktop/project_peanutbutterlee/데이터/2023년도_1학기_학교별 개설 교과목.json"

with open(json_file_path, "r", encoding="utf-8") as json_file:
    json_data = json.load(json_file)

# 정규표현식을 사용하여 A-Z, 0-9, [보강] 등의 텍스트 제거
def clean_subject(subject):
    # 알파벳 (소문자 및 대문자)과 숫자 제거
    cleaned_sub = re.sub(r'[a-zA-Z0-9①-②]', '', subject)
    # [,보,강,]이 모두 포함되어 있을 때만 삭제
    if all(char in subject for char in '[보강]'):
        cleaned_sub = re.sub(r'\[보강\]', '', subject)
    return cleaned_sub

# 삭제하고자 하는 교과목 목록
subjects_to_remove = ['진로', '진로활동', '봉사활동','봉사','진로 활동','봉사 활동','토요휴업일','자율활동','자율','자율 활동',"자율활동적응","자율활동자치",'동아리활동','동아리 활동','동아리']

# 데이터를 수정하여 저장할 딕셔너리 초기화
cleaned_data_dict = {}

# 각 데이터 항목을 순회하면서 교과목 수정
for item in json_data:
    key = (item["SCHUL_NM"], item["AY"], item["SEM"], item["GRADE"])
    if key not in cleaned_data_dict:
        cleaned_data_dict[key] = {
            "ATPT_OFCDC_SC_CODE": item["ATPT_OFCDC_SC_CODE"],
            "ATPT_OFCDC_SC_NM": item["ATPT_OFCDC_SC_NM"],
            "SD_SCHUL_CODE": item["SD_SCHUL_CODE"],
            "SCHUL_NM": item["SCHUL_NM"],
            "AY": item["AY"],
            "SEM": item["SEM"],
            "GRADE": item["GRADE"],
            "SUBJECT": {},
        }

    # 각 교과목의 이름에서 특정 부분 추출 (예: A10, C10)
    for sub, credit in item["SUBJECT"].items():
        cleaned_sub = clean_subject(sub)
        if cleaned_sub not in subjects_to_remove:
            if cleaned_sub in cleaned_data_dict[key]["SUBJECT"]:
                cleaned_data_dict[key]["SUBJECT"][cleaned_sub] += credit
            else:
                cleaned_data_dict[key]["SUBJECT"][cleaned_sub] = credit

# 딕셔너리를 리스트로 변환하여 수정된 데이터를 JSON 파일로 저장
cleaned_data = list(cleaned_data_dict.values())
output_file_path = "C:/Users/POWER/Desktop/project_peanutbutterlee/데이터/2023년도_1학기_학교별 개설 교과목_clean.json"
with open(output_file_path, "w", encoding="utf-8") as json_file:
    json.dump(cleaned_data, json_file, ensure_ascii=False, indent=2)

print("데이터가 성공적으로 저장되었습니다.")
