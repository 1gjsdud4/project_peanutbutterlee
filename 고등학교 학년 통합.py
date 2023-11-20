import json

json_file_path = "C:/Users/POWER/Desktop/project_peanutbutterlee/데이터/2023년도_1학기_학교별 개설 교과목(위치 및 종류 추가).json"

with open(json_file_path, "r", encoding="utf-8") as json_file:
    json_data = json.load(json_file)

# 새로운 데이터를 저장할 딕셔너리 초기화
merged_data = {}

# 각 데이터 항목을 순회하면서 데이터 병합
for item in json_data:
    key = (item["SCHUL_NM"], item["AY"], item["SEM"])  # 학교명, 학기를 키로 사용
    if key not in merged_data:
        merged_data[key] = {
            "ATPT_OFCDC_SC_CODE": item["ATPT_OFCDC_SC_CODE"],
            "ATPT_OFCDC_SC_NM": item["ATPT_OFCDC_SC_NM"],
            "SD_SCHUL_CODE": item["SD_SCHUL_CODE"],
            "SCHUL_NM": item["SCHUL_NM"],
            "AY": item["AY"],
            "all_SUBJECT": {},
            "Latitude": item["Latitude"],
            "Longitude": item["Longitude"],
            "고등학교구분명": item["고등학교구분명"]
        }

    # 학년별로 데이터 병합
    for subject, credit in item["SUBJECT"].items():
        if subject not in merged_data[key]["all_SUBJECT"]:
            merged_data[key]["all_SUBJECT"][subject] = 0
        merged_data[key]["all_SUBJECT"][subject] += credit

# 최종적으로 병합된 데이터를 리스트로 변환
merged_data_list = list(merged_data.values())

# JSON 파일로 저장
output_file_path = "C:/Users/POWER/Desktop/project_peanutbutterlee/데이터/2023년도_1학기_학교별 개설 교과목 통합.json"
with open(output_file_path, "w", encoding="utf-8") as json_file:
    json.dump(merged_data_list, json_file, ensure_ascii=False, indent=2)

print("데이터가 성공적으로 저장되었습니다.")
