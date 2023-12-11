import json

file_path_1 = "데이터/2022년도_1학기_학교별 개설 교과목_clean_위치_학년통합.json"
file_path_2 = "데이터/2022년도_2학기_학교별 개설 교과목_clean_위치_학년통합.json"

# JSON 파일 읽기
with open(file_path_1, 'r', encoding='utf-8') as file:
    data_1 = json.load(file)

with open(file_path_2, 'r', encoding='utf-8') as file:
    data_2 = json.load(file)

# 학교 코드를 기준으로 데이터 병합
merged_data = []
for school_1st_sem in data_1:
    for school_2nd_sem in data_2:
        if school_1st_sem["SD_SCHUL_CODE"] == school_2nd_sem["SD_SCHUL_CODE"]:
            merged_subjects = {}
            for subject in set(school_1st_sem["all_SUBJECT"]).union(school_2nd_sem["all_SUBJECT"]):
                # 교과목별로 수업시수 합치기
                hours_1st_sem = school_1st_sem["all_SUBJECT"].get(subject, 0)
                hours_2nd_sem = school_2nd_sem["all_SUBJECT"].get(subject, 0)
                merged_subjects[subject] = hours_1st_sem + hours_2nd_sem

            merged_school = {
                "ATPT_OFCDC_SC_CODE": school_1st_sem["ATPT_OFCDC_SC_CODE"],
                "ATPT_OFCDC_SC_NM": school_1st_sem["ATPT_OFCDC_SC_NM"],
                "SD_SCHUL_CODE": school_1st_sem["SD_SCHUL_CODE"],
                "SCHUL_NM": school_1st_sem["SCHUL_NM"],
                "SEM_1": 3,
                "AY_1": school_1st_sem["AY"],
                "Latitude_1": school_1st_sem["Latitude"],
                "Longitude_1": school_1st_sem["Longitude"],
                "고등학교구분명": school_1st_sem["고등학교구분명"],
                "SEM_2": school_2nd_sem["SEM"],
                "AY_2": school_2nd_sem["AY"],
                "Latitude_2": school_2nd_sem["Latitude"],
                "Longitude_2": school_2nd_sem["Longitude"],
                "all_SUBJECT": merged_subjects,
            
            }
            merged_data.append(merged_school)


# JSON 파일로 저장
output_file_path = "데이터/2022년도_학교별 개설 교과목_clean_위치_학년통합_학기통합.json"
with open(output_file_path, "w", encoding="utf-8") as json_file:
    json.dump(merged_data, json_file, ensure_ascii=False, indent=2)

print("데이터가 성공적으로 저장되었습니다.")
