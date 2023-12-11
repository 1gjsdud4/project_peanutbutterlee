import pandas as pd
import json

# JSON 파일로부터 학교 데이터 읽어오기
json_file_path = "데이터/2022년도_1학기_학교별 개설 교과목_clean.json"
with open(json_file_path, "r", encoding="utf-8") as json_file:
    school_data = json.load(json_file)

# CSV 파일로부터 위치 정보 및 고등학교 구분 정보를 읽어와서 데이터프레임으로 저장
location_file_path = "C:/Users/POWER/Desktop/project_peanutbutterlee/데이터/고등학교기본정보(위도경도 추가).csv"
df_locations = pd.read_csv(location_file_path, encoding="utf-8")

# 데이터프레임에서 불필요한 열 제거
df_locations = df_locations[["행정표준코드", "Latitude", "Longitude", "고등학교구분명"]]


# 학교 코드를 인덱스로 설정하여 딕셔너리로 변환
school_locations = df_locations.set_index("행정표준코드").to_dict(orient="index")

print(school_locations)

count = 1

# 학교 정보에 위도, 경도, 그리고 고등학교종류 딕셔너리 추가
for school in school_data:
    print(count)
    count += 1
    school_code = school["SD_SCHUL_CODE"]
    # 해당 학교 코드에 대한 위치 정보 가져오기
    location_info = school_locations.get(int(school_code), {"Latitude": 0.0, "Longitude": 0.0, "고등학교구분명": "일반고"})

    # 학교 정보에 위치 정보 추가
    school.update({
        "Latitude": location_info["Latitude"],
        "Longitude": location_info["Longitude"],
        "고등학교구분명": location_info["고등학교구분명"]
    })

# JSON 파일로 저장
output_json_path = "데이터/2022년도_1학기_학교별 개설 교과목_clean_위치.json"
with open(output_json_path, "w", encoding="utf-8") as output_json_file:
    json.dump(school_data, output_json_file, ensure_ascii=False, indent=4)

print(f"JSON 파일이 성공적으로 생성되었습니다: {output_json_path}")
