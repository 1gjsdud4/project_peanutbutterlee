import json

def get_departments(selected_job_codes, json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        occupations = json.load(file)

    selected_departments = {}

    for selected_job_code in selected_job_codes:
        for occupation in occupations:
            if occupation['job_cd'] == selected_job_code:
                selected_departments[selected_job_code] = [depart['depart_name'] for depart in occupation['depart_list']]
                break

    return selected_departments

# 학생이 선택한 희망 진로 코드들 (예시로 국회의원 코드 1061과 기업고위임원 코드 238 사용)
selected_job_codes = [1061, 238]

# JSON 파일 경로 (적절히 수정 필요)
json_file_path = '데이터/직업_학과데이터.json'

# 선택한 진로에 따른 학과 리스트 출력
selected_departments = get_departments(selected_job_codes, json_file_path)

if selected_departments:
    print("선택한 진로와 관련된 학과:")
    for job_code, departments in selected_departments.items():
        print(f"\n진로 코드 {job_code}:")
        for department in departments:
            print(f"- {department}")
else:
    print("해당하는 진로가 데이터에 없습니다.")
