import json

# JSON 파일로부터 데이터 로드
file_path_major = '데이터/직업_학과데이터.json'  # 직업_학과데이터.json 파일 경로에 맞게 수정
file_path_subjects = '데이터/학과_고교교과목_데이터.json'  # 학과_교과목데이터.json 파일 경로에 맞게 수정

with open(file_path_major, 'r', encoding='utf-8') as file:
    major_data = json.load(file)

with open(file_path_subjects, 'r', encoding='utf-8') as file:
    subjects_data = json.load(file)

def find_related_subjects(selected_major):
    for major_info in subjects_data:
        if major_info['major'] == selected_major:
            related_subjects = major_info['related_subject']
            return related_subjects

    # 선택한 학과가 데이터에 없을 경우
    return None

def recommend_subjects(selected_majors):
    recommended_subjects = set()

    # 각 선택한 진로에 대해 관련된 학과의 교과목을 찾아 합집합을 구함
    for selected_major in selected_majors:
        related_subjects = find_related_subjects(selected_major)
        if related_subjects:
            recommended_subjects.update(related_subjects)

    return recommended_subjects

# 사용자가 선택한 진로
selected_majors = "행정부고위공무원" # 선택한 진로에 맞게 수정

# 선택한 진로와 관련된 교과목 추천
recommended_subjects = recommend_subjects(selected_majors)

# 결과 출력
if recommended_subjects:
    print("추천 교과목:")
    for subject in recommended_subjects:
        print(f" - {subject}")
else:
    print("선택한 진로에 대한 정보가 데이터에 없습니다.")
