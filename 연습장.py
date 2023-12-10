import json

def subjects_of_same_cluster(student_school_code, schools_data):
    # 입력된 학교 코드로 학교 정보 가져오기
    student_school_data = next((school for school in schools_data if school["SD_SCHUL_CODE"] == student_school_code), None)

    if student_school_data is None:
        print("학교 정보를 찾을 수 없습니다.")
        return []

    all_subjects_same_cluster = set()

    # 입력된 학교의 final_Cluster를 가져옴
    student_cluster = student_school_data["final_Cluster"]

    # final_Cluster가 같은 학교들의 데이터 필터링
    relevant_schools = [school for school in schools_data if school["final_Cluster"] == student_cluster]

    # 교과목 추천을 위한 데이터 생성
    for school in relevant_schools:
        # 각 학교의 모든 교과목을 종합
        all_subjects_same_cluster.update(school["all_SUBJECT"].keys())

    return list(all_subjects_same_cluster)

school_path = '데이터/최종_결과.json'
with open(school_path, 'r', encoding='utf-8') as json_file:
    schools_data = json.load(json_file)

# 학생의 "SD_SCHUL_CODE"를 입력 받기
student_school_code = "7010057"

# 교과목 추천 함수 호출
all_subjects_same_cluster = subjects_of_same_cluster(student_school_code, schools_data)

print(all_subjects_same_cluster)