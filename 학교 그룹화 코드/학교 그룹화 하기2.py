import json
import itertools

def get_subjects_in_group(group, school_data):
    subjects = set()
    for school_code in group:
        for subject in school_data[school_code]['all_SUBJECT']:
            subjects.add(subject)
    return subjects

def merge_subjects_in_group(group, school_data):
    merged_subjects = {}
    for school_code in group:
        subjects = school_data[school_code]['all_SUBJECT']
        for subject, hours in subjects.items():
            if subject in merged_subjects:
                merged_subjects[subject] += hours
            else:
                merged_subjects[subject] = hours
    return merged_subjects

def find_optimal_grouping(school_data, nodes_per_group):
    # 학교 코드 추출
    school_codes = list(school_data.keys())
    print(f"All School Codes: {school_codes}")

    # 가능한 모든 그룹 조합 생성
    all_group_combinations = list(itertools.combinations(school_codes, nodes_per_group))
    
    # 최대 교과목 종류와 그룹 조합 초기화
    max_subjects = set()
    optimal_grouping = None

    # 각 그룹 조합에 대해 교과목 종류 계산
    for i, group_combination in enumerate(all_group_combinations, start=1):
        subjects_in_group = get_subjects_in_group(group_combination, school_data)
        
        # 진행 상황 퍼센트 계산 및 출력
        progress_percent = (i / len(all_group_combinations)) * 100
        print(f"\rProgress: {progress_percent:.2f}%", end="")
        
        if len(subjects_in_group) > len(max_subjects):
            max_subjects = subjects_in_group
            optimal_grouping = group_combination
    
    return optimal_grouping, max_subjects

def main():
    # JSON 파일에서 학교 데이터 불러오기
    with open('데이터/2023년도_1학기_학교별 개설교과목_교원 수.json', 'r', encoding='utf-8') as file:
        school_data = {school['SD_SCHUL_CODE']: school for school in json.load(file)}
    
    # 그룹 내 노드 수 선택 (3 또는 4)
    nodes_per_group = 3
    
    # 최적의 그룹 조합 찾기
    optimal_grouping, max_subjects = find_optimal_grouping(school_data, nodes_per_group)
    
    # 각 그룹에 속한 학교별 개설 교과목과 수업 시수 출력
    for i, school_code in enumerate(optimal_grouping, start=1):
        subjects_in_group = merge_subjects_in_group(optimal_grouping, school_data)
        print(f"\nGroup {i} - School Code: {school_code}")
        print(f"Subjects: {subjects_in_group}")

    # 결과 출력
    print(f"\nOptimal Grouping: {optimal_grouping}")
    print(f"Max Subjects: {max_subjects}")

if __name__ == "__main__":
    main()
