import json
import pandas as pd
from sklearn.cluster import KMeans
import itertools

def kmeans_clustering(locations, clusters):
    # KMeans 모델 생성 (위도 경도를 기반으로 그룹화)
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    locations["Cluster"] = kmeans.fit_predict(locations)
    print("K-means clustering completed.")

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

def find_optimal_group_in_cluster(group, school_data):
    # 가능한 모든 그룹 조합 생성
    all_group_combinations = list(itertools.combinations(group, 3))
    
    # 최대 교과목 종류와 그룹 조합 초기화
    max_subjects = set()
    optimal_grouping = None

    # 각 그룹 조합에 대해 교과목 종류 계산
    for i, group_combination in enumerate(all_group_combinations, start=1):
        subjects_in_group = get_subjects_in_group(group_combination, school_data)
        
        # 진행 상황 퍼센트 계산 및 출력
        progress_percent = (i / len(all_group_combinations)) * 100
        print(f"\rFinding optimal group in cluster: {progress_percent:.2f}%", end="")
        
        if len(subjects_in_group) > len(max_subjects):
            max_subjects = subjects_in_group
            optimal_grouping = group_combination
    
    print("\nOptimal group found in the cluster.")
    return optimal_grouping, max_subjects

def find_optimal_grouping(locations, school_data):
    optimal_grouping = {}
    max_subjects = set()
    
    for cluster in locations["Cluster"].unique():
        # 현재 클러스터에 속한 학교들의 코드 추출
        current_group = locations[locations["Cluster"] == cluster].index.tolist()
        
        # 최적의 그룹 조합 및 교과목 종류 계산
        current_optimal_group, current_max_subjects = find_optimal_group_in_cluster(current_group, school_data)
        
        # 전체 결과 갱신
        optimal_grouping[cluster] = current_optimal_group
        max_subjects.update(current_max_subjects)
    
        # 결과 출력
        print(f"\nCluster {cluster + 1} - Optimal Group: {current_optimal_group}")
        print(f"Max Subjects: {current_max_subjects}")
    
    return optimal_grouping, max_subjects

def find_optimal_average_grouping(locations, school_data, target_groups):
    optimal_grouping = {}
    max_average_subjects = []
    
    for cluster in target_groups:
        # 현재 클러스터에 속한 학교들의 코드 추출
        current_group = locations[locations["Cluster"] == cluster].index.tolist()
        
        # 가능한 모든 그룹 조합 생성
        all_group_combinations = list(itertools.combinations(current_group, 3))
        
        # 최대 교과목 종류와 평균 교과목 수 초기화
        max_subjects = set()
        max_average = 0

        # 각 그룹 조합에 대해 교과목 종류와 평균 계산
        for i, group_combination in enumerate(all_group_combinations, start=1):
            subjects_in_group = get_subjects_in_group(group_combination, school_data)
            average_subjects = len(subjects_in_group) / 3
            
            # 진행 상황 퍼센트 계산 및 출력
            progress_percent = (i / len(all_group_combinations)) * 100
            print(f"\rFinding optimal average group in cluster: {progress_percent:.2f}%", end="")
            
            if average_subjects > max_average:
                max_subjects = subjects_in_group
                max_average = average_subjects
        
        # 전체 결과 갱신
        optimal_grouping[cluster] = max_subjects
        max_average_subjects.append(max_subjects)
        
        # 결과 출력
        print(f"\nCluster {cluster + 1} - Max Subjects for Average: {max_subjects}")
    
    return optimal_grouping, max_average_subjects

def main():
    # JSON 파일에서 학교 데이터 불러오기
    with open('데이터/2023년도_1학기_학교별 개설교과목_교원 수.json', 'r', encoding='utf-8') as file:
        school_data = {school['SD_SCHUL_CODE']: school for school in json.load(file)}
    
    # 데이터프레임으로 변환
    df = pd.DataFrame(school_data).T
    
    # 위치 정보 추출
    locations = df[["Latitude", "Longitude"]]
    
    # 클러스터 수 선택 (15)
    clusters = 15
    
    # K-means 클러스터링
    kmeans_clustering(locations, clusters)
    
    # 최적의 그룹 조합 찾기
    optimal_grouping, max_subjects = find_optimal_grouping(locations, school_data)
    
    # 최적의 평균 그룹 조합 찾기
    target_groups = list(optimal_grouping.keys())
    optimal_average_grouping, max_average_subjects = find_optimal_average_grouping(locations, school_data, target_groups)
    
    # 결과 출력
    print(f"\nFinal Optimal Grouping: {optimal_grouping}")
    print(f"Final Max Subjects: {max_subjects}")
    print(f"Final Optimal Average Grouping: {optimal_average_grouping}")
    print(f"Final Max Average Subjects: {max_average_subjects}")

if __name__ == "__main__":
    main()
