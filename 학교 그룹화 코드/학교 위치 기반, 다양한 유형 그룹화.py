import json
import random
from geopy.distance import geodesic


def create_school_data_from_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    schools = []

    for school_data in json_data:
        school_info = {
            "name": school_data["SD_SCHUL_CODE"],
            "location": [float(school_data["Latitude"]), float(school_data["Longitude"])],
            "sub_Cluster": int(school_data["sub_Cluster"]),
            "location_Cluster": school_data["location_Cluster"]
            # 필요한 다른 데이터 추가
        }
        schools.append(school_info)

    return schools

def create_random_group(location_with_sub, max_size, retry_count, min_sort):
    best_solution = []

    for sub_location_group in location_with_sub:
        best_solution_sub, best_avg_distance = optimize_subgroup(sub_location_group, max_size, retry_count, min_sort)

        if best_solution_sub:
            best_solution += best_solution_sub

    # 빈 그룹 제거
    best_solution = [group for group in best_solution if group]

    return best_solution

def make_list_for_locationCluster(schools_data):
    locations_with_sub = {}

    for school in schools_data:
        location_cluster = school["location_Cluster"]
        if location_cluster not in locations_with_sub:
            locations_with_sub[location_cluster] = []

        locations_with_sub[location_cluster].append(school)
    location_data = list(locations_with_sub.values())

    return location_data

def optimize_subgroup(sub_location_group, max_size, retry_count, min_sort):
    current_solution, current_avg_distance = generate_random_solution_for_subgroup(sub_location_group, max_size, min_sort)
    best_solution = current_solution
    best_avg_distance = current_avg_distance
    count = 0

    for _ in range(retry_count):
        count += 1
        print(count, "번 실행")
        new_solution, new_avg_distance = generate_random_solution_for_subgroup(sub_location_group, max_size, min_sort)

        if new_avg_distance < best_avg_distance:
            best_solution, best_avg_distance = new_solution, new_avg_distance

    print(f"서브그룹 진행 완료, 거리 평균 : {best_avg_distance}")

    return best_solution, best_avg_distance


def generate_random_solution_for_subgroup(sub_location_group, max_size, min_sort):
    school_group = []
    random.shuffle(sub_location_group)
    grouped_schools = []

    for school in sub_location_group:
        if len(school_group) < max_size:
            school_group.append(school)
        else:
            grouped_schools.append(list(school_group))  # 새로운 그룹 시작
            school_group = [school]  # 현재 학교를 포함하는 새로운 그룹 시작

    # 남은 학교 추가
    grouped_schools.append(school_group) if school_group else None

    check_valid(grouped_schools, min_sort)
    avg_distance = calculate_average_distance(grouped_schools)

    return grouped_schools, avg_distance

def check_valid(sub_group, min_cluster_count):
    for i, group in enumerate(sub_group):
        sub_clusters = set(school["sub_Cluster"] for school in group)
        if len(sub_clusters) < min_cluster_count:
            print(f"그룹 {i}에 기준에 미치지 않는 학교가 있습니다. 다른 그룹으로 재배치합니다.")
            reassign_school_to_group(group, sub_group, min_cluster_count)



def reassign_school_to_group(affected_group, all_groups, min_cluster_count):
    schools_to_remove = []

    for school in affected_group:
        for other_group in all_groups:
            if len(set(school["sub_Cluster"] for school in other_group)) >= min_cluster_count:
                other_group.append(school)
                schools_to_remove.append(school)

    for school in schools_to_remove:
        if school in affected_group:
            affected_group.remove(school)
            print(f"{school['name']} 학교가 그룹 재배치되었습니다.")

def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).kilometers

def calculate_average_distance(groups):
    total_distance = 0
    total_pairs = 0

    for group in groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                total_distance += calculate_distance(group[i]["location"], group[j]["location"])
                total_pairs += 1

    average_distance = total_distance / total_pairs if total_pairs > 0 else 0.0
    return average_distance

# 실행 코드
file_path = '데이터/위치_유형_클러스터링_결과.json'
max_size = 5
min_sort = 3
retry_count = 200


schools_data = create_school_data_from_json(file_path)
locations_with_sub = make_list_for_locationCluster(schools_data)
print(locations_with_sub)
best_solution = create_random_group(locations_with_sub, max_size, retry_count,min_sort)
print(best_solution)




def match_final_clusters(existing_data, best_solution):
    # 기존 데이터를 학교 코드를 기준으로 딕셔너리로 변환
    existing_data_dict = {school_info["SD_SCHUL_CODE"]: school_info for school_info in existing_data}

    # 최종 클러스터 결과를 학교 코드에 맞게 넣기
    for group_index, group in enumerate(best_solution, start=1):
        for school_info in group:
            school_code = school_info["name"]
            existing_data_dict[school_code]["final_Cluster"] = group_index

    return existing_data

def save_to_json(data, result_file_path):
    with open(result_file_path, 'w', encoding='utf-8') as result_file:
        json.dump(data, result_file, ensure_ascii=False, indent=2)

# 실행 코드
json_file_path = '데이터/위치_유형_클러스터링_결과.json'
result_file_path = '데이터/클러스터일 최종_결과.json'

# 기존 JSON 파일의 내용 읽기
with open(json_file_path, 'r', encoding='utf-8') as file:
    existing_data = json.load(file)

# 결과를 기존 데이터에 추가
updated_data = match_final_clusters(existing_data, best_solution)

# 결과를 새로운 파일에 쓰기
save_to_json(updated_data, result_file_path)

print(f"결과가 {result_file_path}에 추가되었습니다.")
