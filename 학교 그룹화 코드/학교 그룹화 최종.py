import json
import random
import math

def data_load(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    cluster_data = {}

    for school in json_data:
        school_code = school["SD_SCHUL_CODE"]
        location_cluster = school["location_Cluster"]
        sub_cluster = school["sub_Cluster"]

        if location_cluster not in cluster_data:
            cluster_data[location_cluster] = []

        cluster_data[location_cluster].append({"학교": school_code, "sub_Cluster": sub_cluster})

        print(f"학교: {school_code}, Location Cluster: {location_cluster}, Sub Cluster: {sub_cluster}")

    schools_data_list = list(cluster_data.values())
    return schools_data_list


def create_random_group(location_with_sub, min_size, min_sort, retry_count, initial_temperature, cooling_rate):
    best_solution = []
    temperature = initial_temperature

    for sub_location_group in location_with_sub:
        best_solution_sub, energy = optimize_subgroup(sub_location_group, min_size, min_sort, retry_count, temperature)
        if energy == 0:
            print("100% 최적화")
        else:
            print(f"{(energy * 100)}% 최적화")
        best_solution += best_solution_sub

    return best_solution


def optimize_subgroup(sub_location_group, min_size, min_sort, retry_count, temperature):
    current_solution, current_energy = generate_random_solution_for_subgroup(sub_location_group, min_size, min_sort)
    best_solution = current_solution
    best_energy = current_energy
    count = 0
    for _ in range(retry_count):
        count +=1
        print(count,"번 실행")
        new_solution, new_energy = generate_random_solution_for_subgroup(sub_location_group, min_size, min_sort)
        delta_energy = new_energy - current_energy

        if delta_energy < 0 or random.random() < math.exp(-delta_energy / temperature):
            current_solution, current_energy = new_solution, new_energy

        if new_energy < best_energy:
            best_solution, best_energy = new_solution, new_energy
        
        if best_energy == 0:
            break

    return best_solution, best_energy


def generate_random_solution_for_subgroup(sub_location_group, min_size, min_sort):
    school_group = []
    random.shuffle(sub_location_group)
    grouped_schools = []

    count = 0
    for school in sub_location_group:
        count += 1
        if len(sub_location_group) - (count - min_size) < min_size:
            grouped_schools[random.randrange(0, len(grouped_schools))].append(school)
        else:
            if len(school_group) < min_size:
                school_group.append(school)
            else:
                grouped_schools.append(school_group)
                school_group = []

    return grouped_schools, energy(grouped_schools, min_size, min_sort)

def energy(sub_group, min_size, min_sort):
    validity_ratio = check_valid(sub_group, min_size, min_sort)
    return validity_ratio


def check_valid(sub_group, min_size, min_sort):
    opt = []
    sort = []
    for group in sub_group:
        if len(group) >= min_size:
            for sub in group:
                if sub["sub_Cluster"] not in sort:
                    sort.append(sub["sub_Cluster"])
            if len(sort) < min_sort:
                opt.append(False)
    result = 1.0
    if False not in opt:
        result = 0.0
    else:
        result = (opt.count(False) / len(opt))
    return result

# 실행 코드
file_path = '데이터/최종클러스터링_결과.json'
min_size = 5
min_sort = 3
retry_count = 200
initial_temperature = 100.0
cooling_rate = 0.99

locations_with_sub = data_load(file_path)
best_solution = create_random_group(locations_with_sub, min_size, min_sort, retry_count, initial_temperature, cooling_rate)
print(best_solution)
