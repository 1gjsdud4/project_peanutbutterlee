import json
import matplotlib.pyplot as plt
from collections import defaultdict

def create_school_data_from_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    schools = []

    for school_data in json_data:
        school_info = {
            "name": school_data["SD_SCHUL_CODE"],
            "location": [float(school_data["Latitude"]), float(school_data["Longitude"])],
            "final_Cluster": school_data["final_Cluster"]
        }
        schools.append(school_info)

    return schools

def visualize_clusters_with_lines(schools):
    # 그룹화된 학교 정보
    clusters = defaultdict(list)
    for school in schools:
        clusters[school["final_Cluster"]].append(school)

    # 시각화
    for cluster, schools_in_cluster in clusters.items():
        lats, lons = zip(*[(school["location"][0], school["location"][1]) for school in schools_in_cluster])
        plt.scatter(lons, lats, label=f'Cluster {cluster}')

        # 같은 클러스터에 속하는 학교들을 선으로 연결
        for i in range(len(schools_in_cluster) - 1):
            plt.plot([schools_in_cluster[i]["location"][1], schools_in_cluster[i+1]["location"][1]],
                     [schools_in_cluster[i]["location"][0], schools_in_cluster[i+1]["location"][0]],
                     color='gray', linestyle='dashed')

    plt.title('School Clusters with Lines')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.show()

# 실행 코드
json_file_path = '데이터/최종_결과.json'  # 실제 파일 경로로 변경
schools_data = create_school_data_from_json(json_file_path)
visualize_clusters_with_lines(schools_data)
