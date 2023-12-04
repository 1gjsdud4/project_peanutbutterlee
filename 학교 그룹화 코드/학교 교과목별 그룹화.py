import pandas as pd
from sklearn.cluster import KMeans
import json
# CSV 파일 읽기
data = pd.read_csv('데이터/결과데이터.csv', encoding='utf-8')

# '학교' 열을 index로 설정
data.set_index('학교', inplace=True)

# K-means 클러스터링 수행
kmeans = KMeans(n_clusters=5)  # 클러스터 개수를 조정할 수 있습니다
clusters = kmeans.fit_predict(data)

# 클러스터 결과를 데이터프레임에 추가
data['sub_Cluster'] = clusters

# 각 그룹별 학교 목록 출력
for i in range(data['sub_Cluster'].nunique()):
    cluster_data = data[data['sub_Cluster'] == i]
    print(f'그룹 {i + 1}에 속한 학교들:')
    print(cluster_data.index)
    print("\n")
# 각 그룹별 공통으로 개설된 교과목 출력
for i in range(data['sub_Cluster'].nunique()):
    cluster_data = data[data['sub_Cluster'] == i].drop('sub_Cluster', axis=1)
    common_subjects = cluster_data.columns[cluster_data.all(axis=0)]
    
    print(f'그룹 {i + 1}에 속한 학교들에서 공통으로 개설된 교과목:')
    print(common_subjects)
    print("\n")


# JSON 파일 읽기
with open('데이터/위치클러스터링_결과.json', 'r', encoding='utf-8') as file:
    data_json = json.load(file)

# JSON 데이터프레임으로 변환
df_json = pd.DataFrame(data_json)
df_json.set_index('SCHUL_NM', inplace=True)

# 학교 교과목 별 클러스터링 결과를 추가
df_json['sub_Cluster'] = data['sub_Cluster']

# JSON 데이터프레임을 JSON 파일로 저장
df_json.to_json('데이터/최종클러스터링_결과.json', orient='records', force_ascii=False, indent=4)