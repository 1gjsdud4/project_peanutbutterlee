import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json

# CSV 파일 읽기
data = pd.read_csv('데이터/2022_학교별 개설 교과 수업시수.csv', encoding='utf-8')

# '학교' 열을 index로 설정
data.set_index('학교', inplace=True)

# 데이터 표준화
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# PCA를 사용하여 차원 축소
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data_scaled)

# K-means 클러스터링 수행
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(data_2d)

# 클러스터 결과를 데이터프레임에 추가
data['sub_Cluster'] = clusters

# 실루엣 스코어 계산
silhouette_avg = silhouette_score(data_2d, clusters)
print(f"실루엣 스코어: {silhouette_avg}")

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
df_json.to_json('데이터/위치_유형_클러스터링_결과.json', orient='records', force_ascii=False, indent=4)
