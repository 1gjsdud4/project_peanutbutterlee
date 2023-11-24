import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import json

# JSON 파일로부터 데이터 읽기
print("Step 1: Reading data from JSON file")
with open('데이터/2023년도_1학기_학교별 개설교과목_교원 수.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 데이터프레임으로 변환
print("Step 2: Converting data to DataFrame")
df = pd.DataFrame(data)

# 위치 정보만 추출
locations = df[["Latitude", "Longitude"]]

# K-평균 모델 생성 (예시로 2개의 그룹으로 나눔)
print("Step 3: Creating KMeans model")
kmeans = KMeans(n_clusters=15)
df["Cluster"] = kmeans.fit_predict(locations)

# 클러스터링 평가 (실루엣 지표)
silhouette_avg = silhouette_score(locations, df["Cluster"])
print(f"Silhouette Score: {silhouette_avg}")

# 클러스터링 평가 (Davies-Bouldin Index)
db_index = davies_bouldin_score(locations, df["Cluster"])
print(f"Davies-Bouldin Index: {db_index}")

# 그룹화 결과 확인
print("Step 4: Viewing clustering results")
print(df[["SCHUL_NM", "Latitude", "Longitude", "Cluster"]])

# 시각화 (예시로 그룹이 2개이므로 2개의 색으로 표시)
print("Step 5: Visualizing clustering results")
plt.scatter(df["Longitude"], df["Latitude"], c=df["Cluster"], cmap='viridis', edgecolor='k')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("School Clusters based on Location")
plt.show()

# JSON 파일로 결과 저장 (보기 쉽게)
output_json_path = 'cluster_results_pretty.json'
df.to_json(output_json_path, orient='records', lines=True, force_ascii=False, indent=2)
print(f"Clustering results saved to {output_json_path}")