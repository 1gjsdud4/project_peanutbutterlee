import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import json

# JSON 파일로부터 데이터 읽기
print("Step 1: Reading data from JSON file")
with open('데이터/2022년도_학교별 개설교과목_교원 수.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 데이터프레임으로 변환
print("Step 2: Converting data to DataFrame")
df = pd.DataFrame(data)

# 위치 정보만 추출
locations = df[["Latitude", "Longitude"]]

# K-평균 모델 생성 
print("Step 3: Creating KMeans model")
kmeans = KMeans(n_clusters=10)
df["location_Cluster"] = kmeans.fit_predict(locations)

# 클러스터링 평가 (실루엣 지표)
silhouette_avg = silhouette_score(locations, df["location_Cluster"])
print(f"Silhouette Score: {silhouette_avg}")

# 클러스터링 평가 (Davies-Bouldin Index)
db_index = davies_bouldin_score(locations, df["location_Cluster"])
print(f"Davies-Bouldin Index: {db_index}")

# 그룹화 결과 확인
print("Step 4: Viewing clustering results")
print(df[["SCHUL_NM", "Latitude", "Longitude", "location_Cluster"]])

# 시각화 (예시로 그룹이 30개이므로 30개의 색으로 표시)
print("Step 5: Visualizing clustering results")
plt.scatter(df["Longitude"], df["Latitude"], c=df["location_Cluster"], cmap='viridis', edgecolor='k')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("School Clusters based on Location")
plt.show()

# 클러스터링 결과를 JSON으로 저장
print("Step 6: Saving clustering results to JSON file")
clustered_data = df.to_dict(orient='records')
with open('데이터/위치클러스터링_결과.json', 'w', encoding='utf-8') as outfile:
    json.dump(clustered_data, outfile, ensure_ascii=False, indent=4)
