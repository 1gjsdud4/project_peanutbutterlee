import pandas as pd
from sklearn.cluster import KMeans

# CSV 파일 읽기
data = pd.read_csv('데이터/결과데이터.csv', encoding='utf-8')

# '학교' 열을 index로 설정
data.set_index('학교', inplace=True)

# K-means 클러스터링 수행
kmeans = KMeans(n_clusters=15)  # 클러스터 개수를 조정할 수 있습니다
clusters = kmeans.fit_predict(data)

# 클러스터 결과를 데이터프레임에 추가
data['Cluster'] = clusters

# 각 그룹별 학교 목록 출력
for i in range(data['Cluster'].nunique()):
    cluster_data = data[data['Cluster'] == i]
    print(f'그룹 {i + 1}에 속한 학교들:')
    print(cluster_data.index)
    print("\n")
# 각 그룹별 공통으로 개설된 교과목 출력
for i in range(data['Cluster'].nunique()):
    cluster_data = data[data['Cluster'] == i].drop('Cluster', axis=1)
    common_subjects = cluster_data.columns[cluster_data.all(axis=0)]
    
    print(f'그룹 {i + 1}에 속한 학교들에서 공통으로 개설된 교과목:')
    print(common_subjects)
    print("\n")