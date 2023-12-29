import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
import numpy as np

# 텍스트 출력 옵션 설정
pd.set_option('display.max_rows', None)  # 행의 최대 출력 수 제한 해제
pd.set_option('display.max_columns', None)  # 열의 최대 출력 수 제한 해제
pd.set_option('display.max_colwidth', None)  # 열의 내용 전체 출력

# CSV 파일 읽기
data = pd.read_csv('데이터/2022_학교별 개설 교과 수업시수.csv', encoding='utf-8')

# '학교' 열을 index로 설정
data.set_index('학교', inplace=True)

# 데이터 표준화
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 주성분 개수 결정
target_variance_ratio = 0.3
cumulative_variance_ratio = 0
n_components = 0

while cumulative_variance_ratio < target_variance_ratio:
    n_components += 1
    pca = PCA(n_components=n_components)
    pca.fit(data_scaled)
    cumulative_variance_ratio = np.sum(pca.explained_variance_ratio_)

print(f"선택된 주성분 개수: {n_components}")
print(f"누적 설명된 분산: {cumulative_variance_ratio}")

# 선택된 주성분 개수로 PCA 적용
pca = PCA(n_components=n_components)
data_2d = pca.fit_transform(data_scaled)

# K-means 클러스터링 수행
kmeans = KMeans(n_clusters=12)
clusters = kmeans.fit_predict(data_2d)

# 클러스터 결과를 데이터프레임에 추가
data['sub_Cluster'] = clusters

# 실루엣 스코어 계산
silhouette_avg = silhouette_score(data_2d, clusters)
print(f"실루엣 스코어: {silhouette_avg}")

# 결과를 텍스트 파일로 저장
result_filename = '결과.txt'
with open(result_filename, 'w', encoding='utf-8') as result_file:
    result_file.write(f"실루엣 스코어: {silhouette_avg}\n\n")

    # 각 그룹별 학교 목록 및 교과목 출력
    for i in range(data['sub_Cluster'].nunique()):
        result_file.write(f'그룹 {i + 1}에 속한 학교들:\n')
        result_file.write(str(data[data['sub_Cluster'] == i].index.to_list()) + '\n\n')

        result_file.write(f'그룹 {i + 1}에 속한 학교들에서 공통으로 개설된 교과목:\n')
        cluster_data = data[data['sub_Cluster'] == i].drop('sub_Cluster', axis=1)
        common_subjects = cluster_data.columns[cluster_data.all(axis=0)].to_list()

        if not common_subjects:
            # 공통된 교과목이 없으면 가장 많은 학교에서 개설된 교과목 중 상위 10개 출력
            most_common_subjects = cluster_data.sum().sort_values(ascending=False).index[:10].tolist()
            result_file.write(str(most_common_subjects) + '\n\n')
        else:
            result_file.write(str(common_subjects) + '\n\n')

        result_file.write(f'그룹 {i + 1}의 핵심 특성:\n')
        for ind in kmeans.cluster_centers_[i].argsort()[::-1][:5]:
            result_file.write(data.columns[ind] + '\n')
        result_file.write('\n')

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

print(f"결과가 '{result_filename}' 파일로 저장되었습니다.")


