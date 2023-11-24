import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# JSON 파일로부터 데이터 로드
file_path = '데이터/직업_학과데이터.json'  # 파일 경로에 맞게 수정
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# DataFrame 생성
df = pd.DataFrame(data)

# 폰트 지정
font_path = "C:/Windows/Fonts/HMKMRHD.TTF"  # 나눔고딕 폰트 경로를 실제 설치된 경로로 수정
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 학과 간의 유사도를 계산하여 가중치 부여 (임의의 가중치)
similarity_weights = {}
for i in range(len(df['depart_list'])):
    for j in range(i+1, len(df['depart_list'])):
        set_i = set((item['depart_id'], item['depart_name']) for item in df['depart_list'][i])
        
        # 예외 처리 추가
        if df['depart_list'][j] is not None:
            set_j = set((item['depart_id'], item['depart_name']) for item in df['depart_list'][j])
            similarity = len(set_i & set_j)
            if similarity > 0:
                edge = (df['depart_list'][i][0]['depart_name'], df['depart_list'][j][0]['depart_name'])
                similarity_weights[edge] = similarity

# 네트워크 생성
G = nx.Graph()
for edge, weight in similarity_weights.items():
    G.add_edge(*edge, weight=weight)

# 가중치를 이용한 네트워크 시각화
pos = nx.spring_layout(G)
edge_labels = {(i, j): f'{d["weight"]}' for i, j, d in G.edges(data=True)}

nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=2000, node_color='skyblue', font_size=8)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.show()
