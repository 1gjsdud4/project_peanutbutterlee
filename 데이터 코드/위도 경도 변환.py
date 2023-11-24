import pandas as pd
from geopy.geocoders import Nominatim

# 데이터프레임 읽기
df = pd.read_csv('C:/Users/POWER/Desktop/project_peanutbutterlee/데이터/고등학교기본정보.csv')

# 위도와 경도 정보를 추가할 새로운 열 생성
df['Latitude'] = None
df['Longitude'] = None

# Nominatim 지오코더 생성
geolocator = Nominatim(user_agent="South Korea")
a=1
# 주소를 이용하여 위도와 경도 정보 가져오기
for index, row in df.iterrows():
    address = row['도로명주소']
    print(a,"개 학교 진행중")
    if pd.notna(address):  # 주소가 존재하는 경우에만 처리
        location = geolocator.geocode(address)
        if location:
            df.at[index, 'Latitude'] = location.latitude
            df.at[index, 'Longitude'] = location.longitude
        a+=1

# 결과를 새로운 CSV 파일로 저장
df.to_csv('C:/Users/POWER/Desktop/project_peanutbutterlee/데이터/고등학교기본정보(위도경도 추가).csv', index=False)
