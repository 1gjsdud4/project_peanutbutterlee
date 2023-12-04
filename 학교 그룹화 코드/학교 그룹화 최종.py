import json
import random



def data_load(file_path):
    # JSON 파일 불러오기
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



def check_valid(sub_group,min_size,min_sort):
    opt = []
    sort = [] 
    for group in sub_group:
        print(group)
        if len(group)>= min_size:
            for sub in group:
                if sub["sub_Cluster"] not in sort:
                    sort.append(sub["sub_Cluster"])
            if len(sort) < min_sort:
                opt.append(False) 
    result = False
    if False in opt:
        result = True
    return result  

def creat_random_group(location_with_sub,min_size,min_sort,retry_count):
    success_group = []
    fali_group = []

    for sub_loction_group in location_with_sub:
        grouped_schools = []
        school_group = []
        print(sub_loction_group)
        random.shuffle(sub_loction_group)
        
        
        tried = 0
        while tried < retry_count:
            tried += 1   
            print(f'{tried}번 째 시도')
            count = 0
            for school in sub_loction_group :
                count += 1
                if len(sub_loction_group)-(count - min_size) < min_size:
                    grouped_schools[random.randrange(0,len(grouped_schools))].append(school)
                else:
                    if len(school_group) < min_size :
                        school_group.append(school)
                    else:
                        grouped_schools.append(school_group)
                        school_group=[]
            
            if check_valid(grouped_schools,min_size,min_sort) == True: 
                success_group.append(grouped_schools)
                print("조합 성공")
                break

            if tried == retry_count:
                fali_group.append(grouped_schools)
                print("조합 실패")
            
    return success_group, fali_group


### 실행 ###

file_path = '데이터/최종클러스터링_결과.json'
min_size = 5 
min_sort = 3 
retry_count = 200


locations_with_sub = data_load(file_path)
suc_group, fail_group  = creat_random_group(locations_with_sub, min_size,min_sort,retry_count)

print(suc_group)
print(fail_group)






