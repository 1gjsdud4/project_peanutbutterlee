from urllib.request import urlopen
import json
import os
import data_function as dataf


# #file_path = dataf.download_jobs_data()
# file_path = 'C:/Users/POWER/Desktop/project_peanutbutterlee/데이터/직업데이터_page_1.json'
# with open(file_path, 'r', encoding='utf-8') as json_file:
#     parsed_data = json.load(json_file)

# job_cd_list = [job["job_cd"] for job in parsed_data["jobs"]]
# print(job_cd_list)

# dataf.download_schoolsubject_data(job_cd_list,)

# dataf.download_depart_data()

file_path_depart = 'C:/Users/POWER/Desktop/project_peanutbutterlee/데이터/학과리스트데이터.json'
with open(file_path_depart, 'r', encoding='utf-8') as json_file:
     parsed_data = json.load(json_file)

major_seq_list = [data["majorSeq"] for data in parsed_data]
print(len(major_seq_list))

dataf.download_HigeSchoolsubject_data(major_seq_list)