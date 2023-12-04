import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json

def format_job_data_from_file():

    file_path = '데이터/직업_학과데이터.json'

    with open(file_path, 'r', encoding='utf-8') as json_file:
        job_data_list = json.load(json_file)

    major_data = {}

    for job_info in job_data_list:
        job_name = job_info.get("job_nm")
        depart_sum = []
        if job_name:
            depart_list = job_info.get("depart_list", [])
            if None in depart_list:
                continue
            else:
                for depart in depart_list:
                        depart_name = depart["depart_name"]
                        depart_sum.append(depart_name)
                major_data[job_name] = depart_sum
    
    return major_data
                
def format_highschool_data_from_file():
    file_path = '데이터/학과_고교교과목_데이터.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    result_dict = {}

    for item in data:
        major_name = item["major"]
        related_subject = item["related_subject"]

        for subject_type, subjects in related_subject.items():
            if major_name not in result_dict:
                result_dict[major_name] = []

            # 교과목 리스트에 추가
            result_dict[major_name].extend(subjects)
    
    return result_dict


def generate_data():
    

    major_data = format_job_data_from_file()
    subjects_data = format_highschool_data_from_file()


    all_majors = list(set(major for majors in major_data.values() for major in majors))
    all_subjects = list(set(subject for subjects in subjects_data.values() for subject in subjects))

    major_to_index = {major: i for i, major in enumerate(all_majors)}
    subject_to_index = {subject: i for i, subject in enumerate(all_subjects)}

    indexed_major_data = {job: [major_to_index[major] for major in majors] for job, majors in major_data.items()}
    indexed_subjects_data = {major: [subject_to_index[subject] for subject in subjects] for major, subjects in subjects_data.items()}

    print(all_majors, all_subjects, indexed_major_data, indexed_subjects_data)
    return all_majors, all_subjects, indexed_major_data, indexed_subjects_data


generate_data()