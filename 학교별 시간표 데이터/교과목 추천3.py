import torch
import torch.nn as nn
import torch.optim as optim
import json


def format_major_data_from_file():
    file_path = '데이터/직업_학과데이터.json'

    with open(file_path, 'r', encoding='utf-8') as json_file:
        job_data_list = json.load(json_file)

    job_to_major = {}

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
                job_to_major[job_name] = depart_sum
    print(job_to_major)

    return job_to_major

def format_subjects_data_from_file():
    file_path = '데이터/학과_고교교과목_데이터.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    major_to_subjects = {}

    for item in data:
        major_name = item["major"]
        related_subject = item["related_subject"]

        for subject_type, subjects in related_subject.items():
            if major_name not in major_to_subjects:
                major_to_subjects[major_name] = []

            # 교과목 리스트에 추가
            major_to_subjects[major_name].extend(subjects)

    print(major_to_subjects)

    return major_to_subjects



job_department_data = format_major_data_from_file()
department_subjects_data = format_subjects_data_from_file()


# 전체 교과목 목록 생성
all_subjects = list(set(subject for subjects in department_subjects_data.values() for subject in subjects))

# 교과목을 one-hot 인코딩으로 변환
def subjects_to_tensor(subjects, all_subjects):
    return torch.tensor([subject in subjects for subject in all_subjects], dtype=torch.float32)

# 데이터를 텐서로 변환
job_department_data_tensor = [(subjects_to_tensor(departments, list(job_department_data.values())),
                               list(job_department_data.keys()).index(job))
                              for job, departments in job_department_data.items()]

department_subjects_data_tensor = [(subjects_to_tensor(subjects, all_subjects),
                                    list(department_subjects_data.keys()).index(department))
                                   for department, subjects in department_subjects_data.items()]

# 간단한 신경망 모델 정의
class RecommenderModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RecommenderModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

# 모델 생성
input_size_job = len(list(job_department_data.values())[0])
output_size_job = len(job_department_data)
model_job = RecommenderModel(input_size_job, output_size_job)

input_size_department = len(all_subjects)
output_size_department = len(department_subjects_data)
model_department = RecommenderModel(input_size_department, output_size_department)

# 손실 함수와 최적화 함수 정의
criterion = nn.CrossEntropyLoss()
optimizer_job = optim.Adam(model_job.parameters(), lr=0.01)
optimizer_department = optim.Adam(model_department.parameters(), lr=0.01)

# 모델 학습
num_epochs = 1000
for epoch in range(num_epochs):
    for departments_tensor, job_index in job_department_data_tensor:
        optimizer_job.zero_grad()
        output_job = model_job(departments_tensor)
        loss_job = criterion(output_job.unsqueeze(0), job_index)
        loss_job.backward()
        optimizer_job.step()

    for subjects_tensor, department_index in department_subjects_data_tensor:
        optimizer_department.zero_grad()
        output_department = model_department(subjects_tensor)
        loss_department = criterion(output_department.unsqueeze(0), department_index)
        loss_department.backward()
        optimizer_department.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss (Job): {loss_job.item()}, Loss (Department): {loss_department.item()}')

# 학습된 모델로 예측
def recommend_department(job, model, all_subjects):
    with torch.no_grad():
        job_index = torch.tensor([list(job_department_data.keys()).index(job)], dtype=torch.long)
        departments_tensor = subjects_to_tensor([], list(job_department_data.values()))
        output = model(departments_tensor)
        _, predicted_index = torch.max(output, 1)
        predicted_department = list(department_subjects_data.keys())[predicted_index.item()]
        return predicted_department

# 예측 테스트
test_job = 'Data Scientist'
predicted_department = recommend_department(test_job, model_job, list(job_department_data.values()))
print(f'For {test_job}, Recommended Department: {predicted_department}')
