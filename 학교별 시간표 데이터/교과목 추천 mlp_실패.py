import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from tqdm import tqdm

class MLPRecommendationModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPRecommendationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def initialize_model(input_size, hidden_size, output_size, learning_rate):
    model = MLPRecommendationModel(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    return model, optimizer, loss_function

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

def generate_data():
    job_to_major = format_major_data_from_file()
    major_to_subjects = format_subjects_data_from_file()

    all_jobs = list(job_to_major.keys())
    all_majors = list(major for majors in job_to_major.values() for major in majors)
    all_subjects = list(set(subject for subjects in major_to_subjects.values() for subject in subjects))

    job_to_index = {job: i for i, job in enumerate(all_jobs)}
    major_to_index = {major: i for i, major in enumerate(all_majors)}


    indexed_job_to_major = {job: [major_to_index[major] for major in majors] for job, majors in job_to_major.items()}
    indexed_major_to_subjects = {major: [subject for subject in subjects] for major, subjects in major_to_subjects.items()}

    return all_jobs, all_majors, all_subjects, indexed_job_to_major, indexed_major_to_subjects, job_to_index, major_to_index

# 데이터 전처리 및 학습에 필요한 함수들
def prepare_input_data(job_index, num_jobs, job_to_index):
    input_data = np.zeros(num_jobs)
    if job_index in job_to_index:
        input_data[job_to_index[job_index]] = 1
    return torch.tensor(input_data, dtype=torch.float32)


def create_training_data(indexed_job_to_major, indexed_major_to_subjects, all_jobs, all_majors):
    positive_pairs = []
    for job, majors in indexed_job_to_major.items():
        for major in majors:
            for subject in indexed_major_to_subjects[all_majors[major]]:
                positive_pairs.append((job, subject, 1))

    negative_pairs = []
    for _ in range(len(positive_pairs)):
        random_job = np.random.randint(len(all_jobs))
        random_subject = np.random.randint(len(all_majors))  # 수정된 부분
        negative_pairs.append((random_job, random_subject, 0))

    return positive_pairs + negative_pairs


# 모델 학습 함수
def train_model(model, optimizer, loss_function, training_data, input_size, num_epochs, job_to_index):
    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = len(training_data)

        tqdm_data = tqdm(enumerate(training_data), total=len(training_data), desc=f'Epoch {epoch + 1}/{num_epochs}')

        for batch_idx, (job_index, major_index, label) in tqdm_data:
            input_data = prepare_input_data(job_index, input_size, job_to_index)
            label = torch.FloatTensor([label])

            optimizer.zero_grad()
            output = model(input_data)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predicted_label = (output >= 0.5).float()
            correct_predictions += (predicted_label == label).sum().item()

            accuracy = correct_predictions / (batch_idx + 1) * 100
            tqdm_data.set_postfix(loss=loss.item(), accuracy=f'{accuracy:.2f}%')

        average_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples * 100
        tqdm_data.close()
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}, Accuracy: {accuracy:.2f}%')



# 희망 직업을 입력으로 받아 교과목 추천 함수
def recommend_subjects_by_job(model, job, all_subjects, job_to_index, input_size):
    job_index = job_to_index.get(job)
    if job_index is not None:
        input_data = prepare_input_data(job_index, input_size, job_to_index)
        model_output = model(input_data)
        predicted_label = (model_output >= 0.5).item()

        if predicted_label:
            recommended_subjects = [all_subjects[i] for i in range(len(all_subjects)) if model_output[i] >= 0.5]
            print(f"{job} 직업에 대한 추천 교과목: {recommended_subjects}")
        else:
            print(f"{job} 직업에 대한 교과목 추천이 어려워보입니다.")
    else:
        print(f"{job} 직업은 데이터에 없습니다.")

# 데이터 생성 및 초기화
all_jobs, all_majors, all_subjects, indexed_job_to_major, indexed_major_to_subjects, job_to_index, major_to_index = generate_data()

# 하이퍼파라미터 설정
input_size = len(all_jobs)
hidden_size = len(all_majors)
output_size = 1 #len(all_subjects)
learning_rate = 0.001
num_epochs = 10

# 모델 초기화
model, optimizer, loss_function = initialize_model(input_size, hidden_size, output_size, learning_rate)

# 학습 데이터 생성
training_data = create_training_data(indexed_job_to_major, indexed_major_to_subjects, all_jobs, all_majors)

# 모델 학습
train_model(model, optimizer, loss_function, training_data, input_size, num_epochs, job_to_index)

# 희망 직업 입력
hope_job = "시각디자이너"

# 교과목 추천
recommend_subjects_by_job(model, hope_job, all_subjects, job_to_index, input_size)