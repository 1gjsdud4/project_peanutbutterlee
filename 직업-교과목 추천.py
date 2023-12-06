import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import torch.nn.functional as F
from tqdm import tqdm

class RecommendationModel(nn.Module):
    def __init__(self, num_majors, num_subjects, embedding_dim):
        super(RecommendationModel, self).__init__()
        self.major_embedding = nn.Embedding(num_majors, embedding_dim)
        self.subject_embedding = nn.Embedding(num_subjects, embedding_dim)

    def forward(self, major_indices, subject_indices):
        major_indices = torch.LongTensor(major_indices)  # 수정: 리스트를 Tensor로 변환
        subject_indices = torch.LongTensor(subject_indices)  # 수정: 리스트를 Tensor로 변환

        major_embedded = self.major_embedding(major_indices)
        subject_embedded = self.subject_embedding(subject_indices)
        similarity_scores = F.cosine_similarity(major_embedded.unsqueeze(1), subject_embedded.unsqueeze(0), dim=2)
        return similarity_scores.view(-1)
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f"모델이 {path}에 저장되었습니다.")

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        print(f"모델이 {path}에서 불러와졌습니다.")






    
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
    print(major_data)

    return major_data

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

def initialize_model(num_majors, num_subjects, embedding_dim):
    model = RecommendationModel(num_majors, num_subjects, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    return model, optimizer, loss_function

def create_training_data(indexed_job_to_major, indexed_major_to_subjects, all_jobs, all_majors, all_subjects, job_to_index):
    positive_pairs = []
    for job, majors in indexed_job_to_major.items():
        job_index = job_to_index.get(job)
        if job_index is not None:
            for major in majors:
                for subject in indexed_major_to_subjects[all_majors[major]]:
                    subject_index = all_subjects.index(subject)  # subject를 인덱스로 변환
                    positive_pairs.append((job_index, subject_index, 1))

    negative_pairs = []
    for _ in range(len(positive_pairs)):
        random_job = np.random.randint(len(all_jobs))
        random_subject = np.random.randint(len(all_subjects))  #
        negative_pairs.append((random_job, random_subject, 0))

    return positive_pairs + negative_pairs

# train_model 함수 수정
def train_model(model, optimizer, loss_function, training_data, num_epochs):
    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = len(training_data)

        tqdm_data = tqdm(enumerate(training_data), total=len(training_data), desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for batch_idx, (job_index, subject_index, label) in tqdm_data:
            job_index = torch.LongTensor([job_index])
            subject_index = torch.LongTensor([subject_index])
            label = torch.FloatTensor([label])  # 라벨을 스칼라에서 텐서로 변환

            optimizer.zero_grad()
            output = model(job_index, subject_index)
            
            # 수정: 라벨 크기를 (1, 1)로 맞춤
            label = label.view_as(output)
            
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 이진 분류에서 정확도 계산
            predicted_labels = (output >= 0.5).float()
            correct_predictions += (predicted_labels == label).sum().item()
            accuracy = correct_predictions / (batch_idx + 1) * 100
            tqdm_data.set_postfix(loss=loss.item(), accuracy=f'{accuracy:.2f}%')

        # 에폭이 끝날 때마다 손실과 정확도 출력
        average_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples
        tqdm_data.close()
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}, Accuracy: {accuracy * 100:.2f}%')



        # 여기에 추가적인 로깅이나 평가 코드를 추가할 수 있습니다.



def recommend_subjects(model, example_job, all_subjects):
    example_job_tensor = torch.LongTensor([example_job])
    all_subject_indices = torch.LongTensor(list(range(len(all_subjects))))
    similarity_scores = model(example_job_tensor, all_subject_indices)
    top_subject_indices = torch.topk(similarity_scores.squeeze(), k=10).indices.numpy()
    top_subjects = [all_subjects[i] for i in top_subject_indices]

    print(f"추천 교과목: {top_subjects}")

# 하이퍼파라미터 설정
embedding_dim = 10
learning_rate = 0.001
num_epochs = 10

# 데이터 생성 및 초기화
all_jobs, all_majors, all_subjects, indexed_job_to_major, indexed_major_to_subjects, job_to_index, major_to_index = generate_data()
model, optimizer, loss_function = initialize_model(len(all_jobs), len(all_subjects), embedding_dim)
print(major_to_index)

# 학습 데이터 생성
training_data = create_training_data(indexed_job_to_major, indexed_major_to_subjects, all_jobs, all_majors, all_subjects,job_to_index)

# 모델 학습
train_model(model, optimizer, loss_function, training_data, num_epochs)

model.save_model('recommendation_model.pth')


# 모델 생성과 초기화
model, optimizer, loss_function = initialize_model(len(all_jobs), len(all_subjects), embedding_dim)

# 저장된 모델 불러오기
model.load_model('recommendation_model.pth')



# 추천 예시
example_major = job_to_index["국회의원"]
recommend_subjects(model, example_major, all_subjects)