import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import torch.nn.functional as F

class RecommendationModel(nn.Module):
    def __init__(self, num_majors, num_subjects, embedding_dim):
        super(RecommendationModel, self).__init__()
        self.major_embedding = nn.Embedding(num_majors, embedding_dim)
        self.subject_embedding = nn.Embedding(num_subjects, embedding_dim)

    def forward(self, major_indices, subject_indices):
        major_embedded = self.major_embedding(major_indices)
        subject_embedded = self.subject_embedding(subject_indices)
        similarity_scores = F.cosine_similarity(major_embedded.unsqueeze(1), subject_embedded.unsqueeze(0), dim=2)
        return similarity_scores.view(-1)  # 1D로 변경
    
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

    return all_majors, all_subjects, indexed_major_data, indexed_subjects_data, major_to_index

def initialize_model(num_majors, num_subjects, embedding_dim):
    model = RecommendationModel(num_majors, num_subjects, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    return model, optimizer, loss_function

def create_training_data(indexed_major_data, indexed_subjects_data, all_majors, all_subjects):
    positive_pairs = []
    for job, majors in indexed_major_data.items():
        for major in majors:
            for subject in indexed_subjects_data[all_majors[major]]:
                # label을 1로 변경
                positive_pairs.append((major, subject, 1))

    negative_pairs = []
    for _ in range(len(positive_pairs)):
        random_major = np.random.randint(len(all_majors))
        random_subject = np.random.randint(len(all_subjects))
        # label을 0으로 변경
        negative_pairs.append((random_major, random_subject, 0))

    return positive_pairs + negative_pairs


def train_model(model, optimizer, loss_function, training_data, num_epochs):
    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = len(training_data)

        for major_index, subject_index, label in training_data:
            major_index = torch.LongTensor([major_index])
            subject_index = torch.LongTensor([subject_index])
            label = torch.FloatTensor([label])

            optimizer.zero_grad()
            output = model(major_index, subject_index)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 이진 분류에서 정확도 계산
            predicted_labels = (output >= 0.5).float()
            correct_predictions += (predicted_labels == label).sum().item()

        # 에폭이 끝날 때마다 손실과 정확도 출력
        average_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}, Accuracy: {accuracy * 100:.2f}%')

        # 여기에 추가적인 로깅이나 평가 코드를 추가할 수 있습니다.



def recommend_subjects(model, example_major, all_subjects):
    example_major_tensor = torch.LongTensor([example_major])
    all_subject_indices = torch.LongTensor(list(range(len(all_subjects))))
    similarity_scores = model(example_major_tensor, all_subject_indices)
    top_subject_indices = torch.topk(similarity_scores.squeeze(), k=5).indices.numpy()
    top_subjects = [all_subjects[i] for i in top_subject_indices]

    print(f"추천 교과목: {top_subjects}")

# 하이퍼파라미터 설정
embedding_dim = 10
learning_rate = 0.001
num_epochs = 1

# 데이터 생성 및 초기화
all_majors, all_subjects, indexed_major_data, indexed_subjects_data, major_to_index = generate_data()
model, optimizer, loss_function = initialize_model(len(all_majors), len(all_subjects), embedding_dim)
print(major_to_index)

# 학습 데이터 생성
training_data = create_training_data(indexed_major_data, indexed_subjects_data, all_majors, all_subjects)

# 모델 학습
train_model(model, optimizer, loss_function, training_data, num_epochs)

# 추천 예시
example_major = major_to_index["국회의원"]
recommend_subjects(model, example_major, all_subjects)