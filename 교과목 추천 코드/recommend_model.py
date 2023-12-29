import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

class RecommendationModel(nn.Module):
    def __init__(self, num_jobs, num_majors, num_subjects, embedding_dim):
        super(RecommendationModel, self).__init__()
        self.num_majors = num_majors  # num_majors 속성 추가
        self.num_subjects = num_subjects  # num_subjects 속성 추가
        self.job_embedding = nn.Embedding(num_jobs, embedding_dim)
        self.major_embedding = nn.Embedding(num_majors, embedding_dim)
        self.subject_embedding = nn.Embedding(num_subjects, embedding_dim)

    def forward(self, job_indices, major_indices, subject_indices):

        # 각각의 임베딩 계산
        job_embedded = self.job_embedding(job_indices)
        major_embedded = self.major_embedding(major_indices)
        subject_embedded = self.subject_embedding(subject_indices)

        # 크기를 맞추기 위해 unsqueeze 사용
        job_major_similarity = F.cosine_similarity(job_embedded.unsqueeze(1), major_embedded.unsqueeze(0), dim=2)
        major_subject_similarity = F.cosine_similarity(major_embedded.unsqueeze(1), subject_embedded.unsqueeze(0), dim=2)

        # 차원을 맞추기 위해 unsqueeze 사용
        job_major_similarity = job_major_similarity.unsqueeze(2)

        final_similarity = job_major_similarity * major_subject_similarity

        return final_similarity.view(-1)


    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f"모델이 {path}에 저장되었습니다.")

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint)
        print(f"모델이 {path}에서 불러와졌습니다.")



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

            if subject_type == "일반선택과목":
                for category, sub_subjects in subjects.items():
                    major_to_subjects[major_name].extend(sub_subjects)
            else:
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
    subject_to_index = {subject: i for i, subject in enumerate(all_subjects)}

    indexed_job_to_major = {job: [major_to_index[major] for major in majors] for job, majors in job_to_major.items()}

    indexed_major_to_subjects = {}
    for major, subjects in major_to_subjects.items():
        if major in major_to_index:
            indexed_major_to_subjects[major_to_index[major]] = [subject_to_index[subject] for subject in subjects]

    # indexed_data 수정: 각 job에 대한 major와 subject의 인덱스를 사용하여 생성
    indexed_data = [(job_to_index[job], major_index, subject_index)
                    for job, majors in job_to_major.items()
                    for major_index in [major_to_index[major] for major in majors]
                    for subject_index in indexed_major_to_subjects.get(major_index, [])]

    return all_jobs, all_majors, all_subjects, indexed_data, indexed_major_to_subjects, job_to_index, major_to_index, subject_to_index

def initialize_model(num_jobs, num_majors, num_subjects, embedding_dim, learning_rate):
    # 임베딩 크기가 너무 작거나 크지 않도록 적절한 값을 선택
    embedding_dim = min(embedding_dim, min(num_jobs, num_majors, num_subjects))
    
    model = RecommendationModel(num_jobs, num_majors, num_subjects, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()   
    
    return model, optimizer, loss_function

def create_training_data(indexed_data, all_jobs, all_majors, all_subjects):
    positive_pairs = []

    for job_index, major_index, subject_index in indexed_data:
        positive_pairs.append((job_index, major_index, subject_index, 1))

    negative_pairs = []
    '''
    for _ in range(len(positive_pairs)):
        while True:
            random_job = np.random.randint(len(all_jobs))
            random_major = np.random.randint(len(all_majors))
            random_subject = np.random.randint(len(all_subjects))

            # 추가: 만약 부정적인 예시가 이미 긍정적인 샘플에 있다면 추가하지 않음
            if (random_job, random_major, random_subject, 1) not in positive_pairs:
                negative_pairs.append((random_job, random_major, random_subject, 0))
                break
    '''
    for _ in range(len(positive_pairs)+len(negative_pairs)):
        while True:
            random_job = np.random.randint(len(all_jobs))
            random_major = np.random.randint(len(all_majors))
            random_subject = np.random.randint(len(all_subjects))

            if (random_job, random_major, random_subject, 1) in positive_pairs:
                positive_pairs.append((random_job, random_major, random_subject, 1))
                break  
            else:
                negative_pairs.append((random_job, random_major, random_subject, 0))
                break

    print('데이터 생성 완료')
    print(len(positive_pairs),len(negative_pairs))
    return positive_pairs, negative_pairs

def systematic_sampling(positive_pairs, negative_pairs, test_size=0.2, random_state=42):
    # 체계적 표본 추출을 위한 간격 계산
    data = positive_pairs + negative_pairs

    train_data, eval_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, eval_data

def evaluate_model(model, data_loader, threshold=0.5):
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in data_loader:
            job_indices, major_indices, subject_indices, labels = batch
            job_index = torch.LongTensor([job_indices])
            major_index = torch.LongTensor([major_indices])
            subject_index = torch.LongTensor([subject_indices])
            label = torch.FloatTensor([labels])

            output = model(job_index, major_index, subject_index)
            predictions = (output >= threshold).float()

            all_labels.extend([label] if isinstance(label, int) else label.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    roc_auc = roc_auc_score(all_labels, all_predictions)

    cm = confusion_matrix(all_labels, all_predictions)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'ROC-AUC Score: {roc_auc:.4f}')
    print('Confusion Matrix:')
    print(cm)

    return accuracy



def train_model_with_validation(model, optimizer, loss_function, training_data, eval_data, num_epochs):
    train_losses = []
    train_accuracies = []
    eval_accuracies = []

    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = len(training_data)

        tqdm_data = tqdm(enumerate(training_data), total=len(training_data), desc=f'Epoch {epoch + 1}/{num_epochs}')

        for batch_idx, (job_index, major_index, subject_index, label) in tqdm_data:
            job_index = torch.LongTensor([job_index])
            major_index = torch.LongTensor([major_index])
            subject_index = torch.LongTensor([subject_index])
            label = torch.FloatTensor([label])

            optimizer.zero_grad()
            output = model(job_index, major_index, subject_index)

            label = label.view_as(output)

            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predicted_labels = (output >= 0.5).float()
            correct_predictions += (predicted_labels == label).sum().item()
            accuracy = correct_predictions / (batch_idx + 1) * 100
            tqdm_data.set_postfix(loss=loss.item(), accuracy=f'{accuracy:.2f}%')

        average_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples
        train_losses.append(average_loss)
        train_accuracies.append(accuracy)
        tqdm_data.close()
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {average_loss}, Train Accuracy: {accuracy * 100:.2f}%')

        # 검증 데이터 평가
        eval_accuracy = evaluate_model(model, eval_data)
        eval_accuracies.append(eval_accuracy)
        print(f'Epoch {epoch + 1}/{num_epochs}, Eval Accuracy: {eval_accuracy * 100:.2f}%')

        # 모델 저장 (필요시)
        model.save_model(f'모델 학습/recommendation_model_epoch_{epoch + 1}.pth')

    # 학습 과정 시각화
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), eval_accuracies, label='Eval Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def train_model_with_validation_and_scheduler(model, optimizer, loss_function, training_data, eval_data, num_epochs, scheduler_step_size= 10, scheduler_gamma=0.1, patience=10):
    train_losses = []
    train_accuracies = []
    eval_accuracies = []

    # 스케줄러 초기화 (StepLR)
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    # 초기화
    best_model_state = None
    best_eval_accuracy = 0.0
    current_patience = patience

    # 데이터 로더 정의
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=1, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=1, shuffle=False)

    # 학습 및 검증 과정
    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = len(training_data)

        tqdm_data = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}')

        for batch_idx, batch in tqdm_data:
            job_index, major_index, subject_index, label = batch
            job_index = torch.LongTensor([job_index])
            major_index = torch.LongTensor([major_index])
            subject_index = torch.LongTensor([subject_index])
            label = torch.FloatTensor([label])

            optimizer.zero_grad()
            output = model(job_index, major_index, subject_index)

            label = label.view_as(output)

            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predicted_labels = (output >= 0.5).float()
            correct_predictions += (predicted_labels == label).sum().item()

            accuracy = correct_predictions / (batch_idx + 1) * 100
            tqdm_data.set_postfix(loss=loss.item(), accuracy=f'{accuracy:.2f}%')

        average_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples
        train_losses.append(average_loss)
        train_accuracies.append(accuracy)
        tqdm_data.close()
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {average_loss}, Train Accuracy: {accuracy * 100:.2f}%')

        # 스케줄러 업데이트
        scheduler.step()

        # 검증 데이터 평가
        eval_accuracy = evaluate_model(model, eval_loader)
        eval_accuracies.append(eval_accuracy)
        print(f'Epoch {epoch + 1}/{num_epochs}, Eval Accuracy: {eval_accuracy * 100:.2f}%')

        # 성능 향상 시 모델 저장 및 초기화
        if eval_accuracy > best_eval_accuracy:
            best_eval_accuracy = eval_accuracy
            best_model_state = model.state_dict()
            current_patience = patience  # 초기화

        # 성능 향상이 없는 경우 patience 감소
        else:
            current_patience -= 1

            # patience가 0이 되면 조기 종료
            if current_patience == 0:
                print("조기 종료: 검증 데이터 성능 향상이 없습니다.")
                break

        tqdm_data.close()

    # 최적 모델로 복원
    model.load_state_dict(best_model_state)

    # 학습 과정 시각화
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch + 2), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch + 2), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, epoch + 2), eval_accuracies, label='Eval Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model

################################################################################################################################

# 하이퍼파라미터 설정
embedding_dim = 100
learning_rate = 0.0005
num_epochs = 100


# Generate data
all_jobs, all_majors, all_subjects, indexed_data, indexed_major_to_subjects, job_to_index, major_to_index, subject_to_index = generate_data()

'''
# Model initialization
model, optimizer, loss_function = initialize_model(len(all_jobs), len(all_majors), len(all_subjects), embedding_dim,learning_rate)

pos, neg = create_training_data(indexed_data, all_jobs, all_majors, all_subjects)
train_data, eval_data = systematic_sampling(pos,neg)

# Model training
train_model_with_validation_and_scheduler(model, optimizer, loss_function, train_data, eval_data,num_epochs)

# Save the trained model
model.save_model('모델 학습/recommendation_model_200_001_100.pth')
'''
loaded_model = RecommendationModel(len(all_jobs), len(all_majors), len(all_subjects), embedding_dim = min(embedding_dim, min(len(all_jobs), len(all_majors), len(all_subjects))))
loaded_model.load_model('모델 학습/recommendation_model_200_001_100.pth')


################################################################################################################################3
# 평가 #

# evaluate_model(loaded_model, eval_data)



##################################################################################################################3
# 추천 함수


def get_final_recommendations(desired_jobs, loaded_model, all_jobs, all_subjects, job_to_index):
    final_recommendations = []

    for example_job in desired_jobs:
        example_job_index = job_to_index.get(example_job, None)

        if example_job_index is not None:
            example_job_embedding = loaded_model.job_embedding(torch.LongTensor([example_job_index]))
            all_subject_embeddings = loaded_model.subject_embedding(torch.LongTensor(range(len(all_subjects))))
            similarities = F.cosine_similarity(example_job_embedding, all_subject_embeddings)
            num_recommendations = 40
            top_recommendations_indices = similarities.argsort(descending=True)[:num_recommendations]
            top_recommendations_subjects = [all_subjects[idx] for idx in top_recommendations_indices]

            final_recommendations.extend(set(top_recommendations_subjects))
            
            print(f"\n희망 직업: {example_job}")
            print(f"유사도 상위 {num_recommendations}개 교과목: {', '.join(top_recommendations_subjects)}")
        else:
            print(f"{example_job}은(는) 데이터에 존재하지 않는 직업입니다.")

    return final_recommendations


# 희망 직업 리스트 입력 (예시)
desired_jobs = ["광고 및 홍보전문가","조향사","광고기획자"]
final_recommendations = get_final_recommendations(desired_jobs, loaded_model, all_jobs, all_subjects, job_to_index)

# 최종 추천 교과목 출력
print("최종 교과목 추천:")
for i, subject in enumerate(final_recommendations, 1):
    print(f"{i}. {subject}")


def get_similar_jobs(target_job, loaded_model, all_jobs, job_to_index):
    similar_jobs = []

    target_job_index = job_to_index.get(target_job, None)

    if target_job_index is not None:
        target_job_embedding = loaded_model.job_embedding(torch.LongTensor([target_job_index]))
        all_job_embeddings = loaded_model.job_embedding(torch.LongTensor(range(len(all_jobs))))
        similarities = F.cosine_similarity(target_job_embedding, all_job_embeddings)
        
        # 상위 N개의 유사한 직업 추출 (예: 상위 5개)
        num_recommendations = 5
        top_recommendations_indices = similarities.argsort(descending=True)[:num_recommendations]
        similar_jobs = [all_jobs[idx] for idx in top_recommendations_indices]
    else:
        print(f"{target_job}은(는) 데이터에 존재하지 않는 직업입니다.")

    return similar_jobs

# 예시: 함수 사용
target_job = "국회의원"
similar_jobs = get_similar_jobs(target_job, loaded_model, all_jobs, job_to_index)

# 유사한 직업 출력
print(f"{target_job}과 유사한 직업 추천:")
for i, job in enumerate(similar_jobs, 1):
    print(f"{i}. {job}")



def get_similar_majors(desired_jobs, loaded_model, all_majors):
    # 희망 직업들에 대한 임베딩 벡터의 평균을 계산
    desired_jobs_indices = [job_to_index.get(job, None) for job in desired_jobs]
    desired_jobs_indices = [idx for idx in desired_jobs_indices if idx is not None]

    if len(desired_jobs_indices) > 0:
        desired_jobs_embeddings = loaded_model.job_embedding(torch.LongTensor(desired_jobs_indices))
        avg_embedding = torch.mean(desired_jobs_embeddings, dim=0, keepdim=True)

        # 모든 대학 학과에 대한 임베딩 벡터 획득
        all_major_embeddings = loaded_model.major_embedding(torch.LongTensor(range(len(all_majors))))

        # 코사인 유사성 계산
        similarities = F.cosine_similarity(avg_embedding, all_major_embeddings)

        # 상위 N개의 유사한 대학 학과 추출 (예: 상위 5개)
        num_recommendations = 5
        top_recommendations_indices = similarities.argsort(descending=True)[:num_recommendations]
        similar_majors = [all_majors[idx] for idx in top_recommendations_indices]
    else:
        print("선택한 희망 직업들에 대한 데이터가 존재하지 않습니다.")
        similar_majors = []

    return similar_majors

# 예시: 함수 사용
desired_jobs = ["국회의원"]
similar_majors = get_similar_majors(desired_jobs, loaded_model, all_majors)

# 유사한 대학 학과 출력
print("선택한 희망 직업들과 유사한 대학 학과 추천:")
for i, major in enumerate(similar_majors, 1):
    print(f"{i}. {major}")


def get_final_subject_recommendations(desired_majors, loaded_model, all_subjects, subject_to_index, indexed_major_to_subjects):
    desired_major_indices = [major_to_index.get(major, None) for major in desired_majors]
    desired_major_indices = [idx for idx in desired_major_indices if idx is not None]

    if len(desired_major_indices) > 0:
        desired_major_embeddings = loaded_model.major_embedding(torch.LongTensor(desired_major_indices))
        avg_embedding = torch.mean(desired_major_embeddings, dim=0, keepdim=True)
        all_subject_embeddings = loaded_model.subject_embedding(torch.LongTensor(range(len(all_subjects))))
        similarities = F.cosine_similarity(avg_embedding, all_subject_embeddings)

        num_recommendations = 10
        top_recommendations_indices = similarities.argsort(descending=True)[:num_recommendations]
        final_subject_recommendations = [all_subjects[idx] for idx in top_recommendations_indices]

        print(f"\n선택한 학과들: {', '.join(desired_majors)}")
        print(f"학과 임베딩의 평균: {avg_embedding}")
        print(f"유사도 상위 {num_recommendations}개 고교 교과목: {', '.join(final_subject_recommendations)}")
    else:
        print("선택한 학과들에 대한 데이터가 존재하지 않습니다.")
        final_subject_recommendations = []

    return final_subject_recommendations

# 예시: 함수 사용
desired_majors = ["교육학과"]
final_subject_recommendations = get_final_subject_recommendations(desired_majors, loaded_model, all_subjects, subject_to_index, indexed_major_to_subjects)

# 최종 교과목 출력
print("선택한 학과들을 기반으로 최종 고교 교과목 추천:")
for i, subject in enumerate(final_subject_recommendations, 1):
    print(f"{i}. {subject}")

def get_major_recommendations_for_subjects(desired_subjects, loaded_model, all_majors, major_to_index, subject_to_index, num_recommendations=5):
    desired_subject_indices = [subject_to_index.get(subject, None) for subject in desired_subjects]
    desired_subject_indices = [idx for idx in desired_subject_indices if idx is not None]

    if len(desired_subject_indices) > 0:
        desired_subject_embeddings = loaded_model.subject_embedding(torch.LongTensor(desired_subject_indices))
        all_major_embeddings = loaded_model.major_embedding(torch.LongTensor(range(len(all_majors))))
        subject_similarity = F.cosine_similarity(all_major_embeddings.unsqueeze(1), desired_subject_embeddings.unsqueeze(0), dim=2)
        avg_subject_similarity = subject_similarity.mean(dim=1)
        top_major_indices = torch.argsort(avg_subject_similarity, descending=True)[:num_recommendations]
        top_majors = [all_majors[idx.item()] for idx in top_major_indices]

        print(f"\n선택한 교과목들: {', '.join(desired_subjects)}")
        print(f"교과목 임베딩의 평균: {desired_subject_embeddings}")
        print(f"상위 {num_recommendations}개 학과 추천: {', '.join(top_majors)}")

        return top_majors
    else:
        print("선택한 교과목들에 대한 데이터가 존재하지 않습니다.")
        return []

# 예시: 함수 사용
desired_subjects = ["윤리와 사상","교육학"]
recommended_majors = get_major_recommendations_for_subjects(desired_subjects, loaded_model, all_majors, major_to_index, subject_to_index)

# 추천된 학과 출력
if recommended_majors:
    print(f"수강한 교과기반 대학 학과 추천 \n {len(recommended_majors)}개 학과 추천:")
    for i, major in enumerate(recommended_majors, 1):
        print(f"{i}. {major}")
else:
    print("추천할 학과가 없습니다.")

def get_subject_recommendations_for_completed_subjects(completed_subjects, loaded_model, all_subjects, subject_to_index, num_recommendations=5):
    # 수강한 교과목들에 대한 임베딩 벡터 획득
    completed_subject_indices = [subject_to_index.get(subject, None) for subject in completed_subjects]
    completed_subject_indices = [idx for idx in completed_subject_indices if idx is not None]

    if len(completed_subject_indices) > 0:
        completed_subject_embeddings = loaded_model.subject_embedding(torch.LongTensor(completed_subject_indices))

        # 모든 교과목에 대한 임베딩 벡터 획득
        all_subject_embeddings = loaded_model.subject_embedding(torch.LongTensor(range(len(all_subjects))))

        # 각 교과목에 대한 유사도 계산
        subject_similarity = F.cosine_similarity(all_subject_embeddings.unsqueeze(1), completed_subject_embeddings.unsqueeze(0), dim=2)

        # 각 교과목에 대한 유사도의 평균 계산
        avg_subject_similarity = subject_similarity.mean(dim=1)

        # 가장 높은 교과목 유사도를 가진 교과목 추출
        top_subject_indices = torch.argsort(avg_subject_similarity, descending=True)[:num_recommendations]
        top_subjects = [all_subjects[idx.item()] for idx in top_subject_indices]

        return top_subjects
    else:
        print("수강한 교과목 데이터가 존재하지 않습니다.")
        return []

# 예시: 함수 사용
completed_subjects = ["수학", "영어"]
recommended_subjects = get_subject_recommendations_for_completed_subjects(completed_subjects, loaded_model, all_subjects, subject_to_index)

# 추천된 교과목 출력
if recommended_subjects:
    print(f"성적이 높은 교과를 기반으로 교과목 추천 \n {len(recommended_subjects)}개 교과목 추천:")
    for i, subject in enumerate(recommended_subjects, 1):
        print(f"{i}. {subject}")
else:
    print("추천할 교과목이 없습니다.")




############################################################################################################################# 
# 학생의 학교를 기준으로 최종 교과목 추천 

def subjects_of_same_cluster(student_school_code, schools_data):
    # 입력된 학교 코드로 학교 정보 가져오기
    student_school_data = next((school for school in schools_data if school["SD_SCHUL_CODE"] == student_school_code), None)

    if student_school_data is None:
        print("학교 정보를 찾을 수 없습니다.")
        return []

    all_subjects_same_cluster = set()

    # 입력된 학교의 final_Cluster를 가져옴
    student_cluster = student_school_data["final_Cluster"]

    # final_Cluster가 같은 학교들의 데이터 필터링
    relevant_schools = [school for school in schools_data if school["final_Cluster"] == student_cluster]

    # 교과목 추천을 위한 데이터 생성
    for school in relevant_schools:
        # 각 학교의 모든 교과목을 종합
        all_subjects_same_cluster.update(school["all_SUBJECT"].keys())

    return list(all_subjects_same_cluster)

school_path = '데이터/클러스터일 최종_결과.json'
with open(school_path, 'r', encoding='utf-8') as json_file:
    schools_data = json.load(json_file)

# 학생의 "SD_SCHUL_CODE"를 입력 받기
student_school_code = "7010057"

# 교과목 추천 함수 호출
all_subjects_same_cluster = subjects_of_same_cluster(student_school_code, schools_data)

desired_jobs = ["국회의원"]
final_recommendations = get_final_recommendations(desired_jobs, loaded_model, all_jobs, all_subjects, job_to_index)
final_recommendations_base_school = [] 
for subject in final_recommendations:
    if subject in all_subjects_same_cluster:
        final_recommendations_base_school.append(subject)

final_recommendations_base_school = set(final_recommendations_base_school)

print('학생의 학교를 기준으로 추천 교과목:',final_recommendations_base_school)

