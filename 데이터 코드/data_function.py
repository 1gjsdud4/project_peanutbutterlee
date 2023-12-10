from urllib.request import urlopen
import json
import os


# 학교, 연도, 학기 별 시간표 데이터를 불러오는 함수 
def download_timetable( region_code, school_code, school_name, year, sem):
    page = 1
    sum_data = []
    api_url = 'https://open.neis.go.kr/hub/hisTimetablebgs'
    api_key = '7508228bfc40454d9942bb797ffb941a' 
    amount = 1000 
    first_file_path = None

    while True:
        url = f'{api_url}?'\
            f'Type=json&pIndex={page}&pSize={amount}'\
            f'&Key={api_key}'\
            f'&ATPT_OFCDC_SC_CODE={region_code}&SD_SCHUL_CODE={school_code}&AY={year}&SEM={sem}'

        result = urlopen(url)

        # 데이터 저장
        file_path = f'데이터/{school_name}_{year}년도_{sem}학기_시간표_page_{page}.json'

        print(f"Downloading page {page} to {file_path}")
        with open(file_path, 'wb') as file:
            file.write(result.read())

        # JSON 파일을 읽고 데이터 파싱
        with open(file_path, 'r', encoding='utf-8') as json_file:
            parsed_data = json.load(json_file)

        # 만약 페이지에 내용이 없다면, 반복을 중단
        if not parsed_data.get("hisTimetable"):
            os.remove(file_path)
            page = 1
            break

        # 파일 병합
        if page == 1:
            first_file_path = file_path
            sum_data.extend(parsed_data["hisTimetable"])
        else:
            with open(first_file_path, 'r', encoding='utf-8') as merged_file:
                sum_data = json.load(merged_file)["hisTimetable"]
                sum_data += parsed_data["hisTimetable"]
            with open(first_file_path, 'w', encoding='utf-8') as merged_file:
                merged_data = {"hisTimetable": sum_data}
                json.dump(merged_data, merged_file, ensure_ascii=False, indent=4)
            print(f'{page}페이지 파일 병합 완료')
            os.remove(file_path)

        # 페이지 번호를 증가시킴
        page += 1

    return first_file_path

#데이터 내에서 개설 과목, 개설 과목에 따른 수업 횟수를 카운트하는 함수 
def count_classes(json_data):
    CLASS_COUNT = {}

    his_timetable = json_data["hisTimetable"]

    for data in his_timetable[1]["row"]:
        atpt_code = data["ATPT_OFCDC_SC_CODE"]
        atpt_name = data["ATPT_OFCDC_SC_NM"]
        school = data['SD_SCHUL_CODE']
        year = data["AY"]
        school_name = data["SCHUL_NM"]
        grade = data['GRADE']
        subject = data['ITRT_CNTNT']
        semester = data['SEM']

        class_key = (atpt_code, atpt_name, school, school_name, year, grade, semester)

        if class_key in CLASS_COUNT:
            if subject in CLASS_COUNT[class_key]:
                CLASS_COUNT[class_key][subject] += 1
            else:
                CLASS_COUNT[class_key][subject] = 1
        else:
            CLASS_COUNT[class_key] = {subject: 1}

    return CLASS_COUNT

#카운트 한것을 json으로 다시 저장
def save_count_data(input_file_path):
    # JSON 데이터를 읽어옴
    with open(input_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    # count_classes 함수를 호출하여 CLASS_COUNT 딕셔너리를 얻음
    CLASS_COUNT = count_classes(data)

    # 결과를 JSON 파일로 저장
    result_data = []
    for key, value in CLASS_COUNT.items():
        atpt_code,atpt_name, school,school_name,year, grade, semester = key
        class_data = {
            "ATPT_OFCDC_SC_CODE": atpt_code,
            "ATPT_OFCDC_SC_NM": atpt_name,
            "SD_SCHUL_CODE": school,
            "SCHUL_NM": school_name,
            "AY": year,
            "SEM": semester,
            "GRADE": grade,
            "SUBJECT": value,
        }
        result_data.append(class_data)

    return result_data

# 커리어넷에서 직업 호출 
def download_jobs_data():
    page = 1
    sum_data = []
    api_url = 'https://www.career.go.kr/cnet/front/openapi/jobs.json?apiKey'
    api_key = 'e1f6006e3e32db940ff48da62df30377' 

    while True:
        # 데이터 요청 및 저장
        url = f'{api_url}={api_key}&pageIndex={page}'
        result = urlopen(url)
        file_path = f'데이터/직업데이터_page_{page}.json'

        print(f"Downloading page {page} to {file_path}")
        with open(file_path, 'wb') as file:
            file.write(result.read())

        # JSON 파일을 읽고 데이터 파싱
        with open(file_path, 'r', encoding='utf-8') as json_file:
            parsed_data = json.load(json_file)

        # 만약 페이지에 내용이 없거나 job 내용이 없다면, 반복을 중단
        if not parsed_data.get("jobs"):
            os.remove(file_path)
            page = 1
            break

        # 파일 병합
        if page == 1:
            first_file_path = file_path
            sum_data.extend(parsed_data["jobs"])
        else:
            with open(first_file_path, 'r', encoding='utf-8') as merged_file:
                sum_data = json.load(merged_file)["jobs"]
                sum_data.extend(parsed_data["jobs"])
            with open(first_file_path, 'w', encoding='utf-8') as merged_file:
                merged_data = {"jobs": sum_data}
                json.dump(merged_data, merged_file, ensure_ascii=False, indent=4)
            print(f'{page}페이지 파일 병합 완료')
            os.remove(file_path)

        # 페이지 번호를 증가시킴
        page += 1
    
    return first_file_path

# 커리어넷 직업별 관련학과 리스트 추출 
def download_schoolsubject_data(job_cd_list):
    sum_data = []

    api_url = 'https://www.career.go.kr/cnet/front/openapi/job.json?apiKey='
    api_key = 'e1f6006e3e32db940ff48da62df30377'

    for job_cd in job_cd_list:
        url = f'{api_url}{api_key}&seq={job_cd}'
        result = urlopen(url)
        file_path = f'C:/Users/POWER/Desktop/project_peanutbutterlee/데이터/직업_학과데이터.json'

        print(f"Downloading job_cd {job_cd} to {file_path}")
        with open(file_path, 'wb') as file:
            file.write(result.read())

        # JSON 파일을 읽고 데이터 파싱
        with open(file_path, 'r', encoding='utf-8') as json_file:
            parsed_data = json.load(json_file)

        job_data = {
            "job_cd": parsed_data["baseInfo"]["job_cd"],
            "job_nm": parsed_data["baseInfo"]["job_nm"],
            "depart_list": parsed_data["departList"]
        }

        sum_data.append(job_data)

        print(f'Data for job_cd {job_cd} saved.')

        # 데이터를 JSON 파일로 저장
        with open(file_path, 'w', encoding='utf-8') as output_json_file:
            json.dump(sum_data, output_json_file, ensure_ascii=False, indent=4)

 
 #커리어넷에서 학과 리스트 만들기 
def download_depart_data():
    sum_data = []

    api_url = 'http://www.career.go.kr/cnet/openapi/getOpenApi.json?apiKey='
    api_key = 'e1f6006e3e32db940ff48da62df30377'

    url = f'{api_url}{api_key}&svcType=api&svcCode=MAJOR&contentType=json&gubun=univ_list&perPage=1000'
    result = urlopen(url)
    file_path = f'C:/Users/POWER/Desktop/project_peanutbutterlee/데이터/학과리스트데이터.json'

    with open(file_path, 'wb') as file:
        file.write(result.read())

    # JSON 파일을 읽고 데이터 파싱
    with open(file_path, 'r', encoding='utf-8') as json_file:
        parsed_data = json.load(json_file)

    for depart in parsed_data["dataSearch"]["content"]:
        depart_data = {
            "lClass": depart["lClass"],
            "mClass": depart["mClass"],
            "majorSeq": depart["majorSeq"]
        }

        sum_data.append(depart_data)

        print(' saved.')

        # 데이터를 JSON 파일로 저장
        with open(file_path, 'w', encoding='utf-8') as output_json_file:
            json.dump(sum_data, output_json_file, ensure_ascii=False, indent=4)


#학과 전공과 그와 관련된 고교 교과목 데이터 추출 
def download_HigeSchoolsubject_data(major_seq_list):
    sum_data = []
    count = 1 
    api_url = 'https://www.career.go.kr/cnet/openapi/getOpenApi?apiKey='
    api_key = 'e1f6006e3e32db940ff48da62df30377'

    for major_seq in major_seq_list:
        url = f'{api_url}{api_key}&svcType=api&svcCode=MAJOR_VIEW&contentType=json&gubun=univ_list&majorSeq={major_seq}'
        result = urlopen(url)
        file_path = f'C:/Users/POWER/Desktop/project_peanutbutterlee/데이터/학과_고교교과목_데이터.json'
        print(url)
        print(f"Downloading job_cd {major_seq} to {file_path}")
        with open(file_path, 'wb') as file:
            file.write(result.read())

        # JSON 파일을 읽고 데이터 파싱
        with open(file_path, 'r', encoding='utf-8') as json_file:
            parsed_data = json.load(json_file)

        content = parsed_data["dataSearch"]["content"]

        if len(content) > 0:
            content = content[0]  # Take the first content item if it exists
        else:
            print(f"No content found for major_seq {major_seq}")
            continue

        relate_subject = content["relate_subject"]
        restructured_data = {
            "공통과목": [],
            "일반선택과목": {},
            "진로선택과목": [],
            "전문교과Ⅰ": [],
            "전문교과Ⅱ": []
        }

        for item in relate_subject:
            subject_description = item["subject_description"]
            subject_name = item["subject_name"]

            if subject_description is None:
                continue

            subjects = [subject.strip() for subject in subject_description.split("<br>")]

            if subject_name == "일반선택과목":
                restructured_data[subject_name] = {}
                for subject in subjects:
                    category_subject = subject.split(" : ")
                    if len(category_subject) > 1:
                        category, subjects_list = category_subject
                        subjects_list = [sub.strip() for sub in subjects_list.split(",")]
                        modified_subjects_list = []
                        # ⅠㆍⅡ 분리
                        for sub in subjects_list:
                            if 'ⅠㆍⅡ' in sub:
                                subject_parts = sub.replace('ⅠㆍⅡ',"")
                                modified_subjects_list.extend([subject_parts + 'Ⅰ', subject_parts + 'Ⅱ'])
                            else:
                                modified_subjects_list.append(sub)
                        restructured_data[subject_name][category] = modified_subjects_list
            else:
                subjects = [sub.split(", ") for sub in subjects]
                restructured_data[subject_name] = [item for sublist in subjects for item in sublist]
                modified_subjects_list = []
                for subject in restructured_data[subject_name]:
                    if 'ⅠㆍⅡ' in subject:
                        subject_parts = subject.replace('ⅠㆍⅡ',"")
                        modified_subjects_list.extend([subject_parts + 'Ⅰ', subject_parts + 'Ⅱ'])
                    else:
                        modified_subjects_list.append(subject)
                restructured_data[subject_name] = modified_subjects_list

        major_data = {
            "major": content["major"],
            "major_seq": major_seq,
            "related_subject": restructured_data
        }

        sum_data.append(major_data)

        print(f'{count}개 학과의 고등 교과목 데이터 저장완료.')
        count +=1

        with open(file_path, 'w', encoding='utf-8') as output_json_file:
            json.dump(sum_data, output_json_file, ensure_ascii=False, indent=4)




