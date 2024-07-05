# Vector DB인 Milvus와 관련된 모듈들은 모듈의 용도를 명확히 구분하기 위해 Vector DB 구축 단계에서 불러왔으며 해당 부분에서 코드를 확인하실 수 있습니다.
import os
import argparse
import json
import time
import logging
import pickle
import sys
from langchain_community.document_loaders import Docx2txtLoader, PyMuPDFLoader
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import (
    DocxReader,
    HWPReader,
    PyMuPDFReader,
)
from pathlib import Path
from tqdm import tqdm
import uuid
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

from env import API_KEY, API_KEY_PRIMARY, COLLECTION_DESCRIPTION, COLLECTION_NAME
from func.chunking import ChunkingExecutor
from func.embedding import EmbeddingExecutor
from func.retrieval import CompletionExecutor


# 로그 설정
logging.basicConfig(filename='./logs/app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    logging.info("프로그램 시작")
    # ArgumentParser 인스턴스 생성
    parser = argparse.ArgumentParser()

    parser.add_argument('--init', action='store_true')
    parser.add_argument('--table', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--pickle', action='store_true')
    parser.add_argument('--extension', type=str, default='docx')

    args = parser.parse_args()
    extension = args.extension
    langchain_loader_dict = {
        'docx': DocxReader,
        'pdf': PyMuPDFReader,
        'hwp': HWPReader,
    }
    chunking_executor = ChunkingExecutor(
        host='clovastudio.apigw.ntruss.com',
        api_key=API_KEY,
        api_key_primary_val=API_KEY_PRIMARY,
        request_id=str(uuid.uuid4())
    )

    embedding_executor = EmbeddingExecutor(
        host='clovastudio.apigw.ntruss.com',
        api_key=API_KEY,
        api_key_primary_val=API_KEY_PRIMARY,
        request_id=str(uuid.uuid4())
    )

    # 사용자의 쿼리를 임베딩하는 함수를 먼저 정의
    def query_embed(text: str):
        request_data = {"text": text}
        response_data = embedding_executor.execute(request_data)
        return response_data

    def docx_chat(realquery: str, collection) -> str:
        # 사용자 쿼리 벡터화
        query_vector = query_embed(realquery)

        collection.load()

        search_params = {"metric_type": "IP", "params": {"ef": 64}}
        results = collection.search(
            data=[query_vector],  # 검색할 벡터 데이터
            anns_field="embedding",  # 검색을 수행할 벡터 필드 지정
            param=search_params,
            limit=10,
            output_fields=["file_name", "text"]
        )

        reference = []

        for hit in results[0]:
            distance = hit.distance
            file_name = hit.entity.get("file_name")
            text = hit.entity.get("text")
            reference.append(
                {"distance": distance, "file_name": file_name, "text": text})

        completion_executor = CompletionExecutor(
            host="https://clovastudio.stream.ntruss.com",
            api_key=API_KEY,
            api_key_primary_val=API_KEY_PRIMARY,
            request_id=str(uuid.uuid4())
        )

        text_template = "- 아래와 같은 형식으로 구성된 답변을 작성해 줘\n## {테이블 타이틀} \n{테이블 타이틀 설명하는 문장} {column 또는 row 에 따른 통계적 내용의 분석 문장들}\n- 아래는 답변 예시중에 하나야.\n## 현재 음주율\n현재 음주율은 최근 30일 동안 1잔 이상 술을 마신 적이 있는 사람의 비율을 의미합니다. 전체 청소년 중 현재 음주율은 2.5%이며, 음주 경험이 있는 청소년 중 현재 음주율은 21.7%입니다. 남자 청소년과 여자 청소년의 현재 음주율을 비교하면 전체 청소년 중 남자는 2.9%, 여자는 2.1%로 남자의 비율이 높습니다. 그러나 음주 경험자 중 현재 음주율을 보면 남자가 20.4%, 여자가 23.1%로 여자 청소년의 현재 음주율이 더 높습니다. 중학생과 고등학생 청소년의 경우 고등학생의 현재 음주율은 25.6%, 중학생의 현재 음주율은 18.9%로 고등학생이 더 높습니다. 고양시 세 개 구별로 비교한 결과, 일산서구의 현재 음주율이 27.3%로 가장 높았고, 다음으로 일산동구가 23.3%, 덕양구가 14.3% 순이었습니다.\n- 힌트: 형식을 직접적으로 복사하여 붙이지 마세요. 제공된 데이터를 이해하고 이를 기반으로 각 정보를 적절히 문장으로 구성해주세요."

        preset_texts = [
            {
                "role": "system",
                "content": "- 너의 역할은 사용자의 질문에 reference를 바탕으로 답변하는거야. \n- 너가 가지고있는 지식은 모두 배제하고, 주어진 reference의 내용만을 바탕으로 답변해야해. \n- 답변의 출처가 되는 docx의 내용인 'file_name'도 답변과 함께 {file_name:}의 형태로 제공해야해. \n- 만약 사용자의 질문이 reference와 관련이 없다면, {제가 가지고 있는 정보로는 답변할 수 없습니다.}라고만 반드시 말해야해."
            }
        ]

        for ref in reference:
            preset_texts.append(
                {
                    "role": "system",
                    "content": f"reference: {ref['text']}, file_name: {ref['file_name']}"
                }
            )

        preset_texts.append({"role": "user", "content": realquery})

        request_data = {
            "messages": preset_texts,
            "topP": 0.9,
            "topK": 50,
            "maxTokens": 1024,
            "temperature": 0.7,
            "repeatPenalty": 1.2,
            "stopBefore": [],
            "includeAiFilters": False
        }

        # LLM 생성 답변 반환
        response_data = completion_executor.execute(request_data)

        return response_data

    try:
        if args.init:
            if not args.pickle:
                logging.info("--init 플래그와 함께 실행됨")
                # .docx 파일을 로드하기 위한 리스트 초기화
                loaders = []

                # './data' 폴더 내의 모든 파일을 순회
                for filename in os.listdir('./data'):
                    if filename.endswith(f'.{extension}'):
                        # 파일 경로 생성
                        file_path = os.path.join('./data', filename)
                        # Docx2txtLoader 인스턴스 생성 및 리스트에 추가
                        loader = langchain_loader_dict['docx']()
                        document = loader.load_data(file=file_path)[0]
                        loaders.append(document)

                # loaders 리스트에 로드된 내용 확인
                print(len(loaders), loaders)
                logging.info("로드 완료")

                # Chunking
                chunked_data = []

                for loader in tqdm(loaders):
                    chunking_request_data = {
                        "postProcessMaxSize": 100,
                        "alpha": 1.5,
                        "segCnt": -1,
                        "postProcessMinSize": 0,
                        "text": loader.text,  # 직접 값을 할당
                        "postProcess": False
                    }
                    while True:
                        response_text = chunking_executor.execute(
                            chunking_request_data)
                        time.sleep(1)
                        if response_text != 'Error':
                            break
                        else:
                            print("Chunking Error 발생")
                            logging.error("Chunking Error 발생")

                    for paragraph in response_text:
                        chunked_document = {
                            "file_name": loader.metadata["file_name"],
                            "text": paragraph
                        }
                        chunked_data.append(chunked_document)

                print(len(chunked_data), chunked_data)
                logging.info("Chunking 완료")

                # raise Exception("Test")

                # Embedding

                try:
                    for i, chunked_document in enumerate(tqdm(chunked_data)):
                        while True:
                            chunking_request_data = {
                                "text": str(chunked_document['text'])}
                            embedding = embedding_executor.execute(
                                chunking_request_data)
                            chunked_document["embedding"] = embedding
                            time.sleep(1)
                            if embedding != 'Error':
                                break
                            else:
                                print("Embedding Error 발생")
                                logging.error("Embedding Error 발생")
                except ValueError as e:
                    print(f"Error at document index {i} with error: {e}")
                    logging.error(
                        f"Error at document index {i} with error: {e}")
                logging.info("처리 완료")

                # 임베딩된 벡터들의 차원 확인
                dimension_set = set()

                for item in chunked_data:
                    if "embedding" in item:
                        dimension = len(item["embedding"])
                        dimension_set.add(dimension)

                print("임베딩된 벡터들의 차원:", dimension_set)
                logging.info(f"임베딩된 벡터들의 차원: {dimension_set}")

                # 데이터 저장
                with open('chunked_data.pkl', 'wb') as f:
                    pickle.dump(chunked_data, f)
                logging.info("chunked_data 저장 완료")
            else:
                # chunked_data 로드
                with open('chunked_data.pkl', 'rb') as f:
                    chunked_data = pickle.load(f)
                logging.info("chunked_data 로드 완료")

            # Vector DB 로딩
            connections.connect("default", host="localhost", port="19530")
            if COLLECTION_NAME in utility.list_collections():
                utility.drop_collection(COLLECTION_NAME)

            fields = [
                FieldSchema(name="id", dtype=DataType.INT64,
                            is_primary=True, auto_id=True),
                FieldSchema(name="file_name", dtype=DataType.VARCHAR,
                            max_length=3000),
                FieldSchema(name="text", dtype=DataType.VARCHAR,
                            max_length=9000),
                FieldSchema(name="embedding",
                            dtype=DataType.FLOAT_VECTOR, dim=1024)
            ]

            schema = CollectionSchema(
                fields, description=COLLECTION_DESCRIPTION)

            collection = Collection(name=COLLECTION_NAME,
                                    schema=schema, using='default', shards_num=2)

            for item in chunked_data:
                item['text'] = ", ".join(item['text'])

                file_name_list = [item['file_name']]
                text_list = [item['text']]
                embedding_list = [item['embedding']]

                entities = [
                    file_name_list,
                    text_list,
                    embedding_list
                ]
                if (len(item['embedding']) != 1024):
                    print("Error at", len(item['embedding']))
                    logging.error(f"Error at {item['file_name']}")
                    continue
                insert_result = collection.insert(entities)
                print("데이터 Insertion이 완료된 ID:", insert_result.primary_keys)
            print("데이터 Insertion이 전부 완료되었습니다")
            logging.info("데이터 Insertion이 전부 완료되었습니다")

            # 불러올 collection 이름을 넣는 곳
            collection = Collection(COLLECTION_NAME)
            utility.load_state(COLLECTION_NAME)
            print(f"{COLLECTION_NAME} 이 성공적으로 로드되었습니다")

            # Indexing
            index_params = {
                "metric_type": "IP",
                "index_type": "HNSW",
                "params": {
                    "M": 8,
                    "efConstruction": 200
                }
            }
            collection.create_index(
                field_name="embedding", index_params=index_params)
            utility.index_building_progress(COLLECTION_NAME)

            print([index.params for index in collection.indexes])
            print("Indexing 완료")
            logging.info("Indexing 완료")

        else:
            connections.connect("default", host="localhost", port="19530")

            # 불러올 collection 이름을 넣는 곳
            collection = Collection(COLLECTION_NAME)
            utility.load_state(COLLECTION_NAME)
            while True:
                if args.table:
                    import re
                    table_num = input("분석하고 싶은 테이블 번호를 입력해주세요: ")
                    try:
                        table_num = int(table_num)
                    except ValueError:
                        print("숫자만 입력해주세요")
                        continue
                    realquery = f"테이블 {table_num} (Table {table_num}) 에 대해 설명해주세요.\n\n정확한 수치를 바탕으로 답변해주세요."
                    answer = docx_chat(realquery, collection)
                    print(answer)
                    if args.save and not answer == "제가 가지고 있는 정보로는 답변할 수 없습니다.":
                        with open(f"./output/table_{table_num}.txt", "w") as f:
                            f.write(answer)
                else:
                    realquery = input("챗봇에게 물어볼 내용을 입력해주세요: ")
                    answer = docx_chat(realquery, collection)
                    print(answer)
                    if args.save and not answer == "제가 가지고 있는 정보로는 답변할 수 없습니다.":
                        with open(f"./output/answer.txt", "w") as f:
                            f.write(answer)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = os.path.split(exc_traceback.tb_frame.f_code.co_filename)[1]
        print("Error:", e)
        print("Filename", fname)
        print("Type", exc_type)
        print("Line_number", exc_traceback.tb_lineno)
        logging.error(f"{e}")
        logging.error(f"\tFilename {fname}")
        logging.error(f"\tType {exc_type}")
        logging.error(f"\tLine_number {exc_traceback.tb_lineno}")
