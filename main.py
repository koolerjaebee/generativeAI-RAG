# Vector DB인 Milvus와 관련된 모듈들은 모듈의 용도를 명확히 구분하기 위해 Vector DB 구축 단계에서 불러왔으며 해당 부분에서 코드를 확인하실 수 있습니다.
import os
import argparse
import json
import time
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

if __name__ == '__main__':
    # ArgumentParser 인스턴스 생성
    parser = argparse.ArgumentParser()

    parser.add_argument('--init', action='store_true')
    parser.add_argument('--extension', type=str, default='docx')

    args = parser.parse_args()
    extension = args.extension
    langchain_loader_dict = {
        'docx': DocxReader,
        'pdf': PyMuPDFReader,
        'hwp': HWPReader,
    }

    if args.init:
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

        # Chunking
        chunking_executor = ChunkingExecutor(
            host='clovastudio.apigw.ntruss.com',
            api_key=API_KEY,
            api_key_primary_val=API_KEY_PRIMARY,
            request_id=str(uuid.uuid4())
        )

        chunked_data = []

        for loader in tqdm(loaders):
            chunking_request_data = {
                "postProcessMaxSize": 1000,
                "alpha": 0.0,
                "segCnt": -1,
                "postProcessMinSize": 300,
                "text": loader.text,  # 직접 값을 할당
                "postProcess": False
            }
            # chunking_request_data_json = json.dumps(
            #     chunking_request_data, ensure_ascii=False)
            response_text = chunking_executor.execute(
                chunking_request_data)
            for paragraph in response_text:
                chunked_document = {
                    "file_name": loader.metadata["file_name"],
                    "text": paragraph
                }
                chunked_data.append(chunked_document)

        print(len(chunked_data), chunked_data)

        # raise Exception("Test")

        # Embedding
        embedding_executor = EmbeddingExecutor(
            host='clovastudio.apigw.ntruss.com',
            api_key=API_KEY,
            api_key_primary_val=API_KEY_PRIMARY,
            request_id=str(uuid.uuid4())
        )

        try:
            for i, chunked_document in enumerate(tqdm(chunked_data)):
                chunking_request_data = {"text": str(chunked_document['text'])}
                embedding = embedding_executor.execute(
                    chunking_request_data)
                chunked_document["embedding"] = embedding
                time.sleep(1)
        except ValueError as e:
            print(f"Error at document index {i} with error: {e}")

        # 임베딩된 벡터들의 차원 확인
        dimension_set = set()

        for item in chunked_data:
            if "embedding" in item:
                dimension = len(item["embedding"])
                dimension_set.add(dimension)

        print("임베딩된 벡터들의 차원:", dimension_set)

        # Milvus 연결
        connections.connect()
        if COLLECTION_NAME in utility.list_collections():
            utility.drop_collection(COLLECTION_NAME)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64,
                        is_primary=True, auto_id=True),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR,
                        max_length=3000),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=9000),
            FieldSchema(name="embedding",
                        dtype=DataType.FLOAT_VECTOR, dim=1024)
        ]

        schema = CollectionSchema(fields, description=COLLECTION_DESCRIPTION)

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
            insert_result = collection.insert(entities)
            print("데이터 Insertion이 완료된 ID:", insert_result.primary_keys)

        print("데이터 Insertion이 전부 완료되었습니다")

    else:
        # Vector DB 로딩
        connections.connect("default", host="localhost", port="19530")

        # 불러올 collection 이름을 넣는 곳
        collection = Collection(COLLECTION_NAME)
        utility.load_state(COLLECTION_NAME)
