import os
import argparse
import uuid
import time
import pickle
from tqdm import tqdm
from langchain_community.document_loaders import Docx2txtLoader, PyMuPDFLoader, UnstructuredHTMLLoader
from llama_index.readers.file import (
    DocxReader,
    HWPReader,
    PyMuPDFReader,
)
import mammoth

from func.chunking import ChunkingExecutor
from env import API_KEY, API_KEY_PRIMARY

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


def docx_to_html(docx_path):
    with open(docx_path, 'rb') as docx_file:
        result = mammoth.convert_to_html(docx_file)
        output = result.value
        messages = result.messages
        # print(messages)
    html_path = docx_path.replace('.docx', '.html')
    with open(html_path, 'w') as f:
        f.write(output)
    return html_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pickle', action='store_true')

    args = parser.parse_args()

    if not args.pickle:
        loaders = []

        # './data' 폴더 내의 모든 파일을 순회
        for filename in os.listdir('./data'):
            if filename.endswith(f'.docx'):
                # 파일 경로 생성
                file_path = os.path.join('./data', filename)
                docx_to_html(file_path)
                loader = UnstructuredHTMLLoader(
                    file_path=file_path.replace('.docx', '.html'))
                documents = loader.load()
        # 각 문서의 "source" 메타데이터 수정
        for document in documents:
            if 'source' in document.metadata:
                document.metadata['source'] = document.metadata['source'].replace(
                    './data/', '')
                loaders.extend(documents)

        # loaders 리스트에 로드된 내용 확인
        print(len(loaders), loaders)

        with open("./output/loaders.txt", "w") as f:
            for i, loader in enumerate(loaders):
                source = loaders[i - 1].metadata["source"] if i > 0 else None
                if source != loader.metadata["source"]:
                    f.write(loader.metadata["source"] + "\n")
                f.write(f"Document {i + 1}\n")
                f.write(f"Content: {loader.page_content}\n")
                f.write("\n")

        # Chunking
        chunked_data = []

        for loader in tqdm(loaders):
            chunking_request_data = {
                "postProcessMaxSize": 1000,  # 100
                "alpha": 0.1,  # 1.5
                "segCnt": -1,
                "postProcessMinSize": 0,
                "text": loader.page_content,  # 직접 값을 할당
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

            for paragraph in response_text:
                chunked_document = {
                    "source": loader.metadata["source"],
                    "text": paragraph
                }
                chunked_data.append(chunked_document)

        with open('./output/chunked_data.pkl', 'wb') as f:
            pickle.dump(chunked_data, f)
        print(len(chunked_data), chunked_data)
    else:
        with open('./output/chunked_data.pkl', 'rb') as f:
            chunked_data = pickle.load(f)
        print(len(chunked_data), chunked_data)

    # Chunked Data 저장
    with open("./output/chunked_data.txt", "w") as f:
        for i, chunked_document in enumerate(chunked_data):
            source = chunked_data[i - 1]["source"] if i > 0 else None
            if source != chunked_document["source"]:
                f.write(chunked_document["source"] + "\n")
            f.write(f"Paragraph {i + 1}\n")
            for line in chunked_document["text"]:
                f.write(line + "\n")
            f.write("\n")
