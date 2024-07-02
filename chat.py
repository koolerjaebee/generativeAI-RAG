import argparse
import uuid

from main import embedding_executor, collection
from func.retrieval import CompletionExecutor
from env import API_KEY, API_KEY_PRIMARY, EMBEDDING_API_URL

# 사용자의 쿼리를 임베딩하는 함수를 먼저 정의


def query_embed(text: str):
    request_data = {"text": text}
    response_data = embedding_executor.execute(request_data)
    return response_data


def html_chat(realquery: str) -> str:
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
        "topP": 0.6,
        "topK": 0,
        "maxTokens": 1024,
        "temperature": 0.5,
        "repeatPenalty": 1.2,
        "stopBefore": [],
        "includeAiFilters": False
    }

    # LLM 생성 답변 반환
    response_data = completion_executor.execute(request_data)

    return response_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--table', action='store_true')
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()
    while True:
        if args.table:
            table_num = input("분석하고 싶은 테이블 번호를 입력해주세요: ")
            realquery = f"테이블 {table_num} (Table {table_num}) 에 대해 알려주세요.\n정확한 수치를 바탕으로 답변해주세요."
            answer = html_chat(realquery)
            print(answer)
            if args.save:
                with open(f"./output/table_{table_num}.txt", "w") as f:
                    f.write(answer)
        else:
            realquery = input("챗봇에게 물어볼 내용을 입력해주세요: ")
            answer = html_chat(realquery)
            print(answer)
            if args.save:
                with open(f"./output/answer.txt", "w") as f:
                    f.write(answer)
