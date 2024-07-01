# Vector DB인 Milvus와 관련된 모듈들은 모듈의 용도를 명확히 구분하기 위해 Vector DB 구축 단계에서 불러왔으며 해당 부분에서 코드를 확인하실 수 있습니다.

# -*- coding: utf-8 -*-

import json
import http.client
import uuid
from env import API_KEY, API_KEY_PRIMARY, CHUNKING_API_URL


class ChunkingExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_id):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id

    def _send_request(self, completion_request):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id
        }
        headers = {k: str(v) for k, v in headers.items()}

        conn = http.client.HTTPSConnection(self._host)
        conn.request('POST', CHUNKING_API_URL,
                     json.dumps(completion_request), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        return result

    def execute(self, completion_request):
        res = self._send_request(completion_request)
        if res['status']['code'] == '20000':
            return res['result']['topicSeg']
        else:
            print(res)
            print(res['status']['code'])
            return 'Error'


if __name__ == '__main__':
    completion_executor = ChunkingExecutor(
        host='clovastudio.apigw.ntruss.com',
        api_key=API_KEY,
        api_key_primary_val=API_KEY_PRIMARY,
        request_id=str(uuid.uuid4())
    )

    request_data = json.loads("""{
  "postProcessMaxSize" : 1000,
  "alpha" : 0.0,
  "segCnt" : -1,
  "postProcessMinSize" : 300,
  "text" : "input text",
  "postProcess" : false
}""", strict=False)

    response_text = completion_executor.execute(request_data)
    print(request_data)
    print(response_text)
