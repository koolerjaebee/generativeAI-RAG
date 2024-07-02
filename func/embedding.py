# -*- coding: utf-8 -*-

import json
import http.client
import uuid

from env import API_KEY, API_KEY_PRIMARY, EMBEDDING_API_URL


class EmbeddingExecutor:
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

        conn = http.client.HTTPSConnection(self._host, timeout=3600)
        conn.request('POST', EMBEDDING_API_URL,
                     json.dumps(completion_request), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        return result

    def execute(self, completion_request):
        res = self._send_request(completion_request)
        if res['status']['code'] == '20000':
            return res['result']['embedding']
        else:
            print(res)
            return 'Error'


if __name__ == '__main__':
    completion_executor = EmbeddingExecutor(
        host='clovastudio.apigw.ntruss.com',
        api_key=API_KEY,
        api_key_primary_val=API_KEY_PRIMARY,
        request_id=str(uuid.uuid4())
    )

    request_data = json.loads("""{
  "text" : "input text"
}""", strict=False)

    response_text = completion_executor.execute(request_data)
    print(request_data)
    print(response_text)
