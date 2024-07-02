# -*- coding: utf-8 -*-

import json
import requests
import uuid

from env import API_KEY, API_KEY_PRIMARY, EMBEDDING_API_URL


class CompletionExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_id):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id

    def execute(self, completion_request, response_type="stream"):
        headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'text/event-stream'
        }

        # # Default
        # with requests.post(self._host + '/testapp/v1/chat-completions/HCX-DASH-001',
        #                    headers=headers, json=completion_request, stream=True) as r:
        #     for line in r.iter_lines():
        #         if line:
        #             print(line.decode("utf-8"))

        # Longest Answer
        final_answer = ""

        with requests.post(
            self._host + "/testapp/v1/chat-completions/HCX-003",
            headers=headers,
            json=completion_request,
            stream=True
        ) as r:
            if response_type == "stream":
                longest_line = ""
                for line in r.iter_lines():
                    if line:
                        decoded_line = line.decode("utf-8")
                        if decoded_line.startswith("data:"):
                            event_data = json.loads(
                                decoded_line[len("data:"):])
                            message_content = event_data.get(
                                "message", {}).get("content", "")
                            if len(message_content) > len(longest_line):
                                longest_line = message_content
                final_answer = longest_line
            elif response_type == "single":
                final_answer = r.json()  # 가정: 단일 응답이 JSON 형태로 반환됨
        return final_answer


if __name__ == '__main__':
    completion_executor = CompletionExecutor(
        host='https://clovastudio.stream.ntruss.com',
        api_key=API_KEY,
        api_key_primary_val=API_KEY_PRIMARY,
        request_id=str(uuid.uuid4())
    )

    preset_text = [{"role": "system", "content": ""},
                   {"role": "user", "content": ""}]

    request_data = {
        'messages': preset_text,
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 256,
        'temperature': 0.5,
        'repeatPenalty': 5.0,
        'stopBefore': [],
        'includeAiFilters': True,
        'seed': 0
    }

    print(preset_text)
    completion_executor.execute(request_data)
