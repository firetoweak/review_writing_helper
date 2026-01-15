from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_project_outline():
    payload = {
        "title": "示例报告",
        "idea": "一些想法",
        "fullWriteRule": "第一章\n第二章",
        "industry": "互联网",
        "prompt": {"outlinePrompt": "请输出大纲"},
    }
    response = client.post("/api/project-outline", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "docGuide" in data
    assert "outline" in data


def test_outline_mapping():
    payload = {
        "sessionId": "map-1",
        "title": "示例报告",
        "idea": "一些想法",
        "outline": [
            {
                "nodeId": "1",
                "level": 1,
                "title": "章节一",
                "children": [
                    {"nodeId": "1.1", "level": 2, "title": "小节一"},
                    {"nodeId": "1.2", "level": 2, "title": "小节二"},
                ],
            }
        ],
    }
    response = client.post("/api/outline-mapping", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sessionId"] == "map-1"
    assert "neighbors" in data


def test_heuristic_start_streaming():
    payload = {
        "sectionWriteRule": "写作规则",
        "textList": [{"sectionId": "s1", "sectionTitle": "小节一", "text": "内容"}],
        "historyTextList": [
            {
                "chapterId": "c1",
                "chapterTitle": "章节一",
                "children": [{"sectionId": "s0", "sectionTitle": "前言", "text": "历史"}],
            }
        ],
        "sectionReviewRule": "评审规则",
        "industry": "制造",
        "title": "报告标题",
        "idea": "思路",
        "sessionId": "sess-1",
        "sectionTitle": "小节一",
        "prompt": {"heuristicWritingPrompt": "提问提示"},
    }
    response = client.post("/api/heuristic-writing", json=payload)
    assert response.status_code == 200
    assert "message.start" in response.text


def test_heuristic_message_non_stream():
    payload = {
        "sessionId": "sess-1",
        "Messages": [
            {"messageId": "m1", "role": "assistant", "type": "question", "content": "问题"},
            {"messageId": "m2", "role": "user", "type": "answer", "content": "回答"},
        ],
        "stream": "false",
    }
    response = client.post("/api/heuristic-writing/message", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sessionId"] == "sess-1"
    assert data["assistantMessage"]["role"] == "assistant"


def test_section_review():
    payload = {
        "textList": [{"sectionId": "s1", "sectionTitle": "小节", "text": "内容"}],
        "sectionWriteRule": "规则",
        "sectionReviewRule": "评审规则",
        "industry": "教育",
        "historyTextList": [],
        "title": "报告",
        "sectionTitle": "小节",
        "prompt": {"sectionReviewPrompt": "评审提示"},
    }
    response = client.post("/api/section-review", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "review" in data
    assert "score" in data["review"]


def test_full_review():
    payload = {
        "title": "报告",
        "reviews": [
            {
                "chapterId": "c1",
                "chapterTitle": "章节一",
                "review": {"score": 8, "evaluate": "好", "suggestion": "改", "to_do_list": []},
            }
        ],
        "fullReviewText": "全文内容",
        "prompt": {"fullReviewPrompt": "全文评审"},
    }
    response = client.post("/api/full-review", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fullReviewAns" in data


def test_help_chat_non_stream():
    payload = {
        "sessionId": "help-1",
        "textList": [{"sectionId": "s1", "sectionTitle": "小节", "text": "内容"}],
        "review": {"score": 7, "evaluate": "评估", "suggestion": "建议", "to_do_list": []},
        "writeRule": {"sectionWriteRule": "规则"},
        "helpText": "需要帮助",
        "prompt": {"helpPrompt": "帮助提示"},
        "messages": [{"messageId": "m1", "role": "user", "content": "问题"}],
        "stream": False,
    }
    response = client.post("/api/i-can/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sessionId"] == "help-1"
    assert "assistantMessage" in data


def test_help_chat_message_non_stream():
    payload = {
        "sessionId": "help-1",
        "messages": [{"messageId": "m1", "role": "user", "content": "问题"}],
        "stream": False,
    }
    response = client.post("/api/i-can/chat/message", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sessionId"] == "help-1"


def test_merge_texts():
    payload = {
        "writeRule": {"sectionWriteRule": "规则"},
        "textList": [
            {"sectionId": "s1", "sectionTitle": "小节1", "text": "内容1"},
            {"sectionId": "s2", "sectionTitle": "小节2", "text": "内容2"},
        ],
        "sessionList": [
            {
                "sessionId": "sess-1",
                "messages": [
                    {"messageId": "m1", "role": "assistant", "content": "建议"}
                ],
            }
        ],
        "historyTextList": [],
        "review": {"score": 6, "evaluate": "评估", "suggestion": "建议", "to_do_list": []},
        "industry": "金融",
        "prompt": {"mergePrompt": "合并提示"},
    }
    response = client.post("/api/merge", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "text" in data


def test_full_polish():
    payload = {
        "task": "fullPolish",
        "fullText": [
            {
                "chapterId": "c1",
                "chapterTitle": "章节一",
                "children": [
                    {"sectionId": "s1", "sectionTitle": "小节", "text": "内容"}
                ],
            }
        ],
        "polishPrompt": "润色提示",
        "stream": False,
    }
    response = client.post("/api/full-polish", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "newFullText" in data


def test_text_restruct():
    payload = {
        "task": "restruct",
        "file_path": "./dummy.txt",
        "restructPrompt": "重组提示",
        "outlinePrompt": "大纲提示",
        "industry": "制造",
    }
    response = client.post("/api/text-restruct", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "docGuide" in data
    assert "outline" in data
    assert "fullText" in data


def test_kb_documents():
    payload = {
        "action": "index",
        "projectId": "proj-1",
        "document_id": "doc-1",
        "file_url": "http://example.com/file",
        "filename": "file.pdf",
    }
    response = client.post("/api/kb/documents", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "doc-1"
    assert data["status"] == "indexed"
