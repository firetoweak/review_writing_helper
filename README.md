# review_writing_helper
写作评审助手

.
├── config.py
├── config.yaml
├── main.py
├── models
│   ├── chat_db.py
│   └── llm_interface_async.py
├── README.md
├── routers
│   ├── help.py
│   ├── heuristic.py
│   ├── kb.py
│   ├── merge.py
│   ├── outline.py
│   ├── polish.py
│   ├── review.py
│   └── writing.py
└── services
    ├── ai_client.py
    ├── contracts.py    # service 接口契约（Protocol/ABC）
    ├── factory.py      # 根据配置选择 Fake 或 Real
    ├── fake
    │   ├── help.py
    │   ├── __init__.py
    │   ├── kb.py
    │   ├── merge.py
    │   ├── review.py
    │   └── writing.py
    ├── kb_client.py
    ├── review_service.py
    └── writing_service.py


# 智能写作助手agent端侧
