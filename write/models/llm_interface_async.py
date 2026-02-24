from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config import settings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


# 你现在的标签格式： [IMAGE_1], [IMAGE_2] ...
_IMAGE_TAG_RE = re.compile(r"\[IMAGE_\d+\]")

SYS_PROMPT_TEMPLATE ="Given the following conversation, relevant context, and a follow up question, reply with an answer to the current question the user is asking. Return only your response to the question given the above information following the users instructions as needed."
RULE_SYS_PROMPT = "\n**所有的回答中，不要出现：“根据xx文档要求”或“按照资料x”等僵化描述类文字。**"


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    timeout_s: int = 300
    temperature: float = 0.7
    max_tokens: int =  10000


def get_llm_config() -> LLMConfig:
    route = settings.model_route
    provider = route.qwen if route.default_provider == "qwen" else route.local
    return LLMConfig(
        base_url=provider.base_url,
        api_key=provider.api_key or "EMPTY",
        model=provider.model,
    )


def get_vlm_config() -> LLMConfig:
    return LLMConfig(
        base_url=settings.chatvlm.base_url,
        api_key=settings.chatvlm.api_key or "EMPTY",
        model=settings.chatvlm.model,
    )


def is_llm_configured() -> bool:
    return bool(get_llm_config().base_url)


def is_vlm_configured() -> bool:
    return bool(get_vlm_config().base_url)


def build_chat_model(
    *, streaming: bool, multimodal: bool = True, model_name: Optional[str] = None
) -> ChatOpenAI:
    if multimodal:
        model = settings.chatvlm
        if model_name == "vlm_a3b_Thinking":
            model = settings.models["vlm_a3b_Thinking"]
        cfg = LLMConfig(
            base_url=model.base_url,
            api_key=model.api_key or "EMPTY",
            model=model.model,
        )
    else:
        cfg = get_llm_config()

    return ChatOpenAI(
        model=cfg.model,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        timeout=cfg.timeout_s,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        streaming=streaming,
        extra_body={
        "chat_template_kwargs": {"enable_thinking": True},
        "top_k": 20,            # 可选：Qwen推荐 thinking 时配合 top_k
        }
    )


# ✅ 新增：找出文本里出现的 tags
def _extract_image_tags(text: str) -> List[str]:
    return [m.group(0) for m in _IMAGE_TAG_RE.finditer(text or "")]

# ✅ 新增：只有“文本tags ∩ image_map keys”非空，才认为需要多模态渲染
def _should_render_multimodal(text: str, image_map: Optional[Dict[str, str]]) -> bool:
    if not text:
        return False
    if not image_map:
        return False
    tags = set(_extract_image_tags(text))
    if not tags:
        print("text 里没有 IMAGE_TAG:", text)
        return False
    
    keys = set((image_map or {}).keys())
    return bool(tags & keys)


def _render_multimodal_blocks(
    prompt: str,
    image_map: Dict[str, str],
    *,
    strict_missing: bool = True,   # ✅ 可选：严格缺失映射就报错；否则当作普通文本保留
) -> List[Dict[str, Any]]:
    """
    把含 [IMAGE_n] 的 prompt 切成多模态 blocks。
    只替换 image_map 里存在映射的 tags；对缺失映射的 tag：
    - strict_missing=True  -> 报错
    - strict_missing=False -> 当普通文本保留
    """
    print("=======图片已经载入分析!=======")

    blocks: List[Dict[str, Any]] = []
    last_end = 0

    def _is_valid_media_url(u: str) -> bool:
        u = (u or "").strip()
        if not u:
            return False
        if u.startswith(("http://", "https://")):
            return True
        if u.startswith("data:image/"):
            return True
        return False


    for match in _IMAGE_TAG_RE.finditer(prompt or ""):
        tag = match.group(0)
        start, end = match.span()

        # 前置文本
        if start > last_end:
            text_seg = prompt[last_end:start]
            if text_seg:
                blocks.append({"type": "text", "text": text_seg})

        url = (image_map.get(tag) or "").strip()
        if _is_valid_media_url(url):
            blocks.append({"type": "image_url", "image_url": {"url": url}})
        else:
            if strict_missing:
                raise ValueError(f"Invalid or missing image url mapping for tag: {tag}")
            blocks.append({"type": "text", "text": tag})


        last_end = end

    # 尾部文本
    if last_end < len(prompt or ""):
        tail = (prompt or "")[last_end:]
        if tail:
            blocks.append({"type": "text", "text": tail})

    if not blocks:
        blocks = [{"type": "text", "text": prompt or ""}]

    return blocks


def build_messages(
    *,
    system_prompt: Optional[str],
    user_text: str,
    messages: Optional[List[Dict[str, Any]]] = None,
    image_map: Optional[Dict[str, str]] = None,
    context_text: Optional[str] = None,
    context_image_map: Optional[Dict[str, str]] = None,
    # ✅ 可选：控制缺失映射策略（默认宽松，避免“解释说明”误伤）
    strict_missing_image_map: bool = False,
) -> List[BaseMessage]:
    out: List[BaseMessage] = []
    # print("user_text:", user_text)
    # print("len(user_text)", len(user_text))
    # print("len(system_prompt)", len(system_prompt or ""))
    # print("system_prompt:", system_prompt)
    # print("messages:", messages)

    if not system_prompt:
        system_prompt = SYS_PROMPT_TEMPLATE
    system_prompt = system_prompt + RULE_SYS_PROMPT    

    out.append(SystemMessage(content=system_prompt))

    # ✅ context：只有在确实能匹配到映射时才走多模态
    if context_text is not None:
        if _should_render_multimodal(context_text or "", context_image_map):
            blocks_ctx = _render_multimodal_blocks(
                context_text or "",
                context_image_map or {},
                strict_missing=strict_missing_image_map,
            )
            out.append(HumanMessage(content=blocks_ctx))
        else:
            out.append(HumanMessage(content=context_text or ""))

    # 历史对话
    for msg in messages or []:
        role = (msg.get("role") or "user").lower()
        content = msg.get("content", "")
        if role == "assistant":
            out.append(AIMessage(content=content))
        else:
            out.append(HumanMessage(content=content))

    # ✅ 当前轮：同理，只在“确实有可替换的tag”时才走多模态
    if _should_render_multimodal(user_text or "", image_map):
        blocks = _render_multimodal_blocks(
            user_text or "",
            image_map or {},
            strict_missing=strict_missing_image_map,
        )
        out.append(HumanMessage(content=blocks))
    else:
        out.append(HumanMessage(content=user_text or ""))

    return out



def extract_image_map_from_text_list(text_list: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    示例 item：
      {
        "text": "...",
        "image_url": {"[IMAGE_1]": "https://...", "[IMAGE_2]": "https://..."}
      }
    """
    image_map: Dict[str, str] = {}
    for item in text_list or []:
        attachments = item.get("image_url") or {}
        if isinstance(attachments, dict):
            for k, v in attachments.items():
                if k and v:
                    image_map[str(k)] = str(v)
    return image_map
