from __future__ import annotations

import json
import regex as re
from typing import Dict, List, Any, Optional

from json_repair import repair_json

from models.llm_interface_async import (
    build_chat_model,
    build_messages,
    is_llm_configured,
    is_vlm_configured,
)

from tools.prompt_templating import (
    build_ctx,
    build_image_map,
    render_prompt,
    DEFAULT_PLACEHOLDER_MAP,
    prompt_has_image_tags,
)

LEVELS = [
    {"name": "<6", "mid": 5.5, "labels": "Level_1"},
    {"name": "6-8", "mid": 7.5, "labels": "Level_2"},
    {"name": "8-9", "mid": 8.5, "labels": "Level_3"},
    {"name": "9-10", "mid": 9.5, "labels": "Level_4"},
]


# Level_1 -> 0, Level_2 -> 1 ...
LABEL_TO_IDX = {lv["labels"]: i for i, lv in enumerate(LEVELS)}
MIDS = [lv["mid"] for lv in LEVELS]

def normalize_level(s: Any) -> Optional[str]:
    """
    把 'Level 1' / 'level-1' / 'LEVEL_1' 统一成 'Level_1'
    若无法识别，返回 None
    """
    if not isinstance(s, str):
        return None
    t = s.strip()
    if not t:
        return None

    # 提取末尾数字 1~4
    m = re.search(r"(?i)level\D*([1-4])", t)
    if not m:
        return None
    n = m.group(1)
    return f"Level_{n}"

def score_to_level_idx(score: float) -> int:
    """按分数区间映射到 Level_1~4 的 idx(0~3)"""
    if score < 6:
        return 0
    if score < 8:
        return 1
    if score < 9:
        return 2
    return 3


def correct_real_score(review: Dict[str, Any]) -> Dict[str, Any]:
    """
    只允许“向下纠偏”：
      - 若 business_level_idx < real_score_level_idx，则 real_score = mid[business_level_idx]
      - 否则不修正（包括：分数低但评级高）
    输出：
      {
        "review": { ... }  # 仅 real_score 可能变化
      }
    """
    score_obj = review.get("score") or {}

    origin_real = score_obj.get("real_score")
    level_norm = normalize_level(score_obj.get("business_level"))
    business_idx = LABEL_TO_IDX.get(level_norm) if level_norm else None

    final_real = origin_real

    # 只有 business_level 可识别且 real_score 可解析为数字时才做区间对比
    if business_idx is not None:
        try:
            real_val = float(origin_real)
            real_idx = score_to_level_idx(real_val)

            # 关键逻辑：只在“评级更差(更低)”且“分数更高(等级更高)”时下调
            if business_idx < real_idx:
                final_real = MIDS[business_idx]
        except (TypeError, ValueError):
            # real_score 不是数值：无法判断“分数等级”，这里默认不修正
            final_real = origin_real
    ideal_score = score_obj.get("ideal_score")

    # 输出：只覆盖 real_score
    out_review = dict(review)
    out_score = dict(score_obj)
    out_score["real_score"] = final_real
    out_score["ideal_score"] = "" if re.match(r"^\[ *\]$", ideal_score) else ideal_score
    out_review["score"] = out_score

    return out_review


class ReviewAgent:
    # -------------------------
    # public
    # -------------------------
    async def review(self, payload: Dict) -> Dict:
        prompt_obj = payload.get("prompt") or {}
        review_prompt = prompt_obj.get("chapterReviewPrompt") or prompt_obj.get("sectionReviewPrompt")  or ""

        ctx = build_ctx(payload)
        prompt = render_prompt(
            review_prompt,
            ctx,
            DEFAULT_PLACEHOLDER_MAP,
            keep_unknown=True,
        )
        print("======评审prompt：", prompt)

        image_map = build_image_map(payload)
        review = await self._run_review_once(
            prompt=prompt,
            image_map=image_map
        )
        if not isinstance(review.get("score"), dict):
            review = {
                "score": {
                    "real_score": review.get("score", ""),
                    "ideal_score": "",
                    "business_level": "",
                    "business_describe": "",
                },
                "evaluate": review.get("evaluate", ""),
                "suggestion": review.get("suggestion", []),
                "to_do_list": review.get("to_do_list", []),
            }
            print("没有修正评审结果", review)
            return {"review": review}
        review = correct_real_score(review)
        print("最终输出", review)
        return {"review": review}

    def full_review(self, payload: Dict) -> Dict:
        prompt_obj = payload.get("prompt") or {}
        review_prompt = prompt_obj.get("fullReviewPrompt") or ""

        ctx = build_ctx(payload)
        prompt = render_prompt(
            review_prompt,
            ctx,
            DEFAULT_PLACEHOLDER_MAP,
            keep_unknown=True,
        )
        model = build_chat_model(streaming=False, multimodal=True)

        lc_messages = build_messages(
            system_prompt=None,
            user_text=prompt,
            messages=None,
            image_map=None,
        )

        result = model.invoke(lc_messages)
        content = getattr(result, "content", "") or "{}"
        return {"fullReviewAns": content}

    # -------------------------
    # core multimodal runner
    # -------------------------
    async def _run_review_once(
        self,
        *,
        prompt: str,
        image_map: Dict[str, str],
        user_input: str = "",
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        has_tags = prompt_has_image_tags(prompt)
        return await self._llm_review(
            prompt=prompt,
            image_map=image_map if has_tags else None,
            user_input=user_input,
        )

    async def _llm_review(
        self,
        *,
        prompt: str,
        image_map: Optional[Dict[str, str]] = None,
        user_input: str = "",
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        image_map = image_map or {}

        has_tags = prompt_has_image_tags(prompt)

        # ✅ 选择模型：有标签走 VLM；无标签优先 LLM（若没配 LLM 则退到 VLM）
        use_multimodal = has_tags or (not is_llm_configured() and is_vlm_configured())
        model = build_chat_model(streaming=False, multimodal=use_multimodal)

        # ✅ 关键点：把最终 prompt 放进 user_text，system_prompt 置空
        # 因为 build_messages() 只会在 user_text 中识别 [IMAGE_n] 并切成多模态 blocks
        lc_messages = build_messages(
            system_prompt=None,
            user_text="\n\n".join([x for x in [prompt, user_input] if x]),
            messages=None,
            image_map=image_map if has_tags else None,
        )

        result = await model.ainvoke(lc_messages)
        content = getattr(result, "content", "") or "{}"
        print("===================原始模型输出======================================", content)
        try:
            data = repair_json(content, return_objects=True)
        except json.JSONDecodeError:
            return self._fallback_review(full_text=prompt, prompt=prompt)

        return data if isinstance(data, dict) else self._fallback_review(full_text=prompt, prompt=prompt)

    # -------------------------
    # fallback
    # -------------------------
    def _summary_from_prompt(self, prompt: str) -> List[str]:
        if not prompt:
            return []
        lines = [line.strip() for line in prompt.splitlines() if line.strip()]
        if lines:
            return lines[:3]
        return [prompt.strip()] if prompt.strip() else []

    def _fallback_review(self, title: str = "", full_text: str = "", prompt: str = "") -> Dict[str, Any]:
        score = min(10, max(1, len(full_text) // 120 + 3))
        summary = self._summary_from_prompt(prompt)
        detail = prompt or f"已接收《{title}》评审请求。"
        help_list = self._summary_from_prompt(prompt)
        return {
            "score": score,
            "evaluate": summary,
            "suggestion": detail,
            "to_do_list": help_list,
        }
