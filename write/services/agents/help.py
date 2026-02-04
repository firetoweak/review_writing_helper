from __future__ import annotations

import json, re, itertools
from typing import Any, AsyncIterator, Dict, List, Optional, TypedDict, Tuple

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph

from models.llm_interface_async import build_chat_model, build_messages, is_llm_configured, is_vlm_configured
from tools.prompt_templating import build_ctx, render_prompt, DEFAULT_PLACEHOLDER_MAP, build_image_map
from tools.streaming_langgraph import graph_to_ndjson_tokens
from services.agents.session_kb import SessionKB, strip_image_tags
import asyncio


class HelpState(TypedDict, total=False):
    # persisted
    sessionId: str
    helpPrompt: str
    messages: List[Dict[str, Any]]
    ai_seq: int

    # ephemeral (per turn only; do NOT persist)
    incoming: List[Dict[str, Any]]
    payload_ctx: Dict[str, Any]

    # computed (per turn)
    system_prompt: str
    user_text: str
    image_map: Dict[str, str]
    multimodal: bool
    lc_messages: Any


class HelpAgent:
    def __init__(
        self,
        *,
        kb: Optional[Any] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ) -> None:
        self._counter = itertools.count(1)
        self._checkpointer = checkpointer or MemorySaver()
        self._ph_map = dict(DEFAULT_PLACEHOLDER_MAP)
        self._session_kb = SessionKB()
        self.kb_client = kb
        # stream graph (model)
        self._model_text = build_chat_model(streaming=True, multimodal=False)
        self._model_mm = build_chat_model(streaming=True, multimodal=True)
        self._stream_graph = self._build_stream_graph()

        # pipeline graph (state machine)
        self._pipe = self._build_pipeline_graph()

    # -------------------------
    # Public API
    # -------------------------
    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        meta = None
        content = ""
        async for line in self.stream(payload):
            obj = self._safe_json(line)
            if not obj:
                continue
            t = obj.get("type")
            if t == "message.start":
                meta = obj
            elif t == "message.delta":
                content += (obj.get("delta") or "")
            elif t == "message.end":
                break

        sid = (meta.get("sessionId") if meta else None) or str(payload.get("sessionId") or "").strip()
        msg = (meta.get("assistantMessage") if meta else None) or {"messageId": self._next_mid(), "role": "assistant", "content": ""}
        msg = dict(msg)
        msg["content"] = content
        return {"sessionId": sid, "assistantMessage": msg}

    async def stream(self, payload: Dict[str, Any]) -> AsyncIterator[str]:
        session_id = str(payload.get("sessionId") or "").strip()
        thread_id = self._thread_id(session_id)

        # initial per-turn state
        incoming = self._coerce_messages(payload)
        prompt_tpl = self._coerce_help_prompt(payload.get("prompt"))  # may be None
        payload_ctx = dict(payload)  # used only for this turn

        # 1) run pipeline state machine (merge -> prompt -> rag -> build_lc)
        state_in: HelpState = {
            "sessionId": session_id,
            "incoming": incoming,
            "payload_ctx": payload_ctx,
        }
        if prompt_tpl is not None:
            state_in["helpPrompt"] = prompt_tpl

        state = await self._pipe.ainvoke(state_in, config={"configurable": {"thread_id": thread_id}})

        # msg id (persisted seq already advanced by pipeline)
        seq = int(state.get("ai_seq") or 0)
        msg_id = f"m_ai_{seq}"

        yield self._ndjson({"type": "message.start", "sessionId": session_id, "assistantMessage": {"messageId": msg_id, "role": "assistant", "content": ""}})

        # 3) stream model tokens (call_model graph)
        content_acc = ""
        completed = False
        try:
            async for ev in graph_to_ndjson_tokens(
                self._stream_graph,
                {"messages": state["lc_messages"], "multimodal": bool(state.get("multimodal"))},
            ):
                obj = self._safe_json(ev)
                if not obj:
                    continue
                if obj.get("type") == "token":
                    tok = obj.get("text") or ""
                    if tok:
                        content_acc += tok
                        yield self._ndjson({"type": "message.delta", "messageId": msg_id, "delta": tok})
                elif obj.get("type") == "done":
                    completed = True
                    break
        finally:
            if content_acc:
                await self._append_assistant(thread_id, {"messageId": msg_id, "role": "assistant", "content": content_acc})

        yield self._ndjson({"type": "message.end", "messageId": msg_id, "done": True, "finishReason": "stop" if completed else "cancelled"})
        yield self._ndjson({"type": "done"})

    # -------------------------
    # Pipeline graph (state machine)
    # -------------------------
    def _build_pipeline_graph(self):
        g = StateGraph(HelpState)

        def merge_input(state: HelpState) -> HelpState:
            # persisted
            hist = state.get("messages") or []
            incoming = state.get("incoming") or []
            merged = self._merge_by_message_id(hist, incoming)

            # prompt
            prompt = (state.get("helpPrompt") or "").rstrip()

            # advance seq (persisted)
            seq = int(state.get("ai_seq") or 0) + 1

            # IMPORTANT: drop incoming/payload_ctx so they don't persist
            return {
                "sessionId": (state.get("sessionId") or "").strip(),
                "helpPrompt": prompt,
                "messages": merged,
                "ai_seq": seq,
                # keep ephemeral for next nodes via explicit carry:
                "payload_ctx": state.get("payload_ctx") or {},
                "incoming": incoming,
            }

        def prepare_prompt(state: HelpState) -> HelpState:
            payload_ctx = state.get("payload_ctx") or {}
            # payload overrides state for placeholder injection
            ctx = build_ctx({**state, **payload_ctx})
            system_prompt = render_prompt(state.get("helpPrompt") or "", ctx, self._ph_map, keep_unknown=True)
            return {"system_prompt": system_prompt}

        async def rag(state: HelpState) -> HelpState:
            sid = (state.get("sessionId") or "").strip()
            incoming = state.get("incoming") or []
            hist = state.get("messages") or []
            payload_ctx = state.get("payload_ctx") or {}
            ctx = build_ctx({**state, **payload_ctx})

            _, curr_imgs = self._session_kb.index_incoming_attachments(session_id=sid, incoming_messages=incoming)

            # query = latest user msg + (optional) helpText
            last_user = self._latest_user_text(incoming) or self._latest_user_text(hist) or ""
            q = last_user
            if ctx.get("helpText"):
                q = (q + "\n任务/意图: " + (ctx.get("helpText") or "")).strip()
            

            # 检索session_kb
            # ✅ 若本轮上传了附件但 query 为空，则使用默认模板，确保触发 KB 检索
            if not (q or "").strip() and self._has_attachments(incoming):
                help_text = (ctx.get("helpText") or "").strip()
                if help_text:
                    q = f"请总结本轮新上传附件的关键信息，并结合以下任务/意图给出建议：{help_text}"
                else:
                    q = "请总结本轮新上传附件的关键信息，提炼关键要点、风险与结论。"

            sess_kb_txt, sess_kb_imgs = self._session_kb.retrieve(
                session_id=sid,
                query_text=q,
                top_k=SessionKB.TOP_K,
                allow_images=is_vlm_configured(),
            )


            # 检索project_kb
            # ProjectKB query：把 help 的关键信息拼进去（跟 outline 的 extra 类似）
            print("当前轮次传输的信息:", json.dumps(payload_ctx, ensure_ascii=False, indent=4))
            project_kb_txt, project_kb_imgs = await self._aretrieve_kb(
                payload=payload_ctx,
                query_text=last_user,
                top_k=3
            )

            if project_kb_imgs:
                # 让 build_image_map() 能把 KB 的 docmeta 映射纳入统一映射
                payload_ctx["image_maps"] = project_kb_imgs
            img_project = build_image_map(payload_ctx)  # tag->url

            merged_imgs = {**(img_project or {}), **(sess_kb_imgs or {}), **(curr_imgs or {})}


            # ✅ only-VLM fallback: no images still use mm if llm not configured
            use_mm = is_vlm_configured() and (bool(merged_imgs) or (not is_llm_configured()))

            parts: List[str] = []
            if project_kb_txt:
                parts.append("【项目知识库命中材料】\n" + project_kb_txt)
            if sess_kb_txt:
                parts.append("【会话附件检索结果】\n" + sess_kb_txt)

            user_text = "\n\n".join([p for p in parts if p]).strip()
            if not use_mm:
                user_text = strip_image_tags(user_text)

            return {"user_text": user_text, "image_map": merged_imgs, "multimodal": use_mm}

        def build_lc_messages(state: HelpState) -> HelpState:
            lc = build_messages(
                system_prompt=state.get("system_prompt") or "",
                user_text=state.get("user_text") or "",
                messages=state.get("messages") or [],
                image_map=(state.get("image_map") if state.get("multimodal") else None),
            )
            # drop ephemeral fields to avoid persisting big dicts
            return {"lc_messages": lc, "incoming": [], "payload_ctx": {}}

        g.add_node("merge_input", merge_input)
        g.add_node("prepare_prompt", prepare_prompt)
        g.add_node("rag", rag)
        g.add_node("build_lc_messages", build_lc_messages)

        g.set_entry_point("merge_input")
        g.add_edge("merge_input", "prepare_prompt")
        g.add_edge("prepare_prompt", "rag")
        g.add_edge("rag", "build_lc_messages")
        g.set_finish_point("build_lc_messages")

        return g.compile(checkpointer=self._checkpointer)

    # -------------------------
    # Stream graph (model)
    # -------------------------
    def _build_stream_graph(self):
        g = StateGraph(dict)

        async def call_model(state: dict) -> dict:
            use_mm = bool(state.get("multimodal"))
            # ✅ extra safety: only-VLM forces mm
            if (not use_mm) and (not is_llm_configured()) and is_vlm_configured():
                use_mm = True
            model = self._model_mm if use_mm else self._model_text
            resp = await model.ainvoke(state["messages"])
            return {"response": resp}

        g.add_node("call_model", call_model)
        g.set_entry_point("call_model")
        g.set_finish_point("call_model")
        return g.compile()

    # -------------------------
    # State helpers
    # -------------------------
    async def _append_assistant(self, thread_id: str, msg: Dict[str, Any]) -> None:
        # minimal “merge + persist” by invoking pipeline’s merge node again
        await self._pipe.ainvoke(
            {"incoming": [msg], "payload_ctx": {}},
            config={"configurable": {"thread_id": thread_id}},
        )
        
    def _has_attachments(self, msgs: Any) -> bool:
        """
        判断本轮 incoming 是否包含附件/图片线索，用于 query 为空时触发默认检索模板。

        
        覆盖常见几种形态：
        - message["attachments"] 为非空 list（附件解析文本/图片 url 往往在这里）
        - message["image_url"] 为非空 dict（或单个 url 字符串）
        - message["content"] 里包含 [IMAGE_1] 这类标签
        """
        if not isinstance(msgs, list):
            return False

        img_tag = re.compile(r"\[IMAGE_\d+\]")

        for m in msgs:
            if not isinstance(m, dict):
                continue

            # 1) attachments list
            atts = m.get("attachments")
            if isinstance(atts, list) and len(atts) > 0:
                return True

            # 2) image_url map or url
            iu = m.get("image_url")
            if isinstance(iu, dict) and len(iu) > 0:
                return True
            if isinstance(iu, str) and iu.strip():
                return True

            # 3) content contains [IMAGE_x]
            content = m.get("content") or ""
            if isinstance(content, str) and img_tag.search(content):
                return True

        return False

    def _merge_by_message_id(self, old: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = {m.get("messageId") for m in old if m.get("messageId")}
        out = list(old)
        for m in new or []:
            mid = m.get("messageId")
            if (not mid) or (mid not in seen):
                out.append(m)
                if mid:
                    seen.add(mid)
        return out

    def _latest_user_text(self, msgs: Any) -> str:
        if not isinstance(msgs, list):
            return ""
        for m in reversed(msgs):
            if not isinstance(m, dict):
                continue
            role = (m.get("role") or "").strip().lower()
            if role in ("user", "human"):
                return (m.get("content") or "").strip()
        return ""

    def _coerce_messages(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        msgs = payload.get("messages")
        if isinstance(msgs, list) and msgs:
            return [m for m in msgs if isinstance(m, dict)]
        msg = payload.get("message")
        if isinstance(msg, dict):
            return [msg]
        return []

    def _coerce_help_prompt(self, p: Any) -> Optional[str]:
        if isinstance(p, str):
            return p
        if isinstance(p, list) and p:
            p = p[0]
        if isinstance(p, dict):
            return p.get("helpPrompt") or p.get("helpPormpt")
        return None

    def _thread_id(self, sid: str) -> str:
        sid = sid or "default"
        return "help:" + re.sub(r"[\s/]+", "_", sid)

    def _ndjson(self, obj: dict) -> str:
        return json.dumps(obj, ensure_ascii=False) + "\n"

    def _safe_json(self, line: str) -> Optional[dict]:
        try:
            return json.loads(line)
        except Exception:
            return None

    def _next_mid(self) -> str:
        return f"m_ai_{next(self._counter)}"

    # -------------------------
    # KB helpers
    # -------------------------

    async def _aretrieve_kb(
        self,
        payload: Dict,
        *,
        query_text: str,
        top_k: int = 3,
    ) -> Tuple[str, Dict[str, Dict[str, str]]]:
        """
        异步检索项目知识库
        返回：
          - kb_text: 命中片段拼接后的文本
          - image_maps: {doc_id: {tag: url}}
        """
        if not payload.get("useKB", True):
            return "", {}

        project_id = (payload.get("projectId") or "").strip()
        if not project_id or not (query_text or "").strip():
            return "", {}
        
        print("==========================query_text:\n", query_text)
        print("==========================project_id:", project_id)


        hits, image_maps = await asyncio.to_thread(
            self.kb_client.search,
            project_id=str(project_id),
            query_text=query_text,
            top_k=top_k,
        )

        kb_text = "\n".join(
            h.document.strip()
            for h in (hits or [])
            if getattr(h, "document", None) and str(h.document).strip()
        ).strip()

        return kb_text, (image_maps or {})
