from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math

# 针对启发式写作任务
SECTION_QUERIES: Dict[str, List[str]] = {
    # 1 市场机会与威胁分析
  "1.1": [
    "1.1 市场规模 TAM SAM SOM 口径 数据来源",
    "1.1 增长率 趋势 驱动因素 需求侧变化",
    "1.1 政策 监管 标准 准入门槛 影响",
    "1.1 竞争格局 主要玩家 市占 定位 差异化",
    "1.1 价格带 定价模型 商业模式 采购模式",
  ],
  "1.2": [
    "1.2 典型场景 流程拆解 角色 任务链路",
    "1.2 核心痛点 成本 效率 错误率 时效 指标",
    "1.2 现状方案 卡点 瓶颈 约束 条件",
    "1.2 需求清单 功能需求 非功能需求 优先级",
    "1.2 访谈纪要 客户反馈 原话摘录 典型案例",
  ],
  "1.3": [
    "1.3 分群 客户类型 规模 行业 适用范围",
    "1.3 普遍性 覆盖率 渗透率 需求强度",
    "1.3 样本 统计 问卷 工单 使用数据 证据",
    "1.3 验证 试点 POC 结果对比 置信度",
    "1.3 反例 边界条件 不适用场景 限制",
  ],
  "1.4": [
    "1.4 成功案例 成功因素 指标 复盘",
    "1.4 不足 短板 差距 根因分析",
    "1.4 客户反馈 工单 高频问题 体验缺陷",
    "1.4 竞品对比 输赢点 丢单原因",
    "1.4 改进方向 优化建议 优先级",
  ],
  "1.5": [
    "1.5 机会点 未满足需求 空白场景 切入点",
    "1.5 威胁 替代方案 竞品压制 进入壁垒",
    "1.5 市场空间 TAM SAM SOM 假设参数",
    "1.5 风险清单 价格战 合规 交付 资源",
    "1.5 优先级 抓手 关键路径 投入产出",
  ],

  # 2 产品价值定位
  "2.1": [
    "2.1 产品组合 策略 覆盖范围 互补 替代关系",
    "2.1 版本规划 分层 分档 套餐 打包策略",
    "2.1 客户分层 目标客群 渠道策略 触达方式",
    "2.1 协同 与存量产品 绑定销售 生态合作",
    "2.1 路线图 里程碑 迭代节奏 依赖项",
  ],
  "2.2": [
    "2.2 价值主张 目标客户 核心问题 一句话定位",
    "2.2 场景 关键价值 降本 增效 提质 降风险",
    "2.2 指标口径 ROI 量化方法 验收标准",
    "2.2 差异化 卖点 与竞品对比 关键优势",
    "2.2 证据 客户反馈 访谈 原话 案例",
  ],
  "2.3": [
    "2.3 关键假设 前提条件 因果链 依赖项",
    "2.3 可验证 指标 采集口径 对比方法",
    "2.3 失败条件 风险点 触发条件 兜底方案",
    "2.3 优先级 假设排序 先验证项",
    "2.3 成本/交付 假设 实施复杂度 复制成本",
  ],
  "2.4": [
    "2.4 POC 试点 验证对象 场景范围",
    "2.4 验证结果 数据 对比口径 指标提升",
    "2.4 反馈 异议 否定证据 反对理由",
    "2.4 结论 支持/否定 决策建议 下一步",
    "2.4 复用 标杆案例 可复制条件 推广限制",
  ],
  "2.5": [
    "2.5 技术挑战 难点 风险等级 根因",
    "2.5 可行性分析 技术路线 备选方案 对比",
    "2.5 资源 成本 周期 人力 依赖系统/数据",
    "2.5 里程碑 验证计划 测试口径 验收标准",
    "2.5 安全 合规 稳定性 可靠性 风险与措施",
  ],
  "2.6": [
    "2.6 可服务市场空间 细分市场 客户数 口径",
    "2.6 可达收入 定价模型 客单价 ARPA 假设",
    "2.6 约束条件 渠道可达 覆盖率 转化漏斗",
    "2.6 市场空间 TAM SAM SOM 数据来源",
    "2.6 获取份额 竞争格局 进入策略 假设",
  ],

  # 3 产品包定义
  "3.1": [
    "3.1 早期突破 目标客户 画像 筛选标准",
    "3.1 行业 区域 客户规模 IT成熟度 特征",
    "3.1 决策链 关键人 采购流程 立项条件",
    "3.1 试点资源 投入 交付支持 风险控制",
    "3.1 切入场景 强痛点 紧迫度 验收指标",
  ],
  "3.2": [
    "3.2 全价值链 端到端 流程地图 角色分布",
    "3.2 环节 痛点 需求 指标 约束 成本",
    "3.2 竞争格局 竞品覆盖 替代方案 对比",
    "3.2 空白点 未覆盖环节 机会点 差异化",
    "3.2 决策者 影响者 使用者 诉求对齐",
  ],
  "3.3": [
    "3.3 根本原因 因果链 5Why 鱼骨图",
    "3.3 瓶颈 约束 关键环节 排队等待 资源冲突",
    "3.3 技术挑战 边界条件 数据问题 系统割裂",
    "3.3 根因分析 组织协同 权责 激励 机制",
    "3.3 解决路径 假设 对策 验证点",
  ],
  "3.4": [
    "3.4 丢单 复盘 记录 竞品对比 输赢点",
    "3.4 不足 原因 产品能力差距 体验/性能问题",
    "3.4 根因 商务条款 定价 采购门槛 合同风险",
    "3.4 交付问题 部署复杂 定制成本 运维负担",
    "3.4 改进策略 优先级 证据 支撑材料",
  ],
  "3.5": [
    "3.5 产品形态 核心模块 关键特性 能力边界",
    "3.5 特性-价值映射 卖点 对应痛点 对应指标",
    "3.5 MVP 最小可用范围 必选特性 取舍依据",
    "3.5 版本分层 高中低配 套餐 打包策略",
    "3.5 可靠性 安全 易用性 约束 非功能需求",
  ],
  "3.6": [
    "3.6 目标价格 目标成本 定价区间 竞品价格",
    "3.6 支付意愿 价格敏感度 计费方式",
    "3.6 成本拆解 BOM 交付 运维 人力 成本项",
    "3.6 毛利 贡献利润 盈亏平衡 关键变量",
    "3.6 敏感性分析 关键假设 变动影响",
  ],
  "3.7": [
    "3.7 关键特性 求证 偏好 排序 权重",
    "3.7 接受阈值 必须具备 最低可接受标准",
    "3.7 验证对象 目标客户 试点用户 样本",
    "3.7 求证结果 反馈 汇总 支持/反对证据",
    "3.7 修改建议 迭代方向 优先级",
  ],
  "3.8": [
    "3.8 分阶段 目标 销售目标 里程碑",
    "3.8 销量 收入 渠道 区域 目标拆分",
    "3.8 关键假设 达成条件 资源需求",
    "3.8 风险清单 概率 影响 预警指标",
    "3.8 纠偏策略 应对措施 兜底方案",
  ],

  # 4 执行策略
  "4.1": [
    "4.1 技术壁垒 壁垒点 关键能力 领先性",
    "4.1 实现路径 技术路线 关键模块 里程碑",
    "4.1 复制难点 数据 生态 工程复杂度",
    "4.1 关键资源 人才 算法 数据 供应链",
    "4.1 风险 备选方案 降级策略",
  ],
  "4.2": [
    "4.2 标准 法规 认证 合规要求 清单",
    "4.2 认证路径 测试项 验收口径 周期",
    "4.2 专利 布局 现有专利 可保护点",
    "4.2 侵权风险 竞品专利 风险点 应对策略",
    "4.2 合规责任 资料留存 审计 追溯",
  ],
  "4.3": [
    "4.3 关键能力 清单 能力缺口 差距分析",
    "4.3 供应链 交付 服务 客户成功 能力建设",
    "4.3 渠道 伙伴 生态 合作模式 分工",
    "4.3 行动策略 资源投入 成本 周期",
    "4.3 保障机制 流程 质量 交付标准",
  ],
  "4.4": [
    "4.4 版本规划 路线图 迭代节奏 里程碑",
    "4.4 上市计划 GTM 交付准备 销售准备",
    "4.4 排期 依赖项 关键路径 风险缓冲",
    "4.4 验收标准 交付物 清单 测试口径",
    "4.4 风险点 延期因素 应对措施",
  ],
  "4.5": [
    "4.5 升级 替换 协同 与存量产品 关系",
    "4.5 兼容性 接口 迁移方案 数据迁移",
    "4.5 迁移成本 对客户影响 风险与缓解",
    "4.5 客户沟通 变更管理 公告 培训",
    "4.5 路线图 逐步迁移 节点",
  ],
  "4.6": [
    "4.6 平台 配套开发 依赖能力 组件清单",
    "4.6 接口 API 协议 对接对象 集成难点",
    "4.6 架构 关键模块 可扩展性 可维护性",
    "4.6 里程碑 验证计划 测试验收",
    "4.6 风险 依赖延误 兜底方案",
  ],
  "4.7": [
    "4.7 团队 角色 组织架构 RACI",
    "4.7 关键岗位 资源配置 人力需求 缺口",
    "4.7 预算 资源投入 外包 采购 计划",
    "4.7 招聘/培养 交付能力 建设计划",
    "4.7 保障机制 沟通机制 决策机制",
  ],
  "4.8": [
    "4.8 重大风险 风险清单 概率 影响",
    "4.8 预警指标 触发条件 监控方式",
    "4.8 应对措施 减缓 转移 接受 兜底",
    "4.8 责任人 RACI 跟踪节奏 闭环机制",
    "4.8 风险复盘 经验教训 改进项",
  ],

  # 5 投资收益分析
  "5.1": [
    "5.1 开发预算 人力成本 研发费用 成本口径",
    "5.1 采购 测试 认证 试点 外包 费用拆分",
    "5.1 周期 资源投入 阶段预算 里程碑",
    "5.1 成本归集 CAPEX OPEX 口径",
    "5.1 预算假设 单价 数量 工期 依据",
  ],
  "5.2": [
    "5.2 投入产出 收入模型 定价 客单价 续费",
    "5.2 成本节省 效率提升 量化口径 指标",
    "5.2 回本周期 盈亏平衡 关键假设",
    "5.2 NPV IRR 现金流 假设参数",
    "5.2 敏感性分析 关键变量 上下波动 影响",
  ]
}


@dataclass
class KBHit:
    id: str
    document: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None


def hits_to_dicts(hits: List[KBHit]) -> List[Dict[str, Any]]:
    """Serialize KBHit list to JSON-safe dicts for snapshots/state."""
    return [
        {
            "id": h.id,
            "document": h.document,
            "metadata": h.metadata,
            "distance": h.distance,
        }
        for h in hits
    ]


class KBClient:
    """In-process KB client for KBStore + Chroma (no FastAPI).

    Works with the updated KBStore design:
    - Chunks store only text + [IMAGE_x] tags
    - image_url map stored in docmeta json on disk via store.get_image_url_map()
    """

    def __init__(
        self,
        store: Any,  # KBStore
        *,
        # Embedding input max tokens = 2048 -> keep a safety margin in split path.
        query_max_tokens: int = 2048,
        # Over-limit queries will be split into token chunks (multi-query).
        query_chunk_tokens: int = 1536,
        query_chunk_overlap: int = 128,
        # safety buffer for per-piece truncation
        per_piece_safety_margin: int = 128,
    ) -> None:
        self.store = store
        self.query_max_tokens = int(query_max_tokens)
        self.query_chunk_tokens = int(query_chunk_tokens)
        self.query_chunk_overlap = int(query_chunk_overlap)
        self.per_piece_safety_margin = int(per_piece_safety_margin)

        self._tokenizer = None  # lazy

    # -------------------------
    # Write APIs (sugar)
    # -------------------------
    def index(
        self,
        *,
        project_id: str,
        document_id: str,
        text: str,
        image_url: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "action": "index",
            "projectId": project_id,
            "document_id": document_id,
            "image_url": image_url or {},
            "text": text,
        }
        return self.store.kb_action(payload)

    def delete(self, *, project_id: str, document_id: str) -> Dict[str, Any]:
        payload = {
            "action": "delete",
            "projectId": project_id,
            "document_id": document_id,
        }
        return self.store.kb_action(payload)

    # -------------------------
    # Image URL map
    # -------------------------
    def get_image_url_map(self, *, project_id: str, document_id: str) -> Dict[str, str]:
        """Read doc-level image_url map (docmeta)."""
        # new KBStore provides get_image_url_map(); fall back to _read_docmeta if needed
        if hasattr(self.store, "get_image_url_map"):
            return self.store.get_image_url_map(project_id=project_id, document_id=document_id)
        if hasattr(self.store, "_read_docmeta"):
            return self.store._read_docmeta(project_id, document_id)
        return {}

    def collect_image_url_maps(self, *, project_id: str, hits: List[KBHit]) -> Dict[str, Dict[str, str]]:
        """Collect doc_id -> image_url_map for docs that appear in hits."""
        doc_ids = {str(h.metadata.get("doc_id")) for h in hits if h.metadata and h.metadata.get("doc_id")}
        out: Dict[str, Dict[str, str]] = {}
        for doc_id in doc_ids:
            out[doc_id] = self.get_image_url_map(project_id=project_id, document_id=doc_id)
        return out

    # -------------------------
    # Tokenizer helpers (optional but recommended)
    # -------------------------
    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception:
            self._tokenizer = None
            return None

        try:
            tok = AutoTokenizer.from_pretrained(self.store.embedding_model, use_fast=True)
            self._tokenizer = tok
            return tok
        except Exception:
            self._tokenizer = None
            return None

    def _count_tokens(self, text: str) -> Optional[int]:
        tok = self._get_tokenizer()
        if tok is None:
            return None
        ids = tok.encode(text, add_special_tokens=False)
        return len(ids)

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        tok = self._get_tokenizer()
        if tok is None:
            # conservative char truncation for Chinese (~1-2 chars/token)
            return (text or "")[: max(0, int(max_tokens * 2))]
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        ids = ids[:max_tokens]
        return tok.decode(ids, skip_special_tokens=True)

    def _split_by_tokens(self, text: str, chunk_tokens: int, overlap_tokens: int) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []

        tok = self._get_tokenizer()
        if tok is None:
            # fallback: conservative char slicing
            chunk_chars = max(200, int(chunk_tokens * 2))
            overlap_chars = max(0, int(overlap_tokens * 2))
            step = max(1, chunk_chars - overlap_chars)
            out: List[str] = []
            i, n = 0, len(text)
            while i < n:
                out.append(text[i : i + chunk_chars].strip())
                i += step
            return [x for x in out if x]

        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) <= chunk_tokens:
            return [text]

        step = max(1, chunk_tokens - max(0, overlap_tokens))
        out: List[str] = []
        i, n = 0, len(ids)
        while i < n:
            piece_ids = ids[i : i + chunk_tokens]
            out.append(tok.decode(piece_ids, skip_special_tokens=True).strip())
            i += step
        return [x for x in out if x]

    # -------------------------
    # Text cleanup (compat)
    # -------------------------
    @staticmethod
    def _strip_image_url_footer(text: str) -> str:
        """Remove legacy footer:
            ---
            [IMAGE_URL_MAP]
            ...
        (Your new KBStore no longer injects it, but keep backward compatibility.)
        """
        if not text:
            return text
        marker = "\n[IMAGE_URL_MAP]\n"
        idx = text.rfind(marker)
        if idx < 0:
            return text.strip()
        cut = text.rfind("\n---\n", 0, idx)
        if cut >= 0:
            return text[:cut].strip()
        return text[:idx].strip()

    # -------------------------
    # Core Query API (token-safe + returns image maps)
    # -------------------------
    def search(
        self,
        *,
        project_id: str,
        query_text: str,
        top_k: int = 8,
        where: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        include: Optional[List[str]] = None,
        long_query_strategy: str = "split",  # "split" or "truncate"
        strip_legacy_footer: bool = True,
        return_image_maps: bool = True,
    ) -> Tuple[List[KBHit], Dict[str, Dict[str, str]]]:
        """Vector search with token overflow protection.

        Returns:
            hits: List[KBHit]
            image_maps: Dict[doc_id, Dict[tag,url]] (empty if return_image_maps=False)
        """
        query_text = (query_text or "").strip()
        if not query_text:
            return [], {}
        col = self.store._get_collection(project_id)

        w = dict(where or {})
        if document_id:
            w["doc_id"] = document_id

        inc = include or ["documents", "metadatas", "distances"]

        # 1) over-limit detection
        tok_n = self._count_tokens(query_text)
        if tok_n is not None:
            over_limit = tok_n > self.query_max_tokens
        else:
            over_limit = len(query_text) > int(self.query_max_tokens * 2)

        # 2) build query pieces
        if not over_limit:
            pieces = [query_text]
        else:
            if long_query_strategy == "truncate":
                pieces = [self._truncate_to_tokens(query_text, self.query_max_tokens)]
            else:
                chunk_tokens = min(self.query_chunk_tokens, max(64, self.query_max_tokens - self.per_piece_safety_margin))
                pieces = self._split_by_tokens(query_text, chunk_tokens, self.query_chunk_overlap)
                safe_max = max(64, self.query_max_tokens - self.per_piece_safety_margin)
                pieces = [self._truncate_to_tokens(p, safe_max) for p in pieces if p.strip()]

        if not pieces:
            return [], {}

        # 3) multi-query and merge (best distance wins)
        per_piece_k = max(3, int(math.ceil(top_k / max(1, len(pieces))) * 3))
        best: Dict[str, KBHit] = {}

        for p in pieces:
            # embed via KBStore pipeline
            try:
                qvec = self.store._embed_texts([p])[0]
            except Exception:
                # last resort: shrink and retry once
                p2 = self._truncate_to_tokens(p, max(64, self.query_max_tokens // 2))
                qvec = self.store._embed_texts([p2])[0]

            res = col.query(
                query_embeddings=[qvec],
                n_results=max(1, per_piece_k),
                where=w if w else None,
                include=inc,
            )

            ids = (res.get("ids") or [[]])[0]
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            dists = (res.get("distances") or [[]])[0] if "distances" in res else [None] * len(ids)

            for i, _id in enumerate(ids):
                dist = dists[i] if i < len(dists) else None
                doc = docs[i] if i < len(docs) else ""
                meta = metas[i] if i < len(metas) else {}
                if strip_legacy_footer:
                    doc = self._strip_image_url_footer(doc)

                hit = KBHit(id=_id, document=doc, metadata=meta, distance=dist)

                if _id not in best:
                    best[_id] = hit
                else:
                    old = best[_id].distance
                    if old is None or (dist is not None and dist < old):
                        best[_id] = hit

        hits = sorted(best.values(), key=lambda h: (h.distance is None, h.distance))
        hits = hits[: max(1, int(top_k))]

        image_maps: Dict[str, Dict[str, str]] = {}
        if return_image_maps:
            image_maps = self.collect_image_url_maps(project_id=project_id, hits=hits)

        return hits, image_maps

    # -------------------------
    # Document fetch helpers
    # -------------------------
    def get_document_chunks(
        self,
        *,
        project_id: str,
        document_id: str,
        include: Optional[List[str]] = None,
        strip_legacy_footer: bool = True,
    ) -> List[KBHit]:
        """Fetch all chunks of a document, sorted by chunk_id if present."""
        col = self.store._get_collection(project_id)
        inc = include or ["documents", "metadatas"]

        res = col.get(where={"doc_id": document_id}, include=inc)

        ids = res.get("ids") or []
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []

        hits: List[KBHit] = []
        for i in range(len(ids)):
            doc = docs[i] if i < len(docs) else ""
            if strip_legacy_footer:
                doc = self._strip_image_url_footer(doc)
            hits.append(KBHit(id=ids[i], document=doc, metadata=metas[i] if i < len(metas) else {}, distance=None))

        def _chunk_key(h: KBHit) -> Tuple[int, str]:
            cid = str(h.metadata.get("chunk_id") or "")
            try:
                return (int(cid), cid)
            except Exception:
                return (10**12, cid)

        hits.sort(key=_chunk_key)
        return hits

    def search_multi(
        self,
        *,
        project_id: str,
        queries: List[str],
        k_each: int = 4,
        k_total: int = 8,
        where: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        include: Optional[List[str]] = None,
        long_query_strategy: str = "split",
        strip_legacy_footer: bool = True,
        return_image_maps: bool = True,
    ) -> Tuple[List[KBHit], Dict[str, Dict[str, str]]]:
        """Run per-query search, then merge/dedupe with final quota control."""
        merged: Dict[str, KBHit] = {}
        for q in queries or []:
            q = (q or "").strip()
            if not q:
                continue
            hits, _ = self.search(
                project_id=project_id,
                query_text=q,
                top_k=max(1, int(k_each)),
                where=where,
                document_id=document_id,
                include=include,
                long_query_strategy=long_query_strategy,
                strip_legacy_footer=strip_legacy_footer,
                return_image_maps=False,
            )
            for h in hits:
                meta = h.metadata or {}
                doc_id = str(meta.get("doc_id") or "")
                chunk_id = meta.get("chunk_id")
                dedupe_key = f"{doc_id}::{chunk_id}" if doc_id and chunk_id not in (None, "") else f"id::{h.id}"
                old = merged.get(dedupe_key)
                if old is None:
                    merged[dedupe_key] = h
                    continue
                old_d, new_d = old.distance, h.distance
                if old_d is None or (new_d is not None and new_d < old_d):
                    merged[dedupe_key] = h

        final_hits = sorted(merged.values(), key=lambda h: (h.distance is None, h.distance))[: max(1, int(k_total))]
        image_maps: Dict[str, Dict[str, str]] = {}
        if return_image_maps:
            image_maps = self.collect_image_url_maps(project_id=project_id, hits=final_hits)
        return final_hits, image_maps

    def search_section(
        self,
        *,
        project_id: str,
        section_id: str,
        section_title: str,
        context_fingerprint: Optional[str] = None,
        snapshot: Optional[Dict[str, Any]] = None,
        reuse_snapshot: bool = True,
        k_each: int = 4,
        k_total: int = 12,
        where: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        include: Optional[List[str]] = None,
        section_queries: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Section-oriented retrieval with query fan-out + snapshot reuse."""
        if (
            reuse_snapshot
            and snapshot
            and snapshot.get("section_id") == section_id
            and snapshot.get("fingerprint") == context_fingerprint
        ):
            return snapshot

        query_bank = section_queries if section_queries is not None else SECTION_QUERIES
        sub_queries = query_bank.get(section_id, [])
        queries = [f"{section_title} | {q}" if section_title else q for q in sub_queries]

        hits, image_maps = self.search_multi(
            project_id=project_id,
            queries=queries,
            k_each=k_each,
            k_total=k_total,
            where=where,
            document_id=document_id,
            include=include,
            return_image_maps=True,
        )
        return {
            "section_id": section_id,
            "section_title": section_title,
            "fingerprint": context_fingerprint,
            "queries": queries,
            "hits": hits_to_dicts(hits),
            "image_maps": image_maps,
        }

    def list_document_ids(self, *, project_id: str) -> List[str]:
        """List doc_ids in a project. May be heavy for huge collections."""
        col = self.store._get_collection(project_id)
        res = col.get(include=["metadatas"])
        metas = res.get("metadatas") or []
        doc_ids = sorted({str(m.get("doc_id")) for m in metas if m and m.get("doc_id")})
        return doc_ids


if __name__ == "__main__":
    # Minimal demo (replace with real KBStore in actual runtime).
    # from write.infra.kb.kb_store import KBStore
    # store = KBStore(...)
    # client = KBClient(store)
    # snap = client.search_section(
    #     project_id="demo-project",
    #     section_id="1.2",
    #     section_title="典型场景与痛点",
    #     context_fingerprint="ctx-v1",
    # )
    # print(snap.keys())
    print("KBClient demo: instantiate KBClient(real_store) and call search_section(...)")
