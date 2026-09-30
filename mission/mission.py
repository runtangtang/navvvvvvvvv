# -*- coding: utf-8 -*-
"""
mission/mission.py —— 任务状态机

P1 阶段只用其中一段（NAVIGATING → HEADING_ALIGN）。
PLANNING 在 P2 接上全局规划；失败分支在 P3 补全。

状态图
------
  IDLE ─▶ PLANNING ─▶ NAVIGATING ─▶ FINE_APPROACH ─▶ HEADING_ALIGN ─▶ DONE
            │             │              │                │
            ▼             ▼              ▼                ▼
         NO_PATH       TIMEOUT        STUCK            TIMEOUT
                       ABORTED        ABORTED          ABORTED

★ 每一条失败路径都必须：
    ① 走 sink.settle()（StopMove + BalanceStand —— 否则腿会冻在抬腿相然后倒下）
    ② 落盘
    ③ 返回明确 reason（"blocked" 和 "no_path" 对下游是完全不同的处理）
"""
from enum import Enum


class Stage(str, Enum):
    IDLE = 'idle'
    PLANNING = 'planning'
    NAVIGATING = 'navigating'
    FINE_APPROACH = 'fine_approach'
    HEADING_ALIGN = 'heading_align'
    DONE = 'done'
    FAILED = 'failed'


class Reason(str, Enum):
    ARRIVED = 'arrived'
    TIMEOUT = 'timeout'
    STUCK = 'stuck'                 # 走不动了（障碍/卡住）
    BLOCKED = 'blocked'             # 被人或物挡住 —— 等一等可以重试
    NO_PATH = 'no_path'             # 地图上过不去 —— 要换目标或换方案
    ABORTED = 'aborted'             # 外部急停
    NO_POSE = 'no_pose'             # 位姿源就没有数据
    STALE = 'stale'                 # 位姿断流
    SOURCE_ERROR = 'source_error'   # 回调抛异常（★ 2026-09-22 真发生过）
    PLANNING_FAILED = 'planning_failed'
    INTERNAL_ERROR = 'internal_error'
    # ★★ 2026-09-26 加：位置阶段【本来是达标的】，但在【朝向对准转身】的过程中
    #    身体被挪偏了，转身结束后位置超差。
    #    真机实测：转 139° 让身体中心挪了 8cm（0.027 → 0.107），刚好超过 0.081 的容差。
    #    ★ 和 STUCK 完全不同：不是走不动，是"到了又歪了"。
    #    下游处理：可以直接重跑一次位置精调（不需要换目标/换方案）。
    TURN_SHIFT = 'turn_shift'


# 失败原因是"可重试"还是"该换方案"，下游处理完全不同
RETRYABLE = {Reason.BLOCKED, Reason.TIMEOUT, Reason.STUCK, Reason.TURN_SHIFT}
PERMANENT = {Reason.NO_PATH, Reason.PLANNING_FAILED, Reason.ABORTED}


def is_retryable(reason: str) -> bool:
    try:
        return Reason(reason) in RETRYABLE
    except ValueError:
        return False
