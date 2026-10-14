# -*- coding: utf-8 -*-
"""
control/fine_approach.py —— 末端精调（位置 + 6D 朝向）

这个文件是 2026-09-22 一天真机实验的结晶。教训直接写在代码里：
------------------------------------------------------------------
✦ 执行器死区（实测）
      vx < 0.15 m/s  → 狗根本不抬脚
      wz < 0.25 rad/s → 狗根本不抬脚
  ⇒ 【比例控制 vx = kp·剩余距离 会把自己锁死】：
        距离越小速度越小 → 掉进死区 → 不动 → 距离不变 → 永远出不来
        真机两次 abort（卡在 7.6cm / 7.2cm），仿真复现（8.6cm）
  ⇒ 正确做法：**恒速 + 提前量急停**，靠"调时间"而不是"调速度"来精细

✦ 制动滑行（实测）
      0.5 m/s 急停后滑行 3.1cm  ⇒ 停的判据里要把这 3.1cm 算进去

✦ 转向（实测）
      实际角速度只有指令的约 50%，且低于 0.2 rad/s 完全不动
  ⇒ 需要转时【至少给 WZ_MIN】

✦ 停稳会转 ~7°、转身会挪身体 ~3.6cm（机制未明）
  ⇒ 朝向阶段放在位置阶段之后，并留出补偿余地
"""
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

from core.types import Pose2D, Velocity, ZERO_VEL, wrap


# ================================================================ 配置
@dataclass(frozen=True)
class ApproachConfig:
    # ---- 位置阶段 ----
    v_cruise: float = 0.50          # 巡航速度（★ 恒定，不减速）
    vx_mode: str = 'const'          # 'const' | 'prop'
                                    # ★ 'prop' 是【反面教材】，留在这里是为了
                                    #   让验收能证明仿真还有效（它一定会卡死）。
                                    #   绝对不要在生产里用。
    kp: float = 1.5                 # 仅 'prop' 模式用
    pos_tol: float = 0.03           # 位置容差 [m]
    coast: float = 0.031            # ★ 实测制动滑行 [m]（0.5 m/s 时）
    # ---- 朝向阶段 ----
    # ★★ 2026-10-08：转向参数整套上调，和 path_follower 同步
    #   依据：遥控转 90° 时 CAPO 测到 91.3°（狗能转、CAPO 准），
    #         而我们发 0.40 时狗"一跳一跳几乎不转"（3 秒 9.9°）
    #   详见 path_follower.FollowerConfig 里同一段注释
    k_ang: float = 1.5              # 位置阶段的朝向修正增益（2026-10-08 退回）
    drive_align_deg: float = 6.0    # ★真机调：朝向偏多少就不许前进（只原地转）
                                    #   ★★ 2026-10-15：10 → 6（用户定：
                                    #     "走之前方向基本对准 goal"）
                                    #   几何：6° 走 1m 偏 10.5cm；10° 偏 17cm；30° 偏 50cm
                                    #   超过阈值 ⇒ vx=0 先原地转（wz 照给），对准了再走
                                    #   ★ 没超阈值也不是放开：走的时候仍用 k_ang 边
                                    #     走边修（wz 有死区下限 0.6，10° 偏差 ~0.3s 修完）
                                    #   和 path_follower 同一个教训：
                                    #   偏差 64° 还全速走 = 斜着冲 → 画弧甩出去
                                    #   （2026-09-22 端到端演示：末端精调因此画弧
                                    #     跑了 >0.8m 触发行程上限 abort）
    wz_min: float = 0.60            # ★★ 2026-10-08：0.40 → 0.60（实测 0.4 几乎转不动）
    wz_max: float = 0.80            # ★★ 2026-10-08：0.60 → 0.80（当天先到 1.00 又退回）
    yaw_tol_deg: float = 3.0        # 朝向容差 [deg]
    k_yaw: float = 1.5              # ★★ 2026-10-08：1.2 → 1.5（当天先到 2.0 又退回）
    # ---- 其他 ----
    turn_deadband: float = 0.0      # ★ 默认 0（关）。开过会卡死，见文件头注释
    control_hz: float = 20.0
    # ---- 失效保护 ----
    timeout: float = 30.0           # 单阶段硬超时 [s]
    # ★ 卡死检测：走和转【各用各的指标】（★ 2026-10-15 拆开）
    #   · 走（在发 vx）⇒ 只看【位置】：3 秒挪不到 3cm ⇒ blocked
    #     （角度不参与 —— 推着东西时身体会晃，别把 blocked 晃没了）
    #   · 转（在发 wz）⇒ 只看【角度】：3 秒转不到 3° ⇒ stuck
    #     ★ 原来是 10°（"位置和角度都没动"的统一判据，2026-09-22 定）——
    #       会把"转得慢"误判成卡死：实测 wz=0.4 时 3 秒转 9.9° < 10° ⇒ 冤判
    #     ⇒ 2026-10-15 用户定：转向判据 10 → 3（"几乎没动"才叫卡死）
    no_progress_s: float = 3.0
    no_progress_min_m: float = 0.03
    no_progress_min_deg: float = 3.0
    max_travel_factor: float = 1.6  # 行程上限 = factor × 初始距离
    stale_s: float = 0.30           # 位姿断流阈值


DEFAULT_CFG = ApproachConfig()


# ================================================================ 纯函数
@dataclass(frozen=True)
class Cmd:
    vel: Velocity
    dist: float          # 到目标的距离 [m]
    ang_err: float       # 朝向偏差 [rad]，目标在左边为正
    done: bool
    reason: str


def compute_position_cmd(pose: Pose2D, goal: Pose2D, cfg: ApproachConfig) -> Cmd:
    """位置阶段：把 (goal − pose) 旋到机体系，恒速前进 + 朝向修正。

    ★ 纯函数，不碰 ROS —— 所以单测和仿真都能直接调它。
    """
    _same_frame(pose, goal)
    ex, ey = goal.x - pose.x, goal.y - pose.y
    dist = math.hypot(ex, ey)

    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    fwd = ex * c + ey * s            # 目标在机体系的前向分量
    left = -ex * s + ey * c          # 左向分量
    ang_err = math.atan2(left, fwd)

    # ★ 停的判据要带【全额】制动滑行补偿 —— 保守，宁可早停也不要冲过头。
    #
    #   试过"按朝向投影补偿"（coast×cos(ang_err)），结果【更糟】：
    #   机头偏大时窗口收紧到 pos_tol，而 0.5 m/s 下每周期走 2.5cm + 滑行 3cm，
    #   5cm 的窗口根本打不中 → 冲过头 → 而 vx≥0【不会倒车】→ 掉头再冲 → 绕圈
    #   → 行程超限 abort（真值误差 32cm，比全额补偿时的 7cm 差得多）。
    #
    #   ⇒ 代价是"承诺的停止点是 pos_tol+coast"而不是 pos_tol，
    #     所以【成功判据必须跟着放宽到 pos_tol+coast】，见 go_to()。
    #     真要做到 pos_tol，得上【脉冲补齐】，那是下一步的事。
    if dist <= cfg.pos_tol + cfg.coast:
        return Cmd(ZERO_VEL, dist, ang_err, True, 'reached')

    wz = max(-cfg.wz_max, min(cfg.wz_max, cfg.k_ang * ang_err))
    if cfg.turn_deadband > 0 and dist < cfg.turn_deadband:
        wz = 0.0
    elif abs(ang_err) > math.radians(cfg.drive_align_deg) and abs(wz) < cfg.wz_min:
        # ★ 低于死区指令 = 狗不抬脚 ⇒ 比例项会把自己锁死
        wz = math.copysign(cfg.wz_min, ang_err)
        # ★★ 2026-10-15：这里的门从 yaw_tol_deg 改成 drive_align_deg ——
        #    两者必须一致，否则有一个【冻死区】：
        #      vx=0 的条件是 |偏差| ≥ drive_align（不许走）
        #      抬升死区的条件是 |偏差| > yaw_tol
        #    若 yaw_tol > drive_align（比如 10 > 6），
        #      偏差落在中间那带 ⇒ 不许走 + 比例项低于死区 ⇒ 狗【完全不动】
        #      ⇒ 3 秒后被判"转不动 ⇒ stuck"（2026-10-15 P2 走廊用例实测踩到）
        #    ⇒ 语义："不许走"的那一刻起，就必须给出一个真能转的指令。

    # ★ 朝向没对准就不许前进 —— 只原地转（wz 照给）
    aligned_enough = abs(ang_err) < math.radians(cfg.drive_align_deg)

    if cfg.vx_mode == 'prop':
        # ❌ 反面教材：速度随距离衰减 → 掉进 0.15 m/s 死区 → 锁死
        vx = max(0.0, min(cfg.v_cruise, cfg.kp * dist)) if aligned_enough else 0.0
    else:
        # ✅ 恒速：永远在死区之上，靠"什么时候停"而不是"多慢"来控制精度
        vx = cfg.v_cruise if aligned_enough else 0.0

    return Cmd(Velocity(vx=vx, vy=0.0, wz=wz), dist, ang_err,
               False, 'navigating')


def compute_align_cmd(pose: Pose2D, target_yaw: float, cfg: ApproachConfig) -> Cmd:
    """朝向阶段：原地转到目标朝向。"""
    yer = wrap(target_yaw - pose.yaw)
    if abs(math.degrees(yer)) <= cfg.yaw_tol_deg:
        return Cmd(ZERO_VEL, 0.0, yer, True, 'aligned')
    wz = max(-cfg.wz_max, min(cfg.wz_max, cfg.k_yaw * yer))
    if abs(wz) < cfg.wz_min:
        wz = math.copysign(cfg.wz_min, yer)     # 同上：不许掉进死区
    return Cmd(Velocity(vx=0.0, vy=0.0, wz=wz), 0.0, yer, False, 'aligning')


def _same_frame(a: Pose2D, b: Pose2D):
    if a.frame != b.frame:
        from core.types import FrameMismatch
        raise FrameMismatch('目标与当前位姿不在同一个坐标系：%s vs %s'
                            % (a.frame, b.frame))


# ================================================================ 驱动
@dataclass
class ApproachResult:
    success: bool
    reason: str                       # reached / timeout / stale / stuck / aborted
    final_pose: Optional[Pose2D]
    final_error: float
    elapsed: float
    n_cycles: int


class FineApproach:
    """把纯函数接到真实的位姿源 + 指令出口上，带全套失效保护。"""

    def __init__(self, cfg: ApproachConfig, source, sink,
                 log: Optional[Callable[[str], None]] = None,
                 should_abort: Optional[Callable[[], bool]] = None,
                 on_cycle: Optional[Callable[[], None]] = None,
                 on_stage: Optional[Callable[[str], None]] = None):
        self.cfg = cfg
        self.source = source
        self.sink = sink
        self.log = log or (lambda s: None)
        self.should_abort = should_abort or (lambda: False)
        # ★ 2026-09-26：上报阶段 —— 位置阶段 / 朝向阶段（给 feedback 的 stage 用）
        #   HEADING_ALIGN 只有这里知道，Navigator 看不到
        self.on_stage = on_stage or (lambda st: None)
        # ★ 每转一圈调一次 —— 给看门狗报到用（"控制循环还活着"）
        self.on_cycle = on_cycle or (lambda: None)
        self._final_error_fn = lambda p: float('nan')

    # ------------------------------------------------------------
    def go_to(self, goal: Pose2D, target_yaw: Optional[float] = None) -> ApproachResult:
        """先走到 goal（位置），再转到 target_yaw（如果给了）。"""
        t0 = time.time()
        p = self._pose()
        if p is None:
            return ApproachResult(False, 'no_pose', None, float('nan'), 0.0, 0)
        start = p
        travel_cap = self.cfg.max_travel_factor * math.hypot(
            goal.x - start.x, goal.y - start.y)
        n = 0
        self._final_error_fn = lambda pp: math.hypot(goal.x - pp.x, goal.y - pp.y)

        # ---------------- 位置阶段 ----------------
        self.on_stage('fine_approach')
        r = self._loop(lambda pp: compute_position_cmd(pp, goal, self.cfg),
                       t0, travel_cap, start)
        n += r.n_cycles
        if not r.success or target_yaw is None:
            return ApproachResult(r.success, r.reason, r.final_pose,
                                  r.final_error, time.time() - t0, n)

        # ---------------- 朝向阶段 ----------------
        self.on_stage('heading_align')
        p2 = r.final_pose
        err_pos = math.hypot(goal.x - p2.x, goal.y - p2.y)
        self.log('  位置阶段完成，误差 %.4f m；开始朝向对准' % err_pos)
        r2 = self._loop(lambda pp: compute_align_cmd(pp, target_yaw, self.cfg),
                        time.time(), 3.0, p2)
        n += r2.n_cycles

        # ★ 必须【同时】满足位置和朝向 —— 只看朝向会报"成功"但其实没到位置
        #   （2026-09-22 的 P1 验收就撞上了：位置差 6.8cm 却报 arrived）
        # ★ 成功判据必须和【停止判据】一致。
        #   停止判据是 pos_tol + coast（保守），所以最终位置可能到 pos_tol+coast。
        #   之前这里写的是 pos_tol，比停止判据还严 ⇒ 停对了也判失败
        #   （2026-09-22 P2 验收撞上：停在 0.0701m，被判 ❌）
        accept = self.cfg.pos_tol + self.cfg.coast
        pos_ok = math.isfinite(r2.final_error) and r2.final_error <= accept
        ok = r2.success and pos_ok
        if r2.success and not pos_ok:
            # ★★ 2026-09-26 区分两种情况 —— 它们的处理方式完全不同：
            if err_pos <= accept:
                # 位置阶段【本来达标】⇒ 是【转身】把身体挪偏的
                # 真机实测：转 139° 挪了 8cm（0.027 → 0.107）
                reason = 'turn_shift'
                self.log('  ⚠️ 位置阶段达标(%.4f)，但【转身后】挪到 %.4f'
                         '（容差 %.3f，挪了 %.4f）' %
                         (err_pos, r2.final_error, accept,
                          r2.final_error - err_pos))
            else:
                # 位置阶段就【没到】⇒ 是走位本身没收敛
                reason = 'position_out_of_tol'
                self.log('  ⚠️ 朝向对了，但位置差 %.4f m（容差 %.3f）—— 判失败'
                         % (r2.final_error, accept))
        else:
            reason = r2.reason
        return ApproachResult(ok, reason, r2.final_pose,
                              r2.final_error, time.time() - t0, n)

    # ------------------------------------------------------------
    def _loop(self, cmd_fn, t0, travel_cap, start) -> ApproachResult:
        """★ 卡死判据统一成"位置和角度【都没动】"——
        不再需要"每个阶段配一个指标"（那是旧设计的补丁）。"""
        cfg = self.cfg
        prog_t, prog_pose = None, None
        prog_walking = None          # ★ 2026-10-15：记录上次的分类（走/转），切换时重置窗口
        last_pose = None
        n = 0
        reason = 'unknown'
        try:
            while True:
                now = time.time()
                n += 1
                self.on_cycle()          # ★ 给看门狗报到

                # ---- 失效保护：位姿源 ----
                if self.should_abort():
                    reason = 'aborted'
                    break
                if self.source.error_count() > 0:
                    reason = 'source_callback_error'
                    break
                t_pose = self.source.get()
                if t_pose is None or (now - t_pose.recv_t) > cfg.stale_s:
                    reason = 'stale_pose'
                    break
                pose = t_pose.data
                last_pose = pose

                # ---- 失效保护：超时 / 行程 ----
                if now - t0 > cfg.timeout:
                    reason = 'timeout'
                    break
                if travel_cap and math.hypot(pose.x - start.x,
                                             pose.y - start.y) > travel_cap:
                    reason = 'travel_cap'
                    break

                cmd = cmd_fn(pose)
                if cmd.done:
                    reason = cmd.reason
                    break

                # ---- 无进展：看【当时在让它干什么】，走和转各用各的指标 ----
                #   （★ 2026-10-15 拆开：见配置里的注释 —— 旧版 10° 会冤判"转得慢"）
                #   ★ 走↔转一切换，窗口【重新计】—— 否则"刚走完切成转"时，
                #     窗口里 yaw 几乎没变（走路本来就不修朝向）⇒ 一进转就冤判
                #     （10-15 全量回归里真踩到：走廊用例被误报 stuck）
                walking = cmd.vel.vx > 0.01
                if prog_t is None or walking != prog_walking:
                    prog_t, prog_pose, prog_walking = now, pose, walking
                elif now - prog_t > cfg.no_progress_s:
                    dm = math.hypot(pose.x - prog_pose.x, pose.y - prog_pose.y)
                    da = abs(math.degrees(wrap(pose.yaw - prog_pose.yaw)))
                    no_prog = (dm < cfg.no_progress_min_m if walking
                               else da < cfg.no_progress_min_deg)
                    if no_prog:
                        # ★★ 2026-09-28：区分 blocked / stuck —— 看【当时在让它干什么】
                        #   走 ⇒ 前方有东西挡着 ⇒ blocked（等一等/绕行能过）
                        #   转 ⇒ 姿态卡死/打滑 ⇒ stuck（要调整姿态）
                        #   ★ 为什么必须分清：组2 的 SPAV 只在 reason=='blocked' 时
                        #     走"绕行"分支，拿到别的直接安全熔断退出整个任务
                        if walking:
                            reason = 'blocked'
                            self.log('  🚧 前方受阻：%.1fs 内走不动（只挪 %.3fm）'
                                     ' ⇒ blocked' % (cfg.no_progress_s, dm))
                        else:
                            reason = 'stuck'
                            self.log('  ⚠️ 姿态卡死：%.1fs 内转不动（只转 %.1f°）'
                                     ' ⇒ stuck' % (cfg.no_progress_s, da))
                        break
                    prog_t, prog_pose = now, pose

                self.sink.move(cmd.vel)
                time.sleep(1.0 / cfg.control_hz)
        except Exception as e:
            reason = 'exception: %s' % e
        finally:
            # ★ 收尾期间照样报到 —— 这不是"控制循环死了"，是在减速/停稳。
            #   不报到的话，看门狗会在"位置阶段结束 → 转向阶段开始"这个
            #   ~0.45s 的间隙里误报 control_loop_dead（2026-09-22 演示撞上）
            self.sink.move(ZERO_VEL)
            self.on_cycle()
            time.sleep(0.05)
            self.sink.stop()

        # 停稳后再读一次（给下面判断用），并做收尾
        for _ in range(6):
            self.on_cycle()
            time.sleep(0.05)
        tp = self.source.get()
        final = tp.data if tp else last_pose
        fe = self._final_error_fn(final) if final is not None else float('nan')
        ok = reason in ('reached', 'aligned')
        return ApproachResult(ok, reason, final, fe, time.time() - t0, n)

    # ------------------------------------------------------------
    def _pose(self) -> Optional[Pose2D]:
        t = self.source.get()
        return t.data if t else None
