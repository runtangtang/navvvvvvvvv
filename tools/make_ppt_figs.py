#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render two PPT diagrams offline. Never import or execute robot code.

WSL example:
  /home/runtangtang/miniconda3/envs/ros2/bin/python tools/make_ppt_figs.py
Optional: --font PATH --out-dir PATH --preview-check PATH
Dependencies: matplotlib, numpy; Pillow only for the optional preview.
"""

import argparse
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


TOOLS = Path(__file__).resolve().parent
NAV = TOOLS.parent
INK = '#183B56'
BLUE = '#3D6B90'
LINE = '#9BB5CB'
PALE = '#F4F8FC'
MODULES = (
    'mission/api.py', 'mission/ros_action.py',
    'mission/mission.py',
    'planning/global_planner.py', 'planning/local_avoid.py',
    'control/path_follower.py', 'control/fine_approach.py',
    'hal/cmd_sink.py', 'safety/watchdog.py',
    'core/types.py', 'core/frames.py', 'hal/pose_source.py',
)


def find_font(requested):
    candidates = [requested] if requested else [
        Path('/mnt/c/Windows/Fonts/msyh.ttc'),
        Path('C:/Windows/Fonts/msyh.ttc'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
        Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'),
    ]
    required = set('系统架构数据流导航层规划控制基础唯一出口安全门场景对齐狗组实时位姿误差①②③④⑤≤→')
    for path in candidates:
        if path and path.is_file():
            glyphs = FT2Font(str(path)).get_charmap()
            if all(ord(char) in glyphs for char in required):
                font_manager.fontManager.addfont(str(path))
                return path
    raise RuntimeError('No complete Chinese font found. Supply --font FONT_FILE; '
                       'no potentially garbled PNG will be saved.')


class Diagram:
    def __init__(self, font, title, subtitle):
        self.fig = plt.figure(figsize=(16, 9), dpi=200, facecolor='white')
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, 16)
        self.ax.set_ylim(0, 9)
        self.ax.axis('off')
        self.font = font
        self.font_family = font_manager.FontProperties(fname=str(font)).get_name()
        self.labels = []
        self.text(0.6, 8.55, title, 29, weight='bold')
        self.text(0.6, 8.10, subtitle, 16, color=BLUE)

    def text(self, x, y, value, size=18, color=INK, weight='normal',
             align='left', bounds=None):
        # Use the explicitly registered Chinese font, with symbol-only fallback
        # for glyphs such as the requested double arrow (not in Microsoft YaHei).
        prop = font_manager.FontProperties(family=[self.font_family, 'DejaVu Sans'],
                                           size=size, weight=weight)
        artist = self.ax.text(x, y, value, fontproperties=prop, color=color,
                              ha=align, va='center', linespacing=1.35, zorder=5)
        self.labels.append((artist, bounds))
        return artist

    def box(self, x, y, w, h, fill=PALE, edge=LINE, radius=0.08, lw=1.1):
        patch = FancyBboxPatch((x, y), w, h,
                              boxstyle=f'round,pad=0,rounding_size={radius}',
                              facecolor=fill, edgecolor=edge, linewidth=lw, zorder=1)
        self.ax.add_patch(patch)

    def arrow(self, points, dashed=False, color=BLUE, lw=1.6):
        for start, end in zip(points[:-2], points[1:-1]):
            self.ax.plot([start[0], end[0]], [start[1], end[1]],
                         color=color, linewidth=lw, linestyle='--' if dashed else '-',
                         zorder=3)
        self.ax.add_patch(FancyArrowPatch(
            points[-2], points[-1], arrowstyle='-|>', mutation_scale=15,
            linewidth=lw, color=color, linestyle='--' if dashed else '-',
            shrinkA=0, shrinkB=0, zorder=3))

    def save(self, path):
        # Check real renderer bounds, including Chinese labels; fail on missing glyphs.
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        extents = []
        for artist, bounds in self.labels:
            extent = artist.get_window_extent(renderer)
            extents.append((artist, extent))
            corners = self.ax.transData.inverted().transform(extent.get_points())
            if bounds is None:
                bounds = (0.05, 0.05, 15.95, 8.95)
            x0, y0, x1, y1 = bounds
            if (corners[0, 0] < x0 - .025 or corners[0, 1] < y0 - .025
                    or corners[1, 0] > x1 + .025 or corners[1, 1] > y1 + .025):
                raise RuntimeError('Text exceeds layout bounds: ' + artist.get_text())
        for i, (first, box_a) in enumerate(extents):
            for second, box_b in extents[i+1:]:
                overlap_x = min(box_a.x1, box_b.x1) - max(box_a.x0, box_b.x0)
                overlap_y = min(box_a.y1, box_b.y1) - max(box_a.y0, box_b.y0)
                if overlap_x > 2 and overlap_y > 2:
                    raise RuntimeError('Overlapping labels: %s / %s' % (
                        first.get_text(), second.get_text()))
        self.fig.savefig(path, dpi=200, facecolor='white')
        plt.close(self.fig)
        print('PASS Chinese glyphs, text bounds, no label overlaps: ' + path.name)


def architecture(font, output):
    d = Diagram(font, '导航系统架构', '六层职责分离  /  纯 Python 核心 + ROS2 桥接  /  当前代码视图')
    rows = [
        ('L5  对外接口', '收 6D 位姿\n返回导航结果', '#DDEAF5', [
            ('mission/api.py', 'Navigator.navigate_to()', '纯 Python，不依赖 ROS'),
            ('mission/ros_action.py', 'ROS2 Action 服务端', '/nav/navigate_to'),
        ]),
        ('L4  任务层', '管流程\n失败有明确原因', '#E1ECF6', [
            ('mission/mission.py', '状态机（6 个主流程状态）', '任务阶段与失败原因定义'),
            ('失败原因', 'timeout / blocked / stuck', 'no_path / aborted / turn_shift'),
        ]),
        ('L3  规划层', '在哪里走\n算路径、查障碍', '#E4EEF7', [
            ('planning/global_planner.py', 'Dijkstra 距离场 → 路径', '占据栅格 + 路径拉直 / 抽稀'),
            ('planning/local_avoid.py', '★ 动态避障：走廊检查 → 停车',
             '★ 可选开启；堵超 7 秒 ⇒ 报 blocked（交给上层绕行）'),
        ]),
        ('L2  控制层', '怎么走\n跟随段转走解耦', '#EAF1F8', [
            ('control/path_follower.py', '转到 ≤2° → 停稳 → 直走', '逐路点跟随 + 无进展检测'),
            ('control/fine_approach.py', '末端位置精调 + 朝向对齐', '末端闭环 + 无进展检测'),
        ]),
        ('L1  执行层', '谁能发指令\n一道门、安全停', '#EFF5FA', [
            ('hal/cmd_sink.py', '唯一出口 + 安全门', '显式运动许可 / 速度硬上限 / settle'),
            ('safety/watchdog.py', '独立线程看门狗', '位姿断流 / 回调异常 / 控制循环失活'),
        ]),
        ('L0  基础层', '坐标系不混用\n全层共享基础', '#F5F8FC', [
            ('core/types.py', '位姿携带 frame 标签', '混用时当场抛异常'),
            ('core/frames.py', 'T：CAPO 系 → 场景系', '首次建立；后续起点校验'),
            ('hal/pose_source.py', 'CAPO / 仿真 / 回放', 'OdomSource / MockSource'),
        ]),
    ]
    top = 7.90
    for i, (title, explanation, fill, items) in enumerate(rows):
        is_mission = title.startswith('L4')
        h = 1.34 if is_mission else .96
        y = top - h
        d.box(.60, y, 14.80, h, fill=fill)
        d.box(.60, y, 2.55, h, fill='#CDDDEB', edge=LINE)
        d.text(.82, y + h - .25, title, 21, weight='bold', bounds=(.70, y, 3.08, y+h))
        d.text(.82, y + (.57 if is_mission else .28), explanation, 16,
               bounds=(.70, y, 3.08, y+h))
        if is_mission:
            left = (3.42, y+.03, 9.08, y+h-.03)
            right = (9.32, y+.03, 15.08, y+h-.03)
            d.text(3.45, y+1.11, 'mission/mission.py', 17, weight='bold', bounds=left)
            d.text(3.45, y+.80, '状态机（6 个主流程状态）', 18, bounds=left)
            d.text(3.45, y+.52, '任务阶段与失败原因定义', 14, color=BLUE, bounds=left)
            d.text(9.35, y+.99,
                   '失败原因：timeout / blocked / stuck\nno_path / aborted / turn_shift',
                   14, bounds=right)
            d.text(9.35, y+.52, '★ blocked 和 stuck 分开 —— 下游处理完全不同',
                   13, color=BLUE, bounds=right)
            d.text(3.45, y+.20,
                   'IDLE → PLANNING → NAVIGATING → FINE_APPROACH → HEADING_ALIGN → DONE',
                   16, bounds=(3.42, y+.03, 15.15, y+.38))
        else:
            width = 11.8 / len(items)
            for j, (module, role, detail) in enumerate(items):
                x = 3.45 + j * width
                bounds = (x - .03, y + .03, x + width - .25, y + h - .03)
                d.text(x, y + .76, module, 17 if len(items) == 2 else 16, weight='bold', bounds=bounds)
                d.text(x, y + .47, role, 18 if len(items) == 2 else 16, bounds=bounds)
                detail_size = 13 if module == 'planning/local_avoid.py' else (15 if len(items) == 2 else 14)
                d.text(x, y + .18, detail, detail_size, color=BLUE, bounds=bounds)
        if i < len(rows) - 1:
            d.arrow([(8.2, y - .007), (8.2, y - .083)], lw=1.2)
        top = y - .09
    d.box(.60, .64, 14.80, .63, fill=INK, edge=INK)
    d.text(8, .955, '铁律：上层不许绕过下层 —— 速度指令必须走 cmd_sink，安全门才堵得住',
           20, color='white', weight='bold', align='center', bounds=(.75, .64, 15.25, 1.27))
    d.text(.65, .31, '箭头表示分层依赖，不是执行时序；L4 状态流只画主流程，不展开失败分支；L0 为各层共享基础。',
           14, color=BLUE)
    d.save(output)


def dataflow(font, output):
    d = Diagram(font, '导航数据流与三方分工', '任务 → 位姿目标 → 规划与闭环控制 → 唯一指令出口 → 运动服务')
    panels = [(0.60, 2.65, '组 2  ·  任务与目标'),
              (3.50, 9.30, '我们  ·  导航节点'),
              (13.05, 2.35, '狗  ·  运动执行')]
    for x, width, title in panels:
        d.box(x, 1.29, width, 6.36, fill=PALE)
        d.box(x, 7.08, width, .57, fill='#DDEAF5')
        d.text(x + width / 2, 7.365, title, 20, weight='bold', align='center',
               bounds=(x+.05, 7.08, x+width-.05, 7.65))
    d.text(8.15, 6.82, '/nav/navigate_to  ·  ROS2 Action', 18, align='center')

    d.box(.78, 5.95, 2.29, .91, fill='white')
    d.text(1.925, 6.59, '任务指令', 20, weight='bold', align='center')
    d.text(1.925, 6.20, '“去桌上拿可乐”', 17, align='center', bounds=(.80, 5.95, 3.05, 6.86))
    d.arrow([(1.925, 5.95), (1.925, 5.54)])
    d.text(2.16, 5.73, '大模型推理', 14, bounds=(2.05, 5.53, 3.20, 5.95))
    d.box(.78, 4.62, 2.29, .91, fill='white')
    d.text(1.925, 5.25, '起点 6D + 终点 6D', 17, weight='bold', align='center',
           bounds=(.80, 4.62, 3.05, 5.53))
    d.text(1.925, 4.87, '场景系  ·  s_* / g_*', 15, align='center')
    d.arrow([(3.07, 5.12), (3.72, 5.12)])
    d.text(3.40, 5.39, 'goal', 13, align='center')

    steps = [('① 对齐', 'T：CAPO\n→ 场景', '首次建立'),
             ('② 规划', '占据栅格\nDijkstra\n→ 路径', '全局路径'),
             ('③ 跟随', '逐路点\n转向 → 停稳\n→ 直走', '足式解耦'),
             ('④ 精调', '末端位置\n闭环精调', '位置到位'),
             ('⑤ 对齐', '末端朝向\n闭环对齐', '朝向到位')]
    for i, (title, role, detail) in enumerate(steps):
        x, y, w, h = 3.72 + i * 1.81, 4.58, 1.60, 1.97
        d.box(x, y, w, h, fill='white')
        bounds = (x+.045, y+.03, x+w-.045, y+h-.03)
        d.text(x+w/2, y+1.64, title, 20, weight='bold', align='center', bounds=bounds)
        d.text(x+w/2, y+.96, role, 16, align='center', bounds=bounds)
        d.text(x+w/2, y+.27, detail, 14, color=BLUE, align='center', bounds=bounds)
        if i < 4:
            d.arrow([(x+w+.025, 5.50), (x+1.81-.025, 5.50)], lw=1.4)

    # Runtime side inputs: pose is transformed by T, then read throughout control.
    d.box(3.72, 3.03, 2.78, 1.02, fill='#E5EEF7')
    d.text(5.11, 3.80, 'CAPO 实时位姿', 18, weight='bold', align='center')
    d.text(5.11, 3.42, '/SMX/Odom → pose_source\nT 后供跟随 / 精调闭环', 14, align='center',
           bounds=(3.77, 3.03, 6.45, 4.05))
    d.arrow([(4.52, 4.05), (4.52, 4.58)], dashed=True)
    d.box(6.97, 3.03, 2.78, 1.02, fill='#E5EEF7')
    d.text(8.36, 3.80, '地图 / 障碍输入', 18, weight='bold', align='center')
    d.text(8.36, 3.42, '栅格 → 规划\n障碍点 → 走廊检查 / 停车', 14, align='center',
           bounds=(7.02, 3.03, 9.70, 4.05))
    d.arrow([(7.60, 4.05), (7.60, 4.30), (6.33, 4.30), (6.33, 4.58)], dashed=True)
    d.arrow([(8.14, 4.05), (8.14, 4.58)], dashed=True)
    d.text(8.95, 4.30, '动态避障可选', 13, align='center')

    d.box(10.16, 3.03, 2.40, 1.02, fill='#D4E3F0', edge=BLUE, lw=1.5)
    d.text(11.36, 3.80, '唯一指令出口', 19, weight='bold', align='center')
    d.text(11.36, 3.41, 'hal/cmd_sink.py\n安全门 + watchdog', 15, align='center',
           bounds=(10.21, 3.03, 12.51, 4.05))
    d.arrow([(11.76, 4.58), (11.76, 4.05)])
    d.text(11.38, 4.30, '控制段持续输出', 12, color=BLUE, align='right')
    d.box(13.25, 3.03, 1.95, 1.36, fill='white')
    d.text(14.225, 4.10, '宇树 sport', 19, weight='bold', align='center')
    d.text(14.225, 3.70, '运动服务（闭源）', 16, align='center',
           bounds=(13.30, 3.03, 15.15, 4.39))
    d.text(14.225, 3.32, '执行速度 / 安全停机', 13, color=BLUE, align='center')
    d.arrow([(12.56, 3.57), (13.25, 3.57)])
    d.text(14.225, 5.18, '/api/sport/request\n速度 (vx, vy, wz)', 14, align='center',
           bounds=(13.10, 4.78, 15.35, 5.60))

    d.box(.78, 1.58, 2.29, 1.01, fill='white')
    d.text(1.925, 2.20, '进度条 / 可视化', 18, weight='bold', align='center')
    d.text(1.925, 1.81, '任务结果', 17, align='center')
    d.arrow([(12.55, 2.20), (3.07, 2.20)], lw=1.4)
    d.text(7.84, 2.46, 'feedback · 默认 10 Hz：阶段 / 剩余距离 / 当前位姿', 17, align='center',
           bounds=(3.55, 2.23, 12.70, 2.75))
    d.arrow([(12.55, 1.64), (3.07, 1.64)], lw=1.4)
    d.text(7.84, 1.89, 'result：成功 / 原因 / 最终位姿 / 内部误差 / 不确定度 / 用时', 16, align='center',
           bounds=(3.55, 1.65, 12.70, 2.14))

    d.text(.65, .95, '① 首个 goal：真实起点必须与 s_* 一致；摆好狗到发第一个目标之间，狗不能移动。', 14)
    d.text(.65, .59, '② 后续 goal：起点一致则沿用 T；不一致时按配置报警 / 重对齐。feedback 默认 10 Hz，供进度可视化。', 14)
    d.text(.65, .23, '③ 接口收 6D，实际导航只用 (x, y, yaw)；内部误差与不确定度估计不是物理世界真实测量误差。', 14)
    d.save(output)


def preview_check(paths, output):
    from PIL import Image, ImageDraw
    canvas = Image.new('RGB', (960, 1130), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, path in enumerate(paths):
        with Image.open(path) as source:
            if source.size != (3200, 1800):
                raise RuntimeError('Unexpected PNG dimensions: ' + str(source.size))
            # Half the width/height of a 1920x1080 slide, not half the export pixels.
            canvas.paste(source.convert('RGB').resize((960, 540), Image.Resampling.LANCZOS), (0, i*565+25))
        draw.text((10, i*565+5), path.name + ' (960x540, half a Full-HD slide)', fill=INK)
    canvas.save(output)
    print('PASS optional half-slide preview (960x540 each): ' + str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', type=Path, help='Chinese TrueType/OpenType font file')
    parser.add_argument('--out-dir', type=Path, default=TOOLS)
    parser.add_argument('--preview-check', type=Path, help='Optional temporary 960x540-per-image half-slide preview')
    args = parser.parse_args()
    missing = [module for module in MODULES if not (NAV / module).is_file()]
    if missing:
        parser.error('Missing diagram modules: ' + ', '.join(missing))
    font = find_font(args.font)
    print('Font: ' + str(font))
    print('PASS module files exist: %d' % len(MODULES))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [args.out_dir / 'ppt_architecture.png', args.out_dir / 'ppt_dataflow.png']
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message=r'Glyph .* missing from .*font.*')
        architecture(font, outputs[0])
        dataflow(font, outputs[1])
    if args.preview_check:
        preview_check(outputs, args.preview_check)
    print('PASS two 3200x1800 PNGs generated; no ROS imports or robot operations')


if __name__ == '__main__':
    main()
