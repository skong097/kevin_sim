"""
Kevin 3D Patrol Simulator — ROS2 자율순찰 로봇 3D 시뮬레이션
Pygame + OpenGL 기반

Features:
  - 3D 환경에서 Kevin 로봇 조종
  - WASD 이동 / 마우스 시점 제어
  - LiDAR 시각화 (360° 레이저 스캔)
  - Nav2 웨이포인트 순찰 모드
  - 낙상 감지 이벤트 시뮬레이션
  - SLAM occupancy grid 실시간 맵 빌딩
  - ROS2 스타일 토픽 모니터 HUD
  - 미니맵 + SLAM 맵 오버레이
  - VBO/DisplayList 성능 최적화

Requirements:
  pip install pygame PyOpenGL PyOpenGL_accelerate numpy

Usage:
  python kevin_3d_sim.py

Controls:
  WASD  : 이동
  Mouse : 시점 회전 (화면 클릭으로 캡처)
  ALT   : 마우스 해제 (커서 자유 이동)
  SPACE : 점프 / 상승
  TAB   : 자동 순찰 모드 토글
  L     : LiDAR 시각화 토글
  F     : 낙상 이벤트 트리거
  M     : 미니맵 토글
  G     : SLAM 맵 3D 시각화 토글
  R     : SLAM 맵 리셋
  1-3   : 카메라 모드 (1:1인칭, 2:3인칭, 3:탑뷰)
  F11   : 전체 화면 토글
  ESC   : 마우스 해제 / 종료
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import random
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum, auto
import heapq

# ═══════════════════════════════════════════════
#  상수
# ═══════════════════════════════════════════════

WIDTH, HEIGHT = 1280, 800
FPS = 60
MOVE_SPEED = 0.08
ROTATE_SPEED = 0.15
MOUSE_SENSITIVITY = 0.2

# 맵 크기
MAP_SIZE = 40
WALL_HEIGHT = 2.5
GRID_SIZE = 2

# 색상 (RGBA float)
COL_FLOOR = (0.15, 0.18, 0.22, 1.0)
COL_WALL = (0.3, 0.35, 0.45, 1.0)
COL_WALL_TOP = (0.4, 0.45, 0.55, 1.0)
COL_ROBOT = (0.2, 0.6, 0.9, 1.0)
COL_ROBOT_ACCENT = (0.1, 0.8, 0.5, 1.0)
COL_LIDAR = (0.0, 1.0, 0.3, 0.6)
COL_WAYPOINT = (1.0, 0.6, 0.1, 0.8)
COL_WAYPOINT_ACTIVE = (1.0, 0.2, 0.2, 1.0)
COL_OBSTACLE = (0.6, 0.25, 0.25, 1.0)
COL_PERSON = (0.9, 0.75, 0.5, 1.0)
COL_ALERT = (1.0, 0.2, 0.2, 1.0)
COL_SKY_TOP = (0.05, 0.05, 0.15)
COL_SKY_BOT = (0.1, 0.12, 0.2)
COL_GRID = (0.2, 0.25, 0.3, 0.3)

# SLAM 색상
COL_SLAM_FREE = (0.1, 0.35, 0.15, 0.25)
COL_SLAM_OCCUPIED = (0.9, 0.3, 0.1, 0.6)
COL_SLAM_FRONTIER = (0.2, 0.7, 1.0, 0.4)

# SLAM 설정
SLAM_RESOLUTION = 0.5       # 셀 크기 (미터)
SLAM_GRID_SIZE = int(MAP_SIZE / SLAM_RESOLUTION) + 4  # 그리드 크기
SLAM_UNKNOWN = -1
SLAM_FREE = 0
SLAM_OCCUPIED = 100

# A* 경로 탐색 색상
COL_PATH_LINE = (1.0, 0.95, 0.2, 0.9)
COL_PATH_OPEN = (0.2, 0.6, 1.0, 0.3)
COL_PATH_CLOSED = (0.5, 0.2, 0.8, 0.2)
COL_PATH_NODE = (1.0, 1.0, 0.3, 0.8)

# Costmap 설정
COSTMAP_RESOLUTION = 1.0        # 셀 크기 (미터)
COSTMAP_GLOBAL_SIZE = int(MAP_SIZE / COSTMAP_RESOLUTION) + 2
COSTMAP_LOCAL_RADIUS = 8.0      # 로컬 costmap 반경 (미터)
COSTMAP_INFLATION = 2.5         # 장애물 팽창 반경 (미터)
COST_LETHAL = 254
COST_INSCRIBED = 200
COST_FREE = 0


# ═══════════════════════════════════════════════
#  맵 생성
# ═══════════════════════════════════════════════

def generate_map():
    """간단한 실내 순찰 환경 생성"""
    walls = []
    obstacles = []

    # 외벽
    for i in range(-MAP_SIZE // 2, MAP_SIZE // 2 + 1, GRID_SIZE):
        walls.append((i, -MAP_SIZE // 2))
        walls.append((i, MAP_SIZE // 2))
        walls.append((-MAP_SIZE // 2, i))
        walls.append((MAP_SIZE // 2, i))

    # 내부 벽 (복도 구조)
    # 수평 벽
    for x in range(-12, 0, GRID_SIZE):
        walls.append((x, 6))
    for x in range(4, 14, GRID_SIZE):
        walls.append((x, 6))
    for x in range(-14, -4, GRID_SIZE):
        walls.append((x, -6))
    for x in range(0, 14, GRID_SIZE):
        walls.append((x, -6))

    # 수직 벽
    for z in range(-6, 6, GRID_SIZE):
        walls.append((-14, z))
    for z in range(-14, -6, GRID_SIZE):
        walls.append((0, z))
    for z in range(6, 14, GRID_SIZE):
        walls.append((8, z))

    # 장애물 (가구 등) — 순찰 경로를 막지 않도록 배치 검증 완료
    obstacle_positions = [
        (-10, 0), (-8, -15), (6, 10), (10, -10),
        (-10, 10), (4, -3), (-4, -12), (12, 2),
        (-10, 16), (18, -8), (-6, 18), (10, -16),
    ]
    for pos in obstacle_positions:
        obstacles.append(pos)

    return walls, obstacles


# 웨이포인트 (순찰 경로) — 전 구간 충돌 시뮬레이션 검증 완료
#
# 맵 문(갭) 위치:
#   z= 6 수평벽: x ≈ -0.7 ~ 2.7 통과 가능
#   z=-6 수평벽: x ≈ -4.7 ~ -1.3 통과 가능
#   x=-14 수직벽: z=-6~4 밀폐 (문 없음)
#   x=0 수직벽: z=-14~-8 밀폐
#   x=8 수직벽: z=6~12 밀폐
#
PATROL_WAYPOINTS = [
    (-8, -12),    # WP0:  좌하단 시작
    (-3, -8),     # WP1:  z=-6 문 접근
    (-3, -4),     # WP2:  z=-6 문 통과
    (-3, 2),      # WP3:  중앙부 (x=-14 벽 오른쪽)
    (1, 4),       # WP4:  z=6 문 접근
    (1, 9),       # WP5:  z=6 문 통과
    (-6, 16),     # WP6:  상단 좌측
    (5, 16),      # WP7:  상단 우측 (x=8 벽 왼쪽)
    (10, 16),     # WP8:  x=8 벽 우회
    (16, 10),     # WP9:  우상단
    (16, 0),      # WP10: 우측 중앙
    (16, -12),    # WP11: 우하단
    (6, -12),     # WP12: 하단 (x=0 벽 오른쪽)
    (4, -17),     # WP13: x=0 수직벽 아래로 우회
    (-5, -17),    # WP14: 좌하단 복귀 경유
]

# 사람 위치 (낙상 감지 대상)
PERSON_POSITIONS = [
    (-6, -8), (6, 8), (-10, 10), (10, -10), (0, 0),
]


# ═══════════════════════════════════════════════
#  공간 해시 (충돌 최적화)
# ═══════════════════════════════════════════════

class SpatialHash:
    """공간 해시 기반 충돌 검사 — O(1) 근접 쿼리"""

    def __init__(self, cell_size=3.0):
        self.cell_size = cell_size
        self.grid = {}

    def _key(self, x, z):
        return (int(x // self.cell_size), int(z // self.cell_size))

    def insert(self, x, z, obj_type, radius):
        key = self._key(x, z)
        for dx in range(-1, 2):
            for dz in range(-1, 2):
                k = (key[0] + dx, key[1] + dz)
                if k not in self.grid:
                    self.grid[k] = []
                self.grid[k].append((x, z, obj_type, radius))

    def query(self, x, z):
        key = self._key(x, z)
        return self.grid.get(key, [])


def build_spatial_hash(walls, obstacles):
    """벽/장애물을 공간 해시에 등록"""
    sh = SpatialHash(3.0)
    for wx, wz in walls:
        sh.insert(wx, wz, 'wall', 1.2)
    for ox, oz in obstacles:
        sh.insert(ox, oz, 'obstacle', 0.8)
    return sh


# ═══════════════════════════════════════════════
#  SLAM Visualizer (Occupancy Grid)
# ═══════════════════════════════════════════════

class SLAMVisualizer:
    """SLAM occupancy grid 실시간 맵 빌딩"""

    def __init__(self):
        self.grid_size = SLAM_GRID_SIZE
        self.resolution = SLAM_RESOLUTION
        self.origin = -MAP_SIZE / 2 - 1  # 그리드 원점 (월드 좌표)

        # Occupancy grid: -1=unknown, 0=free, 100=occupied
        self.grid = np.full((self.grid_size, self.grid_size), SLAM_UNKNOWN, dtype=np.int8)

        # Log-odds 맵 (확률적 업데이트용)
        self.log_odds = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.l_free = -0.4    # free 관측 시 log-odds 감소
        self.l_occ = 0.85     # occupied 관측 시 log-odds 증가
        self.l_max = 5.0
        self.l_min = -3.0

        # 탐사율 통계
        self.explored_cells = 0
        self.occupied_cells = 0
        self.total_cells = self.grid_size * self.grid_size

        # 3D 시각화 디스플레이 리스트
        self._display_list = None
        self._dirty = True
        self._rebuild_counter = 0

    def world_to_grid(self, wx, wz):
        """월드 좌표 → 그리드 인덱스"""
        gx = int((wx - self.origin) / self.resolution)
        gz = int((wz - self.origin) / self.resolution)
        return gx, gz

    def grid_to_world(self, gx, gz):
        """그리드 인덱스 → 월드 좌표 (셀 중심)"""
        wx = self.origin + (gx + 0.5) * self.resolution
        wz = self.origin + (gz + 0.5) * self.resolution
        return wx, wz

    def _in_bounds(self, gx, gz):
        return 0 <= gx < self.grid_size and 0 <= gz < self.grid_size

    def update_from_lidar(self, robot_x, robot_z, robot_yaw, spatial_hash, num_rays=72, max_range=12.0):
        """LiDAR 스캔 결과로 occupancy grid 업데이트 (Bresenham ray march)"""
        rx, rz = self.world_to_grid(robot_x, robot_z)

        for i in range(num_rays):
            angle = robot_yaw + (2 * math.pi * i / num_rays)
            dx = math.sin(angle)
            dz = math.cos(angle)

            # 레이캐스트 (공간 해시 사용)
            hit_dist = max_range
            for step_f in range(1, int(max_range * 4)):
                step = step_f / 4.0
                px = robot_x + dx * step
                pz = robot_z + dz * step

                near = spatial_hash.query(px, pz)
                hit = False
                for nx, nz, ntype, nrad in near:
                    if abs(px - nx) < nrad and abs(pz - nz) < nrad:
                        hit_dist = step
                        hit = True
                        break
                if hit:
                    break

            # Bresenham 라인으로 free 셀 마킹
            end_x = robot_x + dx * hit_dist
            end_z = robot_z + dz * hit_dist
            ex, ez = self.world_to_grid(end_x, end_z)

            # 시작점에서 끝점까지 free 마킹
            points = self._bresenham(rx, rz, ex, ez)
            for j, (gx, gz) in enumerate(points):
                if not self._in_bounds(gx, gz):
                    continue
                if j < len(points) - 1:
                    # free 셀
                    self.log_odds[gx, gz] = max(self.l_min,
                                                 self.log_odds[gx, gz] + self.l_free)
                elif hit_dist < max_range:
                    # occupied 셀 (히트 포인트)
                    self.log_odds[gx, gz] = min(self.l_max,
                                                 self.log_odds[gx, gz] + self.l_occ)

        # Log-odds → occupancy grid 변환
        self.grid = np.where(self.log_odds > 0.5, SLAM_OCCUPIED,
                             np.where(self.log_odds < -0.5, SLAM_FREE, SLAM_UNKNOWN)).astype(np.int8)

        # 통계 업데이트
        self.explored_cells = int(np.sum(self.grid != SLAM_UNKNOWN))
        self.occupied_cells = int(np.sum(self.grid == SLAM_OCCUPIED))
        self._dirty = True

    def _bresenham(self, x0, z0, x1, z1):
        """Bresenham 라인 알고리즘"""
        points = []
        dx = abs(x1 - x0)
        dz = abs(z1 - z0)
        sx = 1 if x0 < x1 else -1
        sz = 1 if z0 < z1 else -1
        err = dx - dz
        while True:
            points.append((x0, z0))
            if x0 == x1 and z0 == z1:
                break
            e2 = 2 * err
            if e2 > -dz:
                err -= dz
                x0 += sx
            if e2 < dx:
                err += dx
                z0 += sz
            # 최대 길이 제한
            if len(points) > 200:
                break
        return points

    def draw_3d(self):
        """SLAM 맵을 3D로 렌더링 (occupied 셀을 바닥 위에 표시)"""
        self._rebuild_counter += 1

        # 매 10프레임마다 디스플레이 리스트 갱신
        if self._dirty and self._rebuild_counter % 10 == 0:
            if self._display_list is not None:
                glDeleteLists(self._display_list, 1)
            self._display_list = glGenLists(1)
            glNewList(self._display_list, GL_COMPILE)

            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            r = self.resolution * 0.48
            h_free = 0.03
            h_occ = 0.4

            # Free 셀 (바닥에 얇은 레이어)
            glBegin(GL_QUADS)
            for gx in range(self.grid_size):
                for gz in range(self.grid_size):
                    val = self.grid[gx, gz]
                    if val == SLAM_FREE:
                        wx, wz = self.grid_to_world(gx, gz)
                        glColor4f(*COL_SLAM_FREE)
                        glVertex3f(wx - r, h_free, wz - r)
                        glVertex3f(wx + r, h_free, wz - r)
                        glVertex3f(wx + r, h_free, wz + r)
                        glVertex3f(wx - r, h_free, wz + r)
                    elif val == SLAM_OCCUPIED:
                        wx, wz = self.grid_to_world(gx, gz)
                        # 윗면
                        glColor4f(*COL_SLAM_OCCUPIED)
                        glVertex3f(wx - r, h_occ, wz - r)
                        glVertex3f(wx + r, h_occ, wz - r)
                        glVertex3f(wx + r, h_occ, wz + r)
                        glVertex3f(wx - r, h_occ, wz + r)
            glEnd()

            # Occupied 셀 측면 (약간의 입체감)
            glBegin(GL_QUADS)
            for gx in range(self.grid_size):
                for gz in range(self.grid_size):
                    if self.grid[gx, gz] == SLAM_OCCUPIED:
                        wx, wz = self.grid_to_world(gx, gz)
                        c = (COL_SLAM_OCCUPIED[0] * 0.6, COL_SLAM_OCCUPIED[1] * 0.6,
                             COL_SLAM_OCCUPIED[2] * 0.6, COL_SLAM_OCCUPIED[3])
                        glColor4f(*c)
                        # 앞
                        glVertex3f(wx - r, 0, wz + r)
                        glVertex3f(wx + r, 0, wz + r)
                        glVertex3f(wx + r, h_occ, wz + r)
                        glVertex3f(wx - r, h_occ, wz + r)
                        # 오른쪽
                        glVertex3f(wx + r, 0, wz - r)
                        glVertex3f(wx + r, 0, wz + r)
                        glVertex3f(wx + r, h_occ, wz + r)
                        glVertex3f(wx + r, h_occ, wz - r)
            glEnd()

            # Frontier 셀 (탐색 경계) — unknown 옆의 free 셀
            glBegin(GL_QUADS)
            for gx in range(1, self.grid_size - 1):
                for gz in range(1, self.grid_size - 1):
                    if self.grid[gx, gz] == SLAM_FREE:
                        # 4방향 인접 셀 중 unknown이 있으면 frontier
                        if (self.grid[gx-1, gz] == SLAM_UNKNOWN or
                            self.grid[gx+1, gz] == SLAM_UNKNOWN or
                            self.grid[gx, gz-1] == SLAM_UNKNOWN or
                            self.grid[gx, gz+1] == SLAM_UNKNOWN):
                            wx, wz = self.grid_to_world(gx, gz)
                            glColor4f(*COL_SLAM_FRONTIER)
                            glVertex3f(wx - r, 0.05, wz - r)
                            glVertex3f(wx + r, 0.05, wz - r)
                            glVertex3f(wx + r, 0.05, wz + r)
                            glVertex3f(wx - r, 0.05, wz + r)
            glEnd()

            glEndList()
            self._dirty = False

        if self._display_list is not None:
            glCallList(self._display_list)

    def draw_on_minimap(self, hud, map_x, map_y, map_size):
        """미니맵 위에 SLAM 그리드 오버레이"""
        cx = map_x + map_size // 2
        cy = map_y + map_size // 2
        scale = map_size / (MAP_SIZE + 4)
        cell_px = max(1.5, self.resolution * scale)

        glBegin(GL_QUADS)
        for gx in range(self.grid_size):
            for gz in range(self.grid_size):
                val = self.grid[gx, gz]
                if val == SLAM_UNKNOWN:
                    continue
                wx, wz = self.grid_to_world(gx, gz)
                mx = cx + wx * scale
                my = cy + wz * scale
                if not (map_x < mx < map_x + map_size and map_y < my < map_y + map_size):
                    continue

                if val == SLAM_FREE:
                    glColor4f(0.1, 0.4, 0.15, 0.3)
                elif val == SLAM_OCCUPIED:
                    glColor4f(1.0, 0.3, 0.1, 0.8)

                hr = cell_px / 2
                glVertex2f(mx - hr, my - hr)
                glVertex2f(mx + hr, my - hr)
                glVertex2f(mx + hr, my + hr)
                glVertex2f(mx - hr, my + hr)
        glEnd()

    def get_exploration_pct(self):
        """탐사율 반환"""
        non_wall = self.total_cells
        if non_wall == 0:
            return 0.0
        return (self.explored_cells / non_wall) * 100.0

    def reset(self):
        """SLAM 맵 초기화"""
        self.grid.fill(SLAM_UNKNOWN)
        self.log_odds.fill(0.0)
        self.explored_cells = 0
        self.occupied_cells = 0
        self._dirty = True
        if self._display_list is not None:
            glDeleteLists(self._display_list, 1)
            self._display_list = None


# ═══════════════════════════════════════════════
#  A* 경로 탐색기
# ═══════════════════════════════════════════════

class AStarPlanner:
    """
    A* 경로 탐색 + 탐색 과정 애니메이션 시각화
    SpatialHash 기반 충돌 맵 사용 (SLAM과 독립적으로 동작)
    """

    # 탐색 해상도 (월드 좌표 단위)
    CELL = 1.0
    HALF = MAP_SIZE // 2 + 1
    GRID_W = int((MAP_SIZE + 2) / 1.0)  # 그리드 폭

    def __init__(self, spatial_hash):
        self.spatial_hash = spatial_hash

        # 탐색 결과
        self.path = []              # 최종 경로 [(wx, wz), ...]
        self.open_set_vis = []      # 시각화용 open set 좌표
        self.closed_set_vis = []    # 시각화용 closed set 좌표

        # 애니메이션 상태
        self.anim_steps = []        # 탐색 과정 전체 기록 [(open, closed, partial_path), ...]
        self.anim_index = 0         # 현재 재생 위치
        self.anim_playing = False
        self.anim_speed = 3         # 프레임당 진행 스텝 수
        self.planning_active = False

        # 3D 시각화 캐시
        self._display_list = None
        self._dirty = True

    def _world_to_grid(self, wx, wz):
        return int((wx + self.HALF) / self.CELL), int((wz + self.HALF) / self.CELL)

    def _grid_to_world(self, gx, gz):
        return -self.HALF + (gx + 0.5) * self.CELL, -self.HALF + (gz + 0.5) * self.CELL

    def _is_walkable(self, gx, gz):
        """그리드 셀이 통행 가능한지 (공간 해시로 충돌 체크)"""
        wx, wz = self._grid_to_world(gx, gz)
        # 맵 범위 체크
        if abs(wx) > MAP_SIZE // 2 or abs(wz) > MAP_SIZE // 2:
            return False
        near = self.spatial_hash.query(wx, wz)
        for nx, nz, ntype, nrad in near:
            if abs(wx - nx) < nrad + 0.3 and abs(wz - nz) < nrad + 0.3:
                return False
        return True

    def _heuristic(self, a, b):
        """옥토일 거리 (대각선 이동 비용 고려)"""
        dx = abs(a[0] - b[0])
        dz = abs(a[1] - b[1])
        return max(dx, dz) + (math.sqrt(2) - 1) * min(dx, dz)

    def plan(self, start_wx, start_wz, goal_wx, goal_wz):
        """A* 경로 탐색 실행 + 애니메이션 데이터 기록"""
        self.path = []
        self.anim_steps = []
        self.anim_index = 0
        self.anim_playing = True
        self.planning_active = True
        self._dirty = True

        start = self._world_to_grid(start_wx, start_wz)
        goal = self._world_to_grid(goal_wx, goal_wz)

        # 8방향 이웃
        neighbors_8 = [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]

        open_heap = []  # (f_score, counter, node)
        counter = 0
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        came_from = {}
        open_set = {start}
        closed_set = set()

        heapq.heappush(open_heap, (f_score[start], counter, start))

        found = False
        step_count = 0

        while open_heap:
            _, _, current = heapq.heappop(open_heap)

            if current in closed_set:
                continue

            if current == goal:
                found = True
                break

            open_set.discard(current)
            closed_set.add(current)

            # 매 N스텝마다 애니메이션 프레임 기록
            step_count += 1
            if step_count % 4 == 0:
                # 현재까지의 부분 경로 (current → start 역추적)
                partial = []
                node = current
                while node in came_from:
                    partial.append(self._grid_to_world(*node))
                    node = came_from[node]
                partial.reverse()

                self.anim_steps.append((
                    [self._grid_to_world(*n) for n in list(open_set)[:200]],
                    [self._grid_to_world(*n) for n in list(closed_set)[-200:]],
                    partial
                ))

            for dx, dz in neighbors_8:
                nx, nz = current[0] + dx, current[1] + dz
                neighbor = (nx, nz)

                if neighbor in closed_set:
                    continue
                if not self._is_walkable(nx, nz):
                    closed_set.add(neighbor)
                    continue

                # 대각선 비용
                move_cost = math.sqrt(2) if (dx != 0 and dz != 0) else 1.0
                tent_g = g_score[current] + move_cost

                if tent_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tent_g
                    f = tent_g + self._heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    open_set.add(neighbor)
                    counter += 1
                    heapq.heappush(open_heap, (f, counter, neighbor))

            # 탐색 제한 (성능 보호)
            if step_count > 5000:
                break

        # 최종 경로 복원
        if found:
            node = goal
            while node in came_from:
                self.path.append(self._grid_to_world(*node))
                node = came_from[node]
            self.path.append(self._grid_to_world(*start))
            self.path.reverse()

        # 최종 프레임 추가
        self.anim_steps.append((
            [],
            [self._grid_to_world(*n) for n in list(closed_set)[-300:]],
            self.path[:]
        ))

        # 시각화용 최종 데이터
        self.open_set_vis = []
        self.closed_set_vis = [self._grid_to_world(*n) for n in closed_set]

        return found

    def update_animation(self):
        """매 프레임 호출 — 애니메이션 진행"""
        if not self.anim_playing or not self.anim_steps:
            return

        self.anim_index += self.anim_speed
        if self.anim_index >= len(self.anim_steps):
            self.anim_index = len(self.anim_steps) - 1
            self.anim_playing = False
        self._dirty = True

    def draw_3d(self):
        """A* 탐색 과정 3D 시각화"""
        if not self.planning_active:
            return

        # 현재 애니메이션 프레임
        if self.anim_steps and self.anim_index < len(self.anim_steps):
            open_pts, closed_pts, partial_path = self.anim_steps[self.anim_index]
        else:
            open_pts = self.open_set_vis
            closed_pts = self.closed_set_vis
            partial_path = self.path

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        r = self.CELL * 0.4

        # Closed set (보라색 바닥 타일)
        if closed_pts:
            glBegin(GL_QUADS)
            glColor4f(*COL_PATH_CLOSED)
            for wx, wz in closed_pts:
                glVertex3f(wx - r, 0.04, wz - r)
                glVertex3f(wx + r, 0.04, wz - r)
                glVertex3f(wx + r, 0.04, wz + r)
                glVertex3f(wx - r, 0.04, wz + r)
            glEnd()

        # Open set (파란색 바닥 타일)
        if open_pts:
            glBegin(GL_QUADS)
            glColor4f(*COL_PATH_OPEN)
            for wx, wz in open_pts:
                glVertex3f(wx - r, 0.05, wz - r)
                glVertex3f(wx + r, 0.05, wz - r)
                glVertex3f(wx + r, 0.05, wz + r)
                glVertex3f(wx - r, 0.05, wz + r)
            glEnd()

        # 경로 라인
        if partial_path and len(partial_path) >= 2:
            t = time.time()
            glLineWidth(3.0)
            glBegin(GL_LINE_STRIP)
            for i, (wx, wz) in enumerate(partial_path):
                pulse = 0.7 + 0.3 * math.sin(t * 4 + i * 0.3)
                glColor4f(COL_PATH_LINE[0], COL_PATH_LINE[1] * pulse,
                          COL_PATH_LINE[2], COL_PATH_LINE[3])
                glVertex3f(wx, 0.15, wz)
            glEnd()
            glLineWidth(1.0)

            # 경로 노드 포인트
            glPointSize(5.0)
            glBegin(GL_POINTS)
            glColor4f(*COL_PATH_NODE)
            for wx, wz in partial_path:
                glVertex3f(wx, 0.16, wz)
            glEnd()
            glPointSize(1.0)

    def draw_on_minimap(self, hud, map_x, map_y, map_size):
        """미니맵에 A* 탐색 과정 오버레이"""
        if not self.planning_active:
            return

        cx = map_x + map_size // 2
        cy = map_y + map_size // 2
        scale = map_size / (MAP_SIZE + 4)

        # 현재 프레임
        if self.anim_steps and self.anim_index < len(self.anim_steps):
            open_pts, closed_pts, partial_path = self.anim_steps[self.anim_index]
        else:
            open_pts, closed_pts, partial_path = [], self.closed_set_vis, self.path

        # Closed set (작은 점)
        if closed_pts:
            glPointSize(1.5)
            glBegin(GL_POINTS)
            glColor4f(0.5, 0.2, 0.8, 0.4)
            for wx, wz in closed_pts:
                mx = cx + wx * scale
                my = cy + wz * scale
                if map_x < mx < map_x + map_size and map_y < my < map_y + map_size:
                    glVertex2f(mx, my)
            glEnd()

        # Open set
        if open_pts:
            glPointSize(2.0)
            glBegin(GL_POINTS)
            glColor4f(0.2, 0.6, 1.0, 0.6)
            for wx, wz in open_pts:
                mx = cx + wx * scale
                my = cy + wz * scale
                if map_x < mx < map_x + map_size and map_y < my < map_y + map_size:
                    glVertex2f(mx, my)
            glEnd()

        # 경로 라인
        if partial_path and len(partial_path) >= 2:
            glLineWidth(2.0)
            glBegin(GL_LINE_STRIP)
            glColor4f(1.0, 0.95, 0.2, 0.9)
            for wx, wz in partial_path:
                mx = cx + wx * scale
                my = cy + wz * scale
                glVertex2f(mx, my)
            glEnd()
            glLineWidth(1.0)
        glPointSize(1.0)

    def clear(self):
        """탐색 결과 초기화"""
        self.path = []
        self.open_set_vis = []
        self.closed_set_vis = []
        self.anim_steps = []
        self.anim_index = 0
        self.anim_playing = False
        self.planning_active = False
        self._dirty = True
        if self._display_list is not None:
            glDeleteLists(self._display_list, 1)
            self._display_list = None


# ═══════════════════════════════════════════════
#  Costmap 시각화 (Global + Local)
# ═══════════════════════════════════════════════

class CostmapVisualizer:
    """
    Nav2 스타일 Costmap 시각화
    - Global Costmap: 전체 맵의 정적 장애물 + inflation 레이어
    - Local Costmap: 로봇 주변 실시간 비용 맵 (LiDAR 기반)
    """

    def __init__(self, spatial_hash):
        self.spatial_hash = spatial_hash
        self.res = COSTMAP_RESOLUTION
        self.size = COSTMAP_GLOBAL_SIZE
        self.origin = -MAP_SIZE / 2 - 1

        # Global costmap (정적, 1회 빌드)
        self.global_map = np.zeros((self.size, self.size), dtype=np.uint8)
        self._build_global(spatial_hash)

        # Local costmap (동적, 매 프레임)
        self.local_radius = COSTMAP_LOCAL_RADIUS
        self.local_size = int(self.local_radius * 2 / self.res) + 1
        self.local_map = np.zeros((self.local_size, self.local_size), dtype=np.uint8)
        self.local_origin_x = 0.0
        self.local_origin_z = 0.0

        # 시각화 모드
        self.show_global = True
        self.show_local = True

        # 디스플레이 리스트 (global은 정적)
        self._global_dl = None
        self._local_dl = None
        self._local_dirty = True

    def _world_to_grid(self, wx, wz):
        return int((wx - self.origin) / self.res), int((wz - self.origin) / self.res)

    def _grid_to_world(self, gx, gz):
        return self.origin + (gx + 0.5) * self.res, self.origin + (gz + 0.5) * self.res

    def _build_global(self, spatial_hash):
        """정적 장애물로 global costmap 빌드 (inflation 포함)"""
        # 1단계: lethal 셀 마킹
        lethal_cells = []
        for gx in range(self.size):
            for gz in range(self.size):
                wx, wz = self._grid_to_world(gx, gz)
                near = spatial_hash.query(wx, wz)
                for nx, nz, ntype, nrad in near:
                    if abs(wx - nx) < nrad and abs(wz - nz) < nrad:
                        self.global_map[gx, gz] = COST_LETHAL
                        lethal_cells.append((gx, gz))
                        break

        # 2단계: inflation (거리 기반 비용 감쇠)
        inf_cells = int(COSTMAP_INFLATION / self.res)
        for lx, lz in lethal_cells:
            for dx in range(-inf_cells, inf_cells + 1):
                for dz in range(-inf_cells, inf_cells + 1):
                    nx, nz = lx + dx, lz + dz
                    if 0 <= nx < self.size and 0 <= nz < self.size:
                        dist = math.sqrt(dx * dx + dz * dz) * self.res
                        if dist < 0.5:
                            cost = COST_INSCRIBED
                        elif dist < COSTMAP_INFLATION:
                            # 지수 감쇠
                            ratio = 1.0 - (dist / COSTMAP_INFLATION)
                            cost = int(COST_INSCRIBED * ratio * ratio)
                        else:
                            continue
                        self.global_map[nx, nz] = max(self.global_map[nx, nz], cost)

    def update_local(self, robot_x, robot_z, lidar_results):
        """LiDAR 결과로 local costmap 업데이트"""
        self.local_origin_x = robot_x - self.local_radius
        self.local_origin_z = robot_z - self.local_radius
        self.local_map.fill(0)

        for hit_dist, end_x, end_z in lidar_results:
            if hit_dist >= 12.0:  # max range = no hit
                continue
            # 히트 포인트를 local grid에 마킹
            lx = int((end_x - self.local_origin_x) / self.res)
            lz = int((end_z - self.local_origin_z) / self.res)
            if 0 <= lx < self.local_size and 0 <= lz < self.local_size:
                self.local_map[lx, lz] = COST_LETHAL
                # 간단 inflation (3셀)
                for dx in range(-2, 3):
                    for dz in range(-2, 3):
                        nx, nz = lx + dx, lz + dz
                        if 0 <= nx < self.local_size and 0 <= nz < self.local_size:
                            dist = math.sqrt(dx * dx + dz * dz) * self.res
                            if dist < 0.3:
                                cost = COST_INSCRIBED
                            elif dist < 2.0:
                                cost = int(150 * max(0, 1.0 - dist / 2.0))
                            else:
                                continue
                            self.local_map[nx, nz] = max(self.local_map[nx, nz], cost)

        self._local_dirty = True

    def draw_3d_global(self):
        """Global costmap 3D 시각화 (DisplayList 캐싱)"""
        if not self.show_global:
            return

        if self._global_dl is None:
            self._global_dl = glGenLists(1)
            glNewList(self._global_dl, GL_COMPILE)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            r = self.res * 0.48
            glBegin(GL_QUADS)
            for gx in range(self.size):
                for gz in range(self.size):
                    cost = self.global_map[gx, gz]
                    if cost == 0:
                        continue
                    wx, wz = self._grid_to_world(gx, gz)

                    if cost >= COST_LETHAL:
                        continue  # lethal은 벽 자체가 표시
                    elif cost >= COST_INSCRIBED:
                        glColor4f(0.9, 0.2, 0.1, 0.25)
                    elif cost > 100:
                        glColor4f(0.9, 0.5, 0.1, 0.15)
                    elif cost > 30:
                        glColor4f(0.9, 0.8, 0.2, 0.08)
                    else:
                        continue  # 아주 낮은 비용은 스킵

                    glVertex3f(wx - r, 0.02, wz - r)
                    glVertex3f(wx + r, 0.02, wz - r)
                    glVertex3f(wx + r, 0.02, wz + r)
                    glVertex3f(wx - r, 0.02, wz + r)
            glEnd()
            glEndList()

        glCallList(self._global_dl)

    def draw_3d_local(self):
        """Local costmap 3D 시각화 (매 프레임 갱신)"""
        if not self.show_local:
            return

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        r = self.res * 0.45
        t = time.time()
        pulse = 0.8 + 0.2 * math.sin(t * 3)

        glBegin(GL_QUADS)
        for lx in range(self.local_size):
            for lz in range(self.local_size):
                cost = self.local_map[lx, lz]
                if cost < 20:
                    continue
                wx = self.local_origin_x + (lx + 0.5) * self.res
                wz = self.local_origin_z + (lz + 0.5) * self.res

                if cost >= COST_INSCRIBED:
                    glColor4f(1.0 * pulse, 0.15, 0.05, 0.35)
                elif cost > 80:
                    glColor4f(1.0, 0.4 * pulse, 0.05, 0.2)
                else:
                    ratio = cost / 150.0
                    glColor4f(1.0, 0.7, 0.1, 0.1 * ratio)

                glVertex3f(wx - r, 0.06, wz - r)
                glVertex3f(wx + r, 0.06, wz - r)
                glVertex3f(wx + r, 0.06, wz + r)
                glVertex3f(wx - r, 0.06, wz + r)
        glEnd()

    def draw_on_minimap(self, hud, map_x, map_y, map_size, robot_x, robot_z):
        """미니맵에 costmap 오버레이"""
        cx = map_x + map_size // 2
        cy = map_y + map_size // 2
        scale = map_size / (MAP_SIZE + 4)

        # Local costmap만 미니맵에 표시 (global은 너무 밀집)
        if self.show_local:
            glBegin(GL_QUADS)
            for lx in range(self.local_size):
                for lz in range(self.local_size):
                    cost = self.local_map[lx, lz]
                    if cost < 30:
                        continue
                    wx = self.local_origin_x + (lx + 0.5) * self.res
                    wz = self.local_origin_z + (lz + 0.5) * self.res
                    mx = cx + wx * scale
                    my = cy + wz * scale
                    if not (map_x < mx < map_x + map_size and map_y < my < map_y + map_size):
                        continue

                    if cost >= COST_INSCRIBED:
                        glColor4f(1.0, 0.2, 0.1, 0.6)
                    else:
                        glColor4f(1.0, 0.6, 0.1, 0.3)

                    hr = max(1.0, self.res * scale * 0.5)
                    glVertex2f(mx - hr, my - hr)
                    glVertex2f(mx + hr, my - hr)
                    glVertex2f(mx + hr, my + hr)
                    glVertex2f(mx - hr, my + hr)
            glEnd()


# ═══════════════════════════════════════════════
#  사운드 시스템 (절차적 생성)
# ═══════════════════════════════════════════════

class SoundSystem:
    """
    절차적 사운드 생성 (외부 파일 불필요)
    numpy로 파형 생성 → pygame.mixer.Sound로 재생
    """

    def __init__(self):
        self.enabled = False
        self.sounds = {}
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            self._generate_sounds()
            self.enabled = True
        except Exception:
            pass  # 사운드 없이도 동작

    def _make_sound(self, samples):
        """numpy float array → pygame Sound"""
        samples = np.clip(samples, -1.0, 1.0)
        data = (samples * 32767).astype(np.int16)
        return pygame.mixer.Sound(buffer=data.tobytes())

    def _generate_sounds(self):
        """모든 사운드 절차적 생성"""
        sr = 22050  # sample rate

        # 1. 경보음 (400Hz → 800Hz 사이렌, 1.5초)
        t = np.linspace(0, 1.5, int(sr * 1.5), dtype=np.float32)
        freq = 400 + 400 * np.sin(2 * np.pi * 2 * t)  # 2Hz 사이렌
        wave = 0.4 * np.sin(2 * np.pi * freq * t)
        # 페이드인/아웃
        fade = np.ones_like(t)
        fade_len = int(sr * 0.05)
        fade[:fade_len] = np.linspace(0, 1, fade_len)
        fade[-fade_len:] = np.linspace(1, 0, fade_len)
        self.sounds['alert'] = self._make_sound(wave * fade)

        # 2. 모터 루프 (낮은 험) (0.5초 루프)
        t = np.linspace(0, 0.5, int(sr * 0.5), dtype=np.float32)
        wave = 0.08 * np.sin(2 * np.pi * 80 * t)   # 80Hz 기본
        wave += 0.04 * np.sin(2 * np.pi * 160 * t)  # 2배음
        wave += 0.02 * np.random.randn(len(t)).astype(np.float32)  # 노이즈
        self.sounds['motor'] = self._make_sound(wave)

        # 3. LiDAR 스캔 틱 (짧은 클릭, 0.05초)
        t = np.linspace(0, 0.05, int(sr * 0.05), dtype=np.float32)
        wave = 0.15 * np.sin(2 * np.pi * 2000 * t)
        env = np.exp(-t * 80)  # 빠른 감쇠
        self.sounds['lidar_tick'] = self._make_sound(wave * env)

        # 4. 웨이포인트 도달 (상승 차임, 0.3초)
        t = np.linspace(0, 0.3, int(sr * 0.3), dtype=np.float32)
        freq = 600 + 400 * t / 0.3  # 600→1000Hz 상승
        wave = 0.25 * np.sin(2 * np.pi * freq * t)
        env = np.exp(-t * 5)
        self.sounds['waypoint'] = self._make_sound(wave * env)

        # 5. 순찰 시작 비프 (0.15초)
        t = np.linspace(0, 0.15, int(sr * 0.15), dtype=np.float32)
        wave = 0.2 * np.sin(2 * np.pi * 880 * t)
        env = 1.0 - t / 0.15
        self.sounds['patrol_start'] = self._make_sound(wave * env)

        # 6. 순찰 정지 비프 (0.2초, 하강)
        t = np.linspace(0, 0.2, int(sr * 0.2), dtype=np.float32)
        freq = 660 - 200 * t / 0.2
        wave = 0.2 * np.sin(2 * np.pi * freq * t)
        env = 1.0 - t / 0.2
        self.sounds['patrol_stop'] = self._make_sound(wave * env)

    def play(self, name, loops=0):
        """사운드 재생 (loops=-1 무한반복)"""
        if self.enabled and name in self.sounds:
            self.sounds[name].play(loops=loops)

    def stop(self, name):
        """사운드 정지"""
        if self.enabled and name in self.sounds:
            self.sounds[name].stop()

    def toggle(self):
        """사운드 ON/OFF"""
        self.enabled = not self.enabled
        if not self.enabled:
            pygame.mixer.stop()


# ═══════════════════════════════════════════════
#  정적 메시 캐시 (DisplayList)
# ═══════════════════════════════════════════════

class StaticMeshCache:
    """변하지 않는 벽/바닥/장애물을 DisplayList로 캐싱"""

    def __init__(self):
        self.walls_list = None
        self.floor_list = None
        self.obstacles_list = None

    def build(self, walls, obstacles):
        """모든 정적 메시를 DisplayList로 컴파일"""
        # 바닥 + 그리드
        self.floor_list = glGenLists(1)
        glNewList(self.floor_list, GL_COMPILE)
        draw_floor()
        glEndList()

        # 벽
        self.walls_list = glGenLists(1)
        glNewList(self.walls_list, GL_COMPILE)
        for wx, wz in walls:
            draw_box(wx, WALL_HEIGHT / 2, wz, GRID_SIZE * 0.9, WALL_HEIGHT, GRID_SIZE * 0.9,
                     COL_WALL, COL_WALL_TOP)
        glEndList()

        # 장애물
        self.obstacles_list = glGenLists(1)
        glNewList(self.obstacles_list, GL_COMPILE)
        for ox, oz in obstacles:
            h = 0.8 + random.Random(hash((ox, oz))).random() * 0.6
            draw_box(ox, h / 2, oz, 0.8, h, 0.8, COL_OBSTACLE)
        glEndList()

    def draw_all(self):
        """캐시된 정적 메시 모두 렌더링"""
        if self.floor_list:
            glCallList(self.floor_list)
        if self.walls_list:
            glCallList(self.walls_list)
        if self.obstacles_list:
            glCallList(self.obstacles_list)

    def cleanup(self):
        for dl in [self.walls_list, self.floor_list, self.obstacles_list]:
            if dl:
                glDeleteLists(dl, 1)


# ═══════════════════════════════════════════════
#  ROS2 토픽 시뮬레이터
# ═══════════════════════════════════════════════

class TopicMonitor:
    """ROS2 스타일 토픽 상태 모니터"""

    def __init__(self):
        self.topics = {
            "/cmd_vel": {"type": "Twist", "hz": 0.0, "data": "linear: 0.0, angular: 0.0", "active": True},
            "/scan": {"type": "LaserScan", "hz": 0.0, "data": "ranges: 360 pts", "active": True},
            "/image_raw": {"type": "Image", "hz": 0.0, "data": "640x480 rgb8", "active": True},
            "/odom": {"type": "Odometry", "hz": 0.0, "data": "x:0.0 y:0.0", "active": True},
            "/detection": {"type": "Detection", "hz": 0.0, "data": "idle", "active": False},
            "/alert": {"type": "String", "hz": 0.0, "data": "no alert", "active": False},
            "/robot_status": {"type": "String", "hz": 0.0, "data": "patrol", "active": True},
            "/nav2/path": {"type": "Path", "hz": 0.0, "data": "waypoints: 10", "active": False},
            "/map": {"type": "OccupancyGrid", "hz": 0.0, "data": "building...", "active": False},
            "/slam/status": {"type": "String", "hz": 0.0, "data": "idle", "active": False},
        }
        self.last_update = time.time()

    def update(self, robot_pos, robot_yaw, speed, patrol_mode, alert_active,
               slam_active=False, exploration_pct=0.0):
        now = time.time()
        dt = now - self.last_update
        self.last_update = now

        self.topics["/cmd_vel"]["hz"] = 50.0 if speed > 0.01 else 0.0
        self.topics["/cmd_vel"]["data"] = f"lin:{speed:.2f} ang:{robot_yaw:.1f}"
        self.topics["/scan"]["hz"] = 10.0
        self.topics["/scan"]["data"] = "ranges: 360 pts"
        self.topics["/image_raw"]["hz"] = 30.0
        self.topics["/odom"]["hz"] = 50.0
        self.topics["/odom"]["data"] = f"x:{robot_pos[0]:.1f} y:{robot_pos[1]:.1f}"
        self.topics["/robot_status"]["data"] = "auto_patrol" if patrol_mode else "manual"
        self.topics["/nav2/path"]["active"] = patrol_mode
        self.topics["/nav2/path"]["hz"] = 1.0 if patrol_mode else 0.0

        if alert_active:
            self.topics["/detection"]["active"] = True
            self.topics["/detection"]["hz"] = 10.0
            self.topics["/detection"]["data"] = "FALL DETECTED!"
            self.topics["/alert"]["active"] = True
            self.topics["/alert"]["hz"] = 1.0
            self.topics["/alert"]["data"] = "⚠ EMERGENCY"
        else:
            self.topics["/detection"]["data"] = "monitoring..."
            self.topics["/detection"]["hz"] = 10.0
            self.topics["/alert"]["active"] = False
            self.topics["/alert"]["hz"] = 0.0
            self.topics["/alert"]["data"] = "no alert"

        # SLAM 토픽
        self.topics["/map"]["active"] = slam_active
        self.topics["/map"]["hz"] = 1.0 if slam_active else 0.0
        self.topics["/map"]["data"] = f"explored:{exploration_pct:.0f}%" if slam_active else "inactive"
        self.topics["/slam/status"]["active"] = slam_active
        self.topics["/slam/status"]["hz"] = 1.0 if slam_active else 0.0
        self.topics["/slam/status"]["data"] = "mapping" if slam_active else "idle"


# ═══════════════════════════════════════════════
#  3D 렌더링 유틸리티
# ═══════════════════════════════════════════════

def draw_box(x, y, z, sx, sy, sz, color, top_color=None):
    """3D 박스 그리기"""
    if top_color is None:
        top_color = color

    hx, hy, hz = sx / 2, sy / 2, sz / 2

    # 앞면
    glColor4f(*color)
    glBegin(GL_QUADS)
    glVertex3f(x - hx, y - hy, z + hz)
    glVertex3f(x + hx, y - hy, z + hz)
    glVertex3f(x + hx, y + hy, z + hz)
    glVertex3f(x - hx, y + hy, z + hz)
    glEnd()

    # 뒷면
    glBegin(GL_QUADS)
    glVertex3f(x - hx, y - hy, z - hz)
    glVertex3f(x + hx, y - hy, z - hz)
    glVertex3f(x + hx, y + hy, z - hz)
    glVertex3f(x - hx, y + hy, z - hz)
    glEnd()

    # 왼쪽
    c2 = tuple(min(1.0, v * 0.8) for v in color[:3]) + (color[3],)
    glColor4f(*c2)
    glBegin(GL_QUADS)
    glVertex3f(x - hx, y - hy, z - hz)
    glVertex3f(x - hx, y - hy, z + hz)
    glVertex3f(x - hx, y + hy, z + hz)
    glVertex3f(x - hx, y + hy, z - hz)
    glEnd()

    # 오른쪽
    glBegin(GL_QUADS)
    glVertex3f(x + hx, y - hy, z - hz)
    glVertex3f(x + hx, y - hy, z + hz)
    glVertex3f(x + hx, y + hy, z + hz)
    glVertex3f(x + hx, y + hy, z - hz)
    glEnd()

    # 윗면
    glColor4f(*top_color)
    glBegin(GL_QUADS)
    glVertex3f(x - hx, y + hy, z - hz)
    glVertex3f(x + hx, y + hy, z - hz)
    glVertex3f(x + hx, y + hy, z + hz)
    glVertex3f(x - hx, y + hy, z + hz)
    glEnd()

    # 아랫면
    glColor4f(*color)
    glBegin(GL_QUADS)
    glVertex3f(x - hx, y - hy, z - hz)
    glVertex3f(x + hx, y - hy, z - hz)
    glVertex3f(x + hx, y - hy, z + hz)
    glVertex3f(x - hx, y - hy, z + hz)
    glEnd()


def draw_cylinder(x, y, z, radius, height, segments, color):
    """원기둥 근사 (다각형)"""
    glColor4f(*color)

    # 측면
    glBegin(GL_QUAD_STRIP)
    for i in range(segments + 1):
        angle = 2 * math.pi * i / segments
        nx = math.cos(angle) * radius
        nz = math.sin(angle) * radius
        glVertex3f(x + nx, y, z + nz)
        glVertex3f(x + nx, y + height, z + nz)
    glEnd()

    # 윗면
    glBegin(GL_TRIANGLE_FAN)
    glVertex3f(x, y + height, z)
    for i in range(segments + 1):
        angle = 2 * math.pi * i / segments
        glVertex3f(x + math.cos(angle) * radius, y + height, z + math.sin(angle) * radius)
    glEnd()


def draw_robot(x, y, z, yaw, lidar_on=True):
    """Kevin 로봇 모델"""
    glPushMatrix()
    glTranslatef(x, y, z)
    glRotatef(math.degrees(yaw), 0, 1, 0)

    # 본체 (사각형 베이스)
    draw_box(0, 0.3, 0, 0.8, 0.35, 1.0, COL_ROBOT, COL_ROBOT_ACCENT)

    # 바퀴 4개
    wheel_color = (0.2, 0.2, 0.25, 1.0)
    for wx, wz in [(-0.45, -0.35), (0.45, -0.35), (-0.45, 0.35), (0.45, 0.35)]:
        draw_box(wx, 0.08, wz, 0.1, 0.16, 0.2, wheel_color)

    # 카메라 마운트
    draw_box(0, 0.6, 0.35, 0.2, 0.15, 0.15, (0.15, 0.15, 0.2, 1.0))
    # 카메라 렌즈
    draw_box(0, 0.6, 0.44, 0.08, 0.08, 0.05, (0.1, 0.1, 0.4, 1.0))

    # LiDAR 센서 (상단 회전체)
    if lidar_on:
        t = time.time()
        lidar_color = (0.0, 0.8 + 0.2 * math.sin(t * 5), 0.3, 0.9)
    else:
        lidar_color = (0.3, 0.3, 0.3, 1.0)
    draw_cylinder(0, 0.5, 0, 0.12, 0.1, 12, lidar_color)

    # 상태 LED
    t = time.time()
    led_brightness = 0.5 + 0.5 * math.sin(t * 3)
    led_color = (0.0, led_brightness, 0.0, 1.0)
    draw_box(0.3, 0.5, 0.4, 0.05, 0.05, 0.05, led_color)
    draw_box(-0.3, 0.5, 0.4, 0.05, 0.05, 0.05, led_color)

    glPopMatrix()


def draw_person(x, y, z, fallen=False):
    """사람 모델 (간단한 스틱 피규어)"""
    glPushMatrix()
    glTranslatef(x, y, z)

    if fallen:
        # 쓰러진 상태
        glRotatef(90, 0, 0, 1)
        glTranslatef(0.5, -0.3, 0)

    # 몸통
    draw_box(0, 0.6, 0, 0.3, 0.5, 0.2, COL_PERSON)
    # 머리
    head_color = (0.85, 0.7, 0.55, 1.0)
    draw_box(0, 1.05, 0, 0.2, 0.25, 0.2, head_color)
    # 다리
    leg_color = (0.3, 0.3, 0.5, 1.0)
    draw_box(-0.08, 0.15, 0, 0.12, 0.35, 0.15, leg_color)
    draw_box(0.08, 0.15, 0, 0.12, 0.35, 0.15, leg_color)

    glPopMatrix()


def raycast_lidar(robot_x, robot_z, robot_yaw, spatial_hash, num_rays=72, max_range=12.0):
    """공간 해시 기반 LiDAR 레이캐스트 — 결과 캐시 반환"""
    results = []
    for i in range(num_rays):
        angle = robot_yaw + (2 * math.pi * i / num_rays)
        dx = math.sin(angle)
        dz = math.cos(angle)

        hit_dist = max_range
        for step_f in range(1, int(max_range * 5)):
            step = step_f / 5.0
            px = robot_x + dx * step
            pz = robot_z + dz * step

            near = spatial_hash.query(px, pz)
            hit = False
            for nx, nz, ntype, nrad in near:
                if abs(px - nx) < nrad and abs(pz - nz) < nrad:
                    hit_dist = step
                    hit = True
                    break
            if hit:
                break

        end_x = robot_x + dx * hit_dist
        end_z = robot_z + dz * hit_dist
        results.append((hit_dist, end_x, end_z))
    return results


def draw_lidar_rays_cached(robot_x, robot_z, lidar_results, max_range=12.0):
    """캐시된 레이캐스트 결과로 LiDAR 시각화"""
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glLineWidth(1.0)

    glBegin(GL_LINES)
    for hit_dist, end_x, end_z in lidar_results:
        ratio = hit_dist / max_range
        glColor4f(0.0, 1.0, 0.3, 0.4 * (1.0 - ratio * 0.7))
        glVertex3f(robot_x, 0.55, robot_z)
        glColor4f(0.0, 1.0, 0.3, 0.1)
        glVertex3f(end_x, 0.55, end_z)
    glEnd()

    # 히트 포인트
    glPointSize(3.0)
    glBegin(GL_POINTS)
    for hit_dist, end_x, end_z in lidar_results:
        if hit_dist < max_range:
            glColor4f(0.0, 1.0, 0.5, 0.8)
            glVertex3f(end_x, 0.55, end_z)
    glEnd()
    glPointSize(1.0)


def draw_lidar_rays(robot_x, robot_z, robot_yaw, walls, obstacles, num_rays=72):
    """LiDAR 레이 시각화"""
    max_range = 12.0

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glLineWidth(1.0)

    glBegin(GL_LINES)
    for i in range(num_rays):
        angle = robot_yaw + (2 * math.pi * i / num_rays)
        dx = math.sin(angle)
        dz = math.cos(angle)

        # 간단한 레이캐스트
        hit_dist = max_range
        for step_f in range(1, int(max_range * 5)):
            step = step_f / 5.0
            px = robot_x + dx * step
            pz = robot_z + dz * step

            # 벽 충돌 체크
            hit = False
            for wx, wz in walls:
                if abs(px - wx) < 1.2 and abs(pz - wz) < 1.2:
                    hit_dist = step
                    hit = True
                    break
            if not hit:
                for ox, oz in obstacles:
                    if abs(px - ox) < 0.8 and abs(pz - oz) < 0.8:
                        hit_dist = step
                        hit = True
                        break
            if hit:
                break

        # 레이 그리기
        end_x = robot_x + dx * hit_dist
        end_z = robot_z + dz * hit_dist

        # 거리에 따른 색상 변화
        ratio = hit_dist / max_range
        glColor4f(0.0, 1.0, 0.3, 0.4 * (1.0 - ratio * 0.7))
        glVertex3f(robot_x, 0.55, robot_z)
        glColor4f(0.0, 1.0, 0.3, 0.1)
        glVertex3f(end_x, 0.55, end_z)

    glEnd()

    # 히트 포인트 표시
    glPointSize(3.0)
    glBegin(GL_POINTS)
    for i in range(num_rays):
        angle = robot_yaw + (2 * math.pi * i / num_rays)
        dx = math.sin(angle)
        dz = math.cos(angle)

        hit_dist = max_range
        for step_f in range(1, int(max_range * 5)):
            step = step_f / 5.0
            px = robot_x + dx * step
            pz = robot_z + dz * step
            hit = False
            for wx, wz in walls:
                if abs(px - wx) < 1.2 and abs(pz - wz) < 1.2:
                    hit_dist = step
                    hit = True
                    break
            if not hit:
                for ox, oz in obstacles:
                    if abs(px - ox) < 0.8 and abs(pz - oz) < 0.8:
                        hit_dist = step
                        hit = True
                        break
            if hit:
                break

        if hit_dist < max_range:
            end_x = robot_x + dx * hit_dist
            end_z = robot_z + dz * hit_dist
            glColor4f(0.0, 1.0, 0.5, 0.8)
            glVertex3f(end_x, 0.55, end_z)
    glEnd()
    glPointSize(1.0)


def draw_waypoint(x, z, active=False, index=0):
    """웨이포인트 마커"""
    color = COL_WAYPOINT_ACTIVE if active else COL_WAYPOINT
    t = time.time()

    # 아래 원 표시
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    pulse = 0.3 + 0.15 * math.sin(t * 3 + index)
    glColor4f(*color[:3], 0.3)
    glBegin(GL_TRIANGLE_FAN)
    glVertex3f(x, 0.02, z)
    for i in range(25):
        angle = 2 * math.pi * i / 24
        r = 0.5 + pulse
        glVertex3f(x + math.cos(angle) * r, 0.02, z + math.sin(angle) * r)
    glEnd()

    # 수직 마커
    bob = 0.1 * math.sin(t * 2 + index)
    draw_box(x, 1.2 + bob, z, 0.15, 0.15, 0.15, color)

    # 기둥
    glColor4f(*color[:3], 0.5)
    glLineWidth(2.0)
    glBegin(GL_LINES)
    glVertex3f(x, 0.02, z)
    glVertex3f(x, 1.1 + bob, z)
    glEnd()
    glLineWidth(1.0)


def draw_floor():
    """바닥 그리드"""
    glBegin(GL_QUADS)
    glColor4f(*COL_FLOOR)
    hs = MAP_SIZE // 2 + 2
    glVertex3f(-hs, 0, -hs)
    glVertex3f(hs, 0, -hs)
    glVertex3f(hs, 0, hs)
    glVertex3f(-hs, 0, hs)
    glEnd()

    # 그리드 라인
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glLineWidth(1.0)
    glBegin(GL_LINES)
    glColor4f(*COL_GRID)
    for i in range(-hs, hs + 1, 2):
        glVertex3f(i, 0.01, -hs)
        glVertex3f(i, 0.01, hs)
        glVertex3f(-hs, 0.01, i)
        glVertex3f(hs, 0.01, i)
    glEnd()


def draw_skybox():
    """간단한 그라데이션 스카이박스"""
    glDisable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(-1, 1, -1, 1, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glBegin(GL_QUADS)
    glColor3f(*COL_SKY_TOP)
    glVertex3f(-1, 1, -0.99)
    glVertex3f(1, 1, -0.99)
    glColor3f(*COL_SKY_BOT)
    glVertex3f(1, -1, -0.99)
    glVertex3f(-1, -1, -0.99)
    glEnd()

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)


# ═══════════════════════════════════════════════
#  HUD (2D 오버레이)
# ═══════════════════════════════════════════════

class HUD:
    def __init__(self):
        pygame.font.init()
        self.fonts = {}
        for size in [12, 14, 16, 18, 20, 24, 28, 32]:
            for font_name in ["NanumGothic", "NanumGothicBold", "Malgun Gothic",
                              "NotoSansCJK-Regular", "D2Coding", "Consolas", "Arial"]:
                try:
                    self.fonts[size] = pygame.font.SysFont(font_name, size)
                    break
                except:
                    continue
            if size not in self.fonts:
                self.fonts[size] = pygame.font.Font(None, size)

    def begin_2d(self, screen_w=WIDTH, screen_h=HEIGHT):
        """2D 렌더링 모드 시작"""
        self._screen_w = screen_w
        self._screen_h = screen_h
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, screen_w, screen_h, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def end_2d(self):
        """2D 렌더링 모드 종료"""
        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def draw_text(self, text, x, y, size=16, color=(240, 240, 245)):
        """텍스트를 텍스처로 렌더링"""
        font = self.fonts.get(size, self.fonts[16])
        surface = font.render(text, True, color)
        data = pygame.image.tostring(surface, "RGBA", True)
        w, h = surface.get_size()

        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)

        glEnable(GL_TEXTURE_2D)
        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(x, y)
        glTexCoord2f(1, 1); glVertex2f(x + w, y)
        glTexCoord2f(1, 0); glVertex2f(x + w, y + h)
        glTexCoord2f(0, 0); glVertex2f(x, y + h)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glDeleteTextures([tex])

        return w, h

    def draw_rect(self, x, y, w, h, color, alpha=0.8):
        """반투명 사각형"""
        glColor4f(color[0] / 255, color[1] / 255, color[2] / 255, alpha)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()

    def draw_topic_monitor(self, topic_monitor, x, y):
        """ROS2 토픽 모니터 패널"""
        panel_w = 320
        line_h = 20
        topics = topic_monitor.topics
        panel_h = 30 + len(topics) * line_h + 10

        # 패널 배경
        self.draw_rect(x, y, panel_w, panel_h, (15, 18, 30), 0.85)
        # 테두리
        glColor4f(0.2, 0.5, 0.8, 0.6)
        glLineWidth(1.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x, y)
        glVertex2f(x + panel_w, y)
        glVertex2f(x + panel_w, y + panel_h)
        glVertex2f(x, y + panel_h)
        glEnd()

        # 타이틀
        self.draw_text("📡 ROS2 Topic Monitor", x + 8, y + 5, 14, (100, 200, 255))

        # 토픽 목록
        ty = y + 28
        for name, info in topics.items():
            # 상태 인디케이터
            if info["active"]:
                self.draw_rect(x + 8, ty + 4, 8, 8, (50, 200, 120), 0.9)
            else:
                self.draw_rect(x + 8, ty + 4, 8, 8, (80, 80, 90), 0.5)

            # 토픽 이름
            color = (200, 200, 210) if info["active"] else (100, 100, 110)
            self.draw_text(name, x + 22, ty, 12, color)

            # Hz
            hz_text = f"{info['hz']:.0f}Hz"
            self.draw_text(hz_text, x + 160, ty, 12, (150, 150, 160))

            # 데이터
            data_color = (255, 100, 100) if "FALL" in info["data"] or "EMERGENCY" in info["data"] \
                else (130, 130, 140)
            self.draw_text(info["data"][:20], x + 210, ty, 11, data_color)

            ty += line_h

    def draw_minimap(self, robot_x, robot_z, robot_yaw, walls, obstacles, waypoints,
                     current_wp, x, y, size=180):
        """미니맵"""
        # 배경
        self.draw_rect(x, y, size, size, (15, 18, 30), 0.85)
        glColor4f(0.2, 0.5, 0.8, 0.4)
        glLineWidth(1.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x, y)
        glVertex2f(x + size, y)
        glVertex2f(x + size, y + size)
        glVertex2f(x, y + size)
        glEnd()

        self.draw_text("MAP", x + size // 2 - 15, y + 2, 12, (100, 200, 255))

        cx = x + size // 2
        cy = y + size // 2
        scale = size / (MAP_SIZE + 4)

        # 벽
        glPointSize(2.0)
        glBegin(GL_POINTS)
        glColor4f(0.4, 0.45, 0.55, 0.7)
        for wx, wz in walls:
            mx = cx + wx * scale
            my = cy + wz * scale
            if x < mx < x + size and y < my < y + size:
                glVertex2f(mx, my)
        glEnd()

        # 장애물
        glColor4f(0.6, 0.3, 0.3, 0.7)
        glBegin(GL_POINTS)
        for ox, oz in obstacles:
            mx = cx + ox * scale
            my = cy + oz * scale
            if x < mx < x + size and y < my < y + size:
                glVertex2f(mx, my)
        glEnd()

        # 웨이포인트
        glPointSize(4.0)
        glBegin(GL_POINTS)
        for i, (wx, wz) in enumerate(waypoints):
            if i == current_wp:
                glColor4f(1.0, 0.2, 0.2, 1.0)
            else:
                glColor4f(1.0, 0.6, 0.1, 0.6)
            mx = cx + wx * scale
            my = cy + wz * scale
            glVertex2f(mx, my)
        glEnd()

        # 로봇
        rx = cx + robot_x * scale
        ry = cy + robot_z * scale
        glPointSize(6.0)
        glBegin(GL_POINTS)
        glColor4f(0.2, 0.8, 1.0, 1.0)
        glVertex2f(rx, ry)
        glEnd()

        # 로봇 방향
        dir_len = 10
        dx = rx + math.sin(robot_yaw) * dir_len
        dy = ry + math.cos(robot_yaw) * dir_len
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor4f(0.2, 0.8, 1.0, 1.0)
        glVertex2f(rx, ry)
        glVertex2f(dx, dy)
        glEnd()
        glLineWidth(1.0)
        glPointSize(1.0)

    def draw_status_bar(self, patrol_mode, lidar_on, cam_mode, alert_active, fps_val,
                         screen_w=WIDTH, screen_h=HEIGHT,
                         slam_active=False, exploration_pct=0.0, slam_3d=True):
        """하단 상태 바"""
        bar_h = 32
        self.draw_rect(0, screen_h - bar_h, screen_w, bar_h, (15, 18, 30), 0.9)

        mode_text = "🤖 AUTO PATROL" if patrol_mode else "🎮 MANUAL"
        mode_color = (50, 200, 120) if patrol_mode else (200, 200, 210)
        self.draw_text(mode_text, 15, screen_h - bar_h + 7, 14, mode_color)

        lidar_text = "LiDAR: ON" if lidar_on else "LiDAR: OFF"
        lidar_color = (50, 255, 120) if lidar_on else (120, 120, 130)
        self.draw_text(lidar_text, 200, screen_h - bar_h + 7, 14, lidar_color)

        cam_names = {1: "1인칭", 2: "3인칭", 3: "탑뷰"}
        self.draw_text(f"CAM: {cam_names.get(cam_mode, '?')}", 330, screen_h - bar_h + 7, 14, (180, 180, 190))

        # SLAM 상태
        if slam_active:
            slam_color = (60, 210, 230) if slam_3d else (100, 150, 170)
            slam_text = f"SLAM:{exploration_pct:.0f}%"
            self.draw_text(slam_text, 460, screen_h - bar_h + 7, 14, slam_color)

        if alert_active:
            t = time.time()
            flash = int(t * 4) % 2 == 0
            if flash:
                self.draw_rect(580, screen_h - bar_h + 2, 180, bar_h - 4, (200, 30, 30), 0.9)
            self.draw_text("⚠ FALL DETECTED!", 590, screen_h - bar_h + 7, 14, (255, 255, 100))

        self.draw_text(f"FPS: {fps_val:.0f}", screen_w - 80, screen_h - bar_h + 7, 12, (100, 100, 110))

    def draw_controls_help(self, screen_w=WIDTH):
        """조작 도움말"""
        help_texts = [
            "WASD:이동  Mouse:시점  TAB:순찰모드  F11:전체화면",
            "L:LiDAR  F:낙상  M:미니맵  G:SLAM  V:Costmap  1-3:카메라",
            "P:A*경로  C:클리어  R:SLAM리셋  E:경로편집  ALT:마우스해제",
        ]
        y = 10
        for text in help_texts:
            self.draw_text(text, screen_w // 2 - 220, y, 12, (120, 120, 140))
            y += 16

    def draw_alert_overlay(self, screen_w=WIDTH, screen_h=HEIGHT):
        """낙상 감지 알림 오버레이"""
        t = time.time()
        alpha = 0.15 + 0.1 * math.sin(t * 6)
        self.draw_rect(0, 0, screen_w, screen_h, (255, 30, 30), alpha)

        # 중앙 알림
        cx = screen_w // 2
        self.draw_rect(cx - 200, 80, 400, 60, (180, 20, 20), 0.9)
        self.draw_text("⚠ 낙상 감지! 경보 발령 중", cx - 130, 95, 20, (255, 255, 100))
        self.draw_text("/alert → guard_brain → 긴급 알림 전송", cx - 155, 120, 14, (200, 200, 200))

    def draw_mouse_hint(self, screen_w=WIDTH, screen_h=HEIGHT):
        """마우스 해제 상태 안내"""
        cx = screen_w // 2
        cy = screen_h // 2

        # 반투명 중앙 안내
        self.draw_rect(cx - 160, cy - 25, 320, 50, (15, 18, 30), 0.75)
        glColor4f(0.3, 0.6, 1.0, 0.6)
        glLineWidth(1.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(cx - 160, cy - 25)
        glVertex2f(cx + 160, cy - 25)
        glVertex2f(cx + 160, cy + 25)
        glVertex2f(cx - 160, cy + 25)
        glEnd()

        self.draw_text("클릭하여 시점 제어 시작", cx - 95, cy - 15, 16, (200, 220, 255))
        self.draw_text("ALT: 마우스 해제  |  F11: 전체화면", cx - 130, cy + 3, 12, (140, 140, 160))


# ═══════════════════════════════════════════════
#  메인 시뮬레이션
# ═══════════════════════════════════════════════

class CameraMode(Enum):
    FIRST_PERSON = 1
    THIRD_PERSON = 2
    TOP_VIEW = 3


class Kevin3DSim:
    def __init__(self):
        pygame.init()

        # 디스플레이 설정
        self.fullscreen = False
        self.display_w = WIDTH
        self.display_h = HEIGHT
        self.screen = pygame.display.set_mode(
            (self.display_w, self.display_h), DOUBLEBUF | OPENGL | RESIZABLE
        )
        pygame.display.set_caption("Kevin 3D Patrol Simulator — ROS2 자율순찰 로봇")
        self.clock = pygame.time.Clock()

        # OpenGL 초기화
        self._init_gl()

        # 맵
        self.walls, self.obstacles = generate_map()

        # 공간 해시 (충돌 최적화)
        self.spatial_hash = build_spatial_hash(self.walls, self.obstacles)

        # 정적 메시 캐시 (DisplayList)
        self.mesh_cache = StaticMeshCache()
        self.mesh_cache.build(self.walls, self.obstacles)

        # SLAM
        self.slam = SLAMVisualizer()
        self.slam_active = True
        self.slam_3d_visible = True
        self.slam_update_counter = 0

        # LiDAR 결과 캐시
        self.lidar_results = []

        # A* 경로 탐색기
        self.path_planner = AStarPlanner(self.spatial_hash)
        self.astar_visible = True

        # Costmap
        self.costmap = CostmapVisualizer(self.spatial_hash)
        self.costmap_visible = True

        # 사운드
        self.sound = SoundSystem()

        # 웨이포인트 (편집 가능 리스트)
        self.waypoints = list(PATROL_WAYPOINTS)

        # 로봇 상태 — 시작 위치는 첫 WP에 직선 도달 가능한 곳
        self.robot_x = -5.0
        self.robot_z = -10.0
        self.robot_y = 0.0
        self.robot_yaw = math.atan2(
            self.waypoints[0][0] - self.robot_x,
            self.waypoints[0][1] - self.robot_z
        )  # 첫 WP 방향으로 초기 yaw
        self.robot_pitch = 0.0
        self.robot_speed = 0.0

        # 카메라
        self.cam_mode = CameraMode.THIRD_PERSON
        self.cam_distance = 5.0
        self.cam_height = 3.0
        self.mouse_captured = True

        # 게임 상태
        self.patrol_mode = False
        self.current_waypoint = 0
        self._stuck_counter = 0
        self.lidar_on = True
        self.minimap_on = True
        self.alert_active = False

        # 순찰 경로 편집기
        self.edit_mode = False
        self.edit_selected = -1      # 선택된 WP 인덱스
        self.edit_dragging = False
        self.edit_hover = -1         # 마우스 호버 중인 WP
        self._prev_cam_mode = CameraMode.THIRD_PERSON
        self._prev_mouse_captured = True
        self.alert_timer = 0
        self.fallen_person_idx = -1

        # 사람 상태
        self.persons = [(px, pz, False) for px, pz in PERSON_POSITIONS]

        # HUD
        self.hud = HUD()
        self.topic_monitor = TopicMonitor()

        # FPS
        self.fps_val = 60.0

        # 마우스 캡처 (시작 시 해제 상태, 화면 클릭으로 캡처)
        self.mouse_captured = False
        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)

    def _toggle_fullscreen(self):
        """전체 화면 / 창 모드 전환"""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            info = pygame.display.Info()
            self.display_w = info.current_w
            self.display_h = info.current_h
            self.screen = pygame.display.set_mode(
                (self.display_w, self.display_h),
                DOUBLEBUF | OPENGL | FULLSCREEN
            )
        else:
            self.display_w = WIDTH
            self.display_h = HEIGHT
            self.screen = pygame.display.set_mode(
                (self.display_w, self.display_h),
                DOUBLEBUF | OPENGL | RESIZABLE
            )
        # OpenGL 뷰포트 재설정
        glViewport(0, 0, self.display_w, self.display_h)
        self._init_gl()

    def _capture_mouse(self):
        """마우스를 화면에 캡처 (시점 이동 모드)"""
        if not self.mouse_captured:
            self.mouse_captured = True
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
            # 중앙으로 워프해서 초기 점프 방지
            pygame.mouse.set_pos(self.display_w // 2, self.display_h // 2)
            pygame.event.get(MOUSEMOTION)  # 잔여 모션 이벤트 소거

    def _release_mouse(self):
        """마우스 캡처 해제 (커서 자유 이동)"""
        if self.mouse_captured:
            self.mouse_captured = False
            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)

    def _init_gl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.05, 0.05, 0.12, 1.0)

        # 간단한 조명
        glEnable(GL_COLOR_MATERIAL)
        glShadeModel(GL_SMOOTH)

        # 투영 설정
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(65, WIDTH / HEIGHT, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def _check_collision(self, x, z):
        """공간 해시 기반 충돌 체크"""
        near = self.spatial_hash.query(x, z)
        for nx, nz, ntype, nrad in near:
            if abs(x - nx) < nrad and abs(z - nz) < nrad:
                return True
        return False

    def _auto_patrol(self):
        """자동 순찰 — LiDAR 기반 장애물 회피 + 스턱 복구"""
        if not self.patrol_mode or self.current_waypoint >= len(self.waypoints):
            return

        tx, tz = self.waypoints[self.current_waypoint]
        dx = tx - self.robot_x
        dz = tz - self.robot_z
        dist = math.sqrt(dx * dx + dz * dz)

        # 웨이포인트 도달 판정
        if dist < 1.5:
            self.current_waypoint = (self.current_waypoint + 1) % len(self.waypoints)
            self._stuck_counter = 0
            self.sound.play('waypoint')
            # 다음 WP까지 A* 경로 재계획
            next_wp = self.waypoints[self.current_waypoint]
            self.path_planner.plan(self.robot_x, self.robot_z, next_wp[0], next_wp[1])
            return

        # 목표 방향 계산
        target_yaw = math.atan2(dx, dz)
        yaw_diff = target_yaw - self.robot_yaw
        while yaw_diff > math.pi:
            yaw_diff -= 2 * math.pi
        while yaw_diff < -math.pi:
            yaw_diff += 2 * math.pi

        # ── LiDAR 기반 전방 장애물 감지 ──
        front_clear = True
        front_min_dist = 99.0
        avoid_dir = 0.0  # 회피 방향 (-: 좌, +: 우)

        # 전방 ±40° 범위 체크 (look-ahead 2.5m)
        check_range = 2.5
        left_space = 0.0
        right_space = 0.0

        for offset_deg in range(-40, 41, 10):
            angle = self.robot_yaw + math.radians(offset_deg)
            cx = math.sin(angle)
            cz = math.cos(angle)

            hit_dist = check_range
            for step_f in range(1, int(check_range * 5) + 1):
                step = step_f / 5.0
                px = self.robot_x + cx * step
                pz = self.robot_z + cz * step
                near = self.spatial_hash.query(px, pz)
                hit = False
                for nx, nz, ntype, nrad in near:
                    if abs(px - nx) < nrad and abs(pz - nz) < nrad:
                        hit_dist = step
                        hit = True
                        break
                if hit:
                    break

            if abs(offset_deg) <= 20 and hit_dist < 1.8:
                front_clear = False
                front_min_dist = min(front_min_dist, hit_dist)

            # 좌우 여유 공간 측정
            if offset_deg < 0:
                left_space += hit_dist
            elif offset_deg > 0:
                right_space += hit_dist

        # ── 회전 및 이동 결정 ──
        if front_clear:
            # 전방 클리어: 목표를 향해 회전 + 전진
            turn_rate = min(0.12, abs(yaw_diff) * 0.15)
            self.robot_yaw += math.copysign(turn_rate, yaw_diff)

            # 방향 차이가 크면 속도 줄임
            speed_factor = max(0.3, 1.0 - abs(yaw_diff) / math.pi)
            speed = min(MOVE_SPEED, dist * 0.1) * speed_factor

            new_x = self.robot_x + math.sin(self.robot_yaw) * speed
            new_z = self.robot_z + math.cos(self.robot_yaw) * speed

            if not self._check_collision(new_x, new_z):
                self.robot_x = new_x
                self.robot_z = new_z
                self.robot_speed = speed
                self._stuck_counter = 0
            else:
                self._stuck_counter += 1
                self.robot_speed = 0
        else:
            # 전방 장애물: 여유 공간이 넓은 쪽으로 회피
            if right_space > left_space:
                self.robot_yaw -= 0.08  # 우회전
            else:
                self.robot_yaw += 0.08  # 좌회전

            # 장애물이 매우 가까우면 후진
            if front_min_dist < 1.0:
                back_x = self.robot_x - math.sin(self.robot_yaw) * MOVE_SPEED * 0.5
                back_z = self.robot_z - math.cos(self.robot_yaw) * MOVE_SPEED * 0.5
                if not self._check_collision(back_x, back_z):
                    self.robot_x = back_x
                    self.robot_z = back_z

            self._stuck_counter += 1
            self.robot_speed = 0

        # ── 스턱 복구 (60프레임 ≈ 1초 이상 멈춤) ──
        if self._stuck_counter > 60:
            # 큰 각도로 회전하여 탈출 시도
            self.robot_yaw += 0.25 if right_space > left_space else -0.25
            back_x = self.robot_x - math.sin(self.robot_yaw) * MOVE_SPEED
            back_z = self.robot_z - math.cos(self.robot_yaw) * MOVE_SPEED
            if not self._check_collision(back_x, back_z):
                self.robot_x = back_x
                self.robot_z = back_z
            if self._stuck_counter > 120:
                # 2초 이상 스턱이면 다음 웨이포인트로 스킵
                self.current_waypoint = (self.current_waypoint + 1) % len(self.waypoints)
                self._stuck_counter = 0

    def _draw_edit_hud(self, w, h):
        """편집 모드 HUD 오버레이"""
        # 상단 배너
        self.hud.draw_rect(0, 0, w, 32, (20, 80, 200), 0.85)
        self.hud.draw_text("✏ 경로 편집 모드", w // 2 - 80, 6, 18, (255, 255, 255))
        self.hud.draw_text(f"WP: {len(self.waypoints)}개", w // 2 + 80, 8, 14, (200, 220, 255))

        # 도움말 패널 (좌측 상단)
        panel_y = 45
        self.hud.draw_rect(10, panel_y, 280, 110, (15, 18, 30), 0.85)
        helps = [
            ("좌클릭 빈 곳", "WP 추가"),
            ("좌클릭+드래그 WP", "WP 이동"),
            ("우클릭 WP", "WP 삭제"),
            ("E", "편집 종료"),
        ]
        for i, (key, desc) in enumerate(helps):
            y = panel_y + 10 + i * 24
            self.hud.draw_text(key, 20, y, 14, (100, 180, 255))
            self.hud.draw_text(desc, 170, y, 14, (200, 210, 230))

        # 선택된 WP 정보
        if 0 <= self.edit_selected < len(self.waypoints):
            wx, wz = self.waypoints[self.edit_selected]
            info = f"선택: WP{self.edit_selected} ({wx:.1f}, {wz:.1f})"
            self.hud.draw_rect(10, panel_y + 115, 280, 24, (80, 20, 20), 0.7)
            self.hud.draw_text(info, 20, panel_y + 118, 14, (255, 200, 200))

    # ───────────────────────────────────────
    #  순찰 경로 편집기
    # ───────────────────────────────────────

    def _toggle_edit_mode(self):
        """편집 모드 ON/OFF"""
        self.edit_mode = not self.edit_mode
        if self.edit_mode:
            # 순찰 중이면 정지
            if self.patrol_mode:
                self.patrol_mode = False
                self.sound.stop('motor')
                self.path_planner.clear()
            # 이전 상태 저장 → 탑뷰 + 마우스 해제
            self._prev_cam_mode = self.cam_mode
            self._prev_mouse_captured = self.mouse_captured
            self.cam_mode = CameraMode.TOP_VIEW
            self._release_mouse()
            self.edit_selected = -1
            self.edit_dragging = False
        else:
            # 편집 종료 → 이전 카메라 복원
            self.cam_mode = self._prev_cam_mode
            if self._prev_mouse_captured:
                self._capture_mouse()
            self.edit_selected = -1
            self.edit_hover = -1
            self.current_waypoint = 0

    def _screen_to_world(self, screen_pos):
        """탑뷰 기준 스크린 좌표 → 월드 (x, z) 변환"""
        w = self.display_w if hasattr(self, 'display_w') else WIDTH
        h = self.display_h if hasattr(self, 'display_h') else HEIGHT
        sx, sy = screen_pos

        # 탑뷰 카메라: gluLookAt(robot_x, 25, robot_z+0.01, robot_x, 0, robot_z, 0,0,-1)
        # 투영: gluPerspective(60, aspect, 0.1, 200)
        aspect = w / h
        fov_rad = math.radians(60)
        half_h_world = 25.0 * math.tan(fov_rad / 2)
        half_w_world = half_h_world * aspect

        # 스크린 좌표를 -1~1 NDC로 변환
        ndc_x = (sx / w) * 2.0 - 1.0
        ndc_y = 1.0 - (sy / h) * 2.0  # y 반전

        # 탑뷰에서 NDC → 월드 (카메라 중심 = robot_x, robot_z)
        world_x = self.robot_x + ndc_x * half_w_world
        world_z = self.robot_z - ndc_y * half_h_world  # z축은 화면 위가 +z

        return world_x, world_z

    def _find_wp_at(self, screen_pos, threshold=1.5):
        """스크린 좌표 근처의 웨이포인트 찾기"""
        wx, wz = self._screen_to_world(screen_pos)
        best_idx = -1
        best_dist = threshold
        for i, (px, pz) in enumerate(self.waypoints):
            d = math.sqrt((wx - px) ** 2 + (wz - pz) ** 2)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _edit_handle_click(self, screen_pos):
        """좌클릭: WP 선택 또는 새 WP 추가"""
        idx = self._find_wp_at(screen_pos)
        if idx >= 0:
            # 기존 WP 선택 → 드래그 시작
            self.edit_selected = idx
            self.edit_dragging = True
            self.sound.play('lidar_tick')
        else:
            # 빈 곳 클릭 → 새 WP 추가
            wx, wz = self._screen_to_world(screen_pos)
            # 충돌 검증
            if not self._check_collision(wx, wz):
                # 선택된 WP 바로 뒤에 삽입 (경로 순서 유지)
                insert_idx = self.edit_selected + 1 if self.edit_selected >= 0 else len(self.waypoints)
                self.waypoints.insert(insert_idx, (wx, wz))
                self.edit_selected = insert_idx
                self.sound.play('waypoint')

    def _edit_handle_right_click(self, screen_pos):
        """우클릭: WP 삭제"""
        idx = self._find_wp_at(screen_pos)
        if idx >= 0 and len(self.waypoints) > 2:
            self.waypoints.pop(idx)
            if self.edit_selected >= len(self.waypoints):
                self.edit_selected = len(self.waypoints) - 1
            self.sound.play('patrol_stop')

    def _edit_handle_drag(self, screen_pos):
        """드래그: 선택된 WP 이동"""
        if self.edit_selected < 0 or self.edit_selected >= len(self.waypoints):
            return
        wx, wz = self._screen_to_world(screen_pos)
        # 충돌 검증
        if not self._check_collision(wx, wz):
            self.waypoints[self.edit_selected] = (wx, wz)

    def _edit_update_hover(self, screen_pos):
        """마우스 호버 시 가까운 WP 하이라이트"""
        self.edit_hover = self._find_wp_at(screen_pos, threshold=2.0)

    def _trigger_fall_event(self):
        """낙상 이벤트 트리거"""
        # 가장 가까운 사람 찾기
        min_dist = float('inf')
        closest_idx = -1
        for i, (px, pz, fallen) in enumerate(self.persons):
            if not fallen:
                dist = math.sqrt((px - self.robot_x) ** 2 + (pz - self.robot_z) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i

        if closest_idx >= 0:
            px, pz, _ = self.persons[closest_idx]
            self.persons[closest_idx] = (px, pz, True)
            self.alert_active = True
            self.alert_timer = 300  # 5초
            self.fallen_person_idx = closest_idx
            self.sound.play('alert')

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            self.fps_val = self.clock.get_fps()

            # 이벤트 처리
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False

                elif event.type == VIDEORESIZE and not self.fullscreen:
                    self.display_w = event.w
                    self.display_h = event.h
                    self.screen = pygame.display.set_mode(
                        (self.display_w, self.display_h),
                        DOUBLEBUF | OPENGL | RESIZABLE
                    )
                    glViewport(0, 0, self.display_w, self.display_h)

                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        if self.mouse_captured:
                            self._release_mouse()
                        else:
                            running = False
                    elif event.key == K_F11:
                        self._toggle_fullscreen()
                    elif event.key in (K_LALT, K_RALT):
                        self._release_mouse()
                    elif event.key == K_TAB:
                        self.patrol_mode = not self.patrol_mode
                        if self.patrol_mode:
                            self.sound.play('patrol_start')
                            self.sound.play('motor', loops=-1)
                            # 순찰 시작 시 첫 WP까지 A* 경로 표시
                            wp_idx = self.current_waypoint % len(self.waypoints)
                            gx, gz = self.waypoints[wp_idx]
                            self.path_planner.plan(
                                self.robot_x, self.robot_z, gx, gz
                            )
                        else:
                            self.sound.play('patrol_stop')
                            self.sound.stop('motor')
                            self.path_planner.clear()
                    elif event.key == K_l:
                        self.lidar_on = not self.lidar_on
                    elif event.key == K_m:
                        self.minimap_on = not self.minimap_on
                    elif event.key == K_f:
                        self._trigger_fall_event()
                    elif event.key == K_g:
                        self.slam_3d_visible = not self.slam_3d_visible
                    elif event.key == K_r:
                        self.slam.reset()
                    elif event.key == K_p:
                        # 현재 위치 → 다음 웨이포인트까지 A* 경로 탐색
                        wp_idx = self.current_waypoint % len(self.waypoints)
                        gx, gz = self.waypoints[wp_idx]
                        self.path_planner.plan(
                            self.robot_x, self.robot_z, gx, gz
                        )
                    elif event.key == K_c:
                        self.path_planner.clear()
                    elif event.key == K_v:
                        self.costmap_visible = not self.costmap_visible
                    elif event.key == K_e:
                        self._toggle_edit_mode()
                    elif event.key == K_1:
                        self.cam_mode = CameraMode.FIRST_PERSON
                    elif event.key == K_2:
                        self.cam_mode = CameraMode.THIRD_PERSON
                    elif event.key == K_3:
                        self.cam_mode = CameraMode.TOP_VIEW

                elif event.type == MOUSEBUTTONDOWN and event.button == 1:
                    if self.edit_mode:
                        self._edit_handle_click(event.pos)
                    elif not self.mouse_captured:
                        self._capture_mouse()

                elif event.type == MOUSEBUTTONDOWN and event.button == 3:
                    # 우클릭: 편집 모드에서 WP 삭제
                    if self.edit_mode:
                        self._edit_handle_right_click(event.pos)

                elif event.type == MOUSEBUTTONUP and event.button == 1:
                    if self.edit_mode:
                        self.edit_dragging = False

                elif event.type == MOUSEMOTION:
                    if self.edit_mode and self.edit_dragging and self.edit_selected >= 0:
                        self._edit_handle_drag(event.pos)
                    elif self.edit_mode:
                        self._edit_update_hover(event.pos)
                    elif self.mouse_captured:
                        mx, my = event.rel
                        self.robot_yaw -= mx * MOUSE_SENSITIVITY * 0.02
                        self.robot_pitch = max(-60, min(60, self.robot_pitch - my * MOUSE_SENSITIVITY * 0.5))

            # 키보드 입력 (이동)
            if not self.patrol_mode:
                keys = pygame.key.get_pressed()
                move_x, move_z = 0, 0
                if keys[K_w]:
                    move_x += math.sin(self.robot_yaw) * MOVE_SPEED
                    move_z += math.cos(self.robot_yaw) * MOVE_SPEED
                if keys[K_s]:
                    move_x -= math.sin(self.robot_yaw) * MOVE_SPEED
                    move_z -= math.cos(self.robot_yaw) * MOVE_SPEED
                if keys[K_a]:
                    move_x += math.cos(self.robot_yaw) * MOVE_SPEED
                    move_z -= math.sin(self.robot_yaw) * MOVE_SPEED
                if keys[K_d]:
                    move_x -= math.cos(self.robot_yaw) * MOVE_SPEED
                    move_z += math.sin(self.robot_yaw) * MOVE_SPEED

                new_x = self.robot_x + move_x
                new_z = self.robot_z + move_z

                if not self._check_collision(new_x, new_z):
                    self.robot_x = new_x
                    self.robot_z = new_z
                    self.robot_speed = math.sqrt(move_x ** 2 + move_z ** 2)
                else:
                    self.robot_speed = 0
            else:
                self._auto_patrol()

            # 알림 타이머
            if self.alert_timer > 0:
                self.alert_timer -= 1
                if self.alert_timer <= 0:
                    self.alert_active = False
                    # 사람 복구
                    if self.fallen_person_idx >= 0:
                        px, pz, _ = self.persons[self.fallen_person_idx]
                        self.persons[self.fallen_person_idx] = (px, pz, False)
                        self.fallen_person_idx = -1

            # LiDAR 캐시 업데이트 (공간 해시 기반)
            self.lidar_results = raycast_lidar(
                self.robot_x, self.robot_z, self.robot_yaw,
                self.spatial_hash
            )

            # SLAM 업데이트 (3프레임마다)
            if self.slam_active:
                self.slam_update_counter += 1
                if self.slam_update_counter % 3 == 0:
                    self.slam.update_from_lidar(
                        self.robot_x, self.robot_z, self.robot_yaw,
                        self.spatial_hash
                    )

            # A* 애니메이션 업데이트
            self.path_planner.update_animation()

            # Local costmap 업데이트
            if self.costmap_visible and self.lidar_results:
                self.costmap.update_local(
                    self.robot_x, self.robot_z, self.lidar_results
                )

            # 토픽 업데이트
            self.topic_monitor.update(
                (self.robot_x, self.robot_z),
                self.robot_yaw,
                self.robot_speed,
                self.patrol_mode,
                self.alert_active,
                self.slam_active,
                self.slam.get_exploration_pct()
            )

            # 렌더링
            self._render()

        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)
        pygame.quit()

    def _render(self):
        w, h = self.display_w, self.display_h
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 스카이박스
        draw_skybox()

        # 카메라 설정
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(65, w / max(1, h), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        if self.cam_mode == CameraMode.FIRST_PERSON:
            eye_x = self.robot_x
            eye_y = 1.2
            eye_z = self.robot_z
            look_x = eye_x + math.sin(self.robot_yaw)
            look_y = eye_y + math.tan(math.radians(self.robot_pitch)) * 0.5
            look_z = eye_z + math.cos(self.robot_yaw)
            gluLookAt(eye_x, eye_y, eye_z, look_x, look_y, look_z, 0, 1, 0)

        elif self.cam_mode == CameraMode.THIRD_PERSON:
            cam_x = self.robot_x - math.sin(self.robot_yaw) * self.cam_distance
            cam_y = self.cam_height
            cam_z = self.robot_z - math.cos(self.robot_yaw) * self.cam_distance
            gluLookAt(cam_x, cam_y, cam_z,
                      self.robot_x, 0.5, self.robot_z,
                      0, 1, 0)

        elif self.cam_mode == CameraMode.TOP_VIEW:
            gluLookAt(self.robot_x, 25, self.robot_z + 0.01,
                      self.robot_x, 0, self.robot_z,
                      0, 0, -1)

        # 바닥 + 벽 + 장애물 (DisplayList 캐시)
        self.mesh_cache.draw_all()

        # SLAM 3D 시각화
        if self.slam_3d_visible and self.slam_active:
            self.slam.draw_3d()

        # A* 경로 탐색 3D 시각화
        if self.astar_visible:
            self.path_planner.draw_3d()

        # Costmap 3D 시각화
        if self.costmap_visible:
            self.costmap.draw_3d_global()
            self.costmap.draw_3d_local()

        # 웨이포인트
        for i, (wx, wz) in enumerate(self.waypoints):
            active = (i == self.current_waypoint) and self.patrol_mode
            # 편집 모드: 선택/호버 시각화
            if self.edit_mode:
                if i == self.edit_selected:
                    draw_waypoint(wx, wz, True, i)  # 선택 = 빨간색
                elif i == self.edit_hover:
                    draw_waypoint(wx, wz, True, i)  # 호버 = 빨간색
                else:
                    draw_waypoint(wx, wz, False, i)
            else:
                draw_waypoint(wx, wz, active, i)

        # 경로 연결 라인 (편집 모드 또는 순찰 모드)
        if (self.edit_mode or self.patrol_mode) and len(self.waypoints) >= 2:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glLineWidth(2.0)
            glBegin(GL_LINE_LOOP)
            t = time.time()
            for i, (wx, wz) in enumerate(self.waypoints):
                pulse = 0.5 + 0.3 * math.sin(t * 2 + i * 0.5)
                if self.edit_mode:
                    glColor4f(0.2, 0.8, 1.0, 0.5 * pulse)
                else:
                    glColor4f(1.0, 0.6, 0.1, 0.3 * pulse)
                glVertex3f(wx, 0.1, wz)
            glEnd()
            glLineWidth(1.0)

        # 사람
        for px, pz, fallen in self.persons:
            draw_person(px, 0, pz, fallen)

        # LiDAR (캐시된 결과 사용)
        if self.lidar_on and self.lidar_results:
            draw_lidar_rays_cached(self.robot_x, self.robot_z, self.lidar_results)

        # 로봇 (1인칭이 아닌 경우에만)
        if self.cam_mode != CameraMode.FIRST_PERSON:
            draw_robot(self.robot_x, self.robot_y, self.robot_z, self.robot_yaw, self.lidar_on)

        # HUD
        self.hud.begin_2d(w, h)

        # 조작 도움말
        self.hud.draw_controls_help(w)

        # 토픽 모니터
        self.hud.draw_topic_monitor(self.topic_monitor, w - 335, 40)

        # 미니맵
        if self.minimap_on:
            minimap_x = 15
            minimap_y = h - 230
            minimap_size = 190
            self.hud.draw_minimap(
                self.robot_x, self.robot_z, self.robot_yaw,
                self.walls, self.obstacles,
                self.waypoints, self.current_waypoint,
                minimap_x, minimap_y, minimap_size
            )
            # SLAM 오버레이
            if self.slam_active:
                self.slam.draw_on_minimap(self.hud, minimap_x, minimap_y, minimap_size)
            # A* 경로 오버레이
            if self.astar_visible:
                self.path_planner.draw_on_minimap(self.hud, minimap_x, minimap_y, minimap_size)
            # Costmap 오버레이
            if self.costmap_visible:
                self.costmap.draw_on_minimap(
                    self.hud, minimap_x, minimap_y, minimap_size,
                    self.robot_x, self.robot_z
                )

        # 상태 바
        self.hud.draw_status_bar(
            self.patrol_mode, self.lidar_on,
            self.cam_mode.value, self.alert_active, self.fps_val,
            w, h, self.slam_active, self.slam.get_exploration_pct(),
            self.slam_3d_visible
        )

        # 낙상 알림 오버레이
        if self.alert_active:
            self.hud.draw_alert_overlay(w, h)

        # 마우스 해제 상태일 때 안내
        if not self.mouse_captured and not self.edit_mode:
            self.hud.draw_mouse_hint(w, h)

        # 편집 모드 HUD
        if self.edit_mode:
            self._draw_edit_hud(w, h)

        self.hud.end_2d()

        pygame.display.flip()


# ═══════════════════════════════════════════════
#  실행
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    sim = Kevin3DSim()
    sim.run()
