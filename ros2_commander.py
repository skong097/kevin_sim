"""
ROS2 Commander — ROS2 학습 게임 (Pygame)
3가지 모드:
  1. Memory Match  — ROS2 개념 쌍 카드 매칭
  2. Command Rush  — CLI 명령어 타임어택
  3. Node Builder  — 아키텍처 연결 퍼즐

Kevin 자율순찰 로봇 프로젝트 시나리오 기반

Requirements:
  pip install pygame

Usage:
  python ros2_commander.py
"""

import pygame
import sys
import random
import time
import math
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

# ═══════════════════════════════════════════════
#  상수 / 색상
# ═══════════════════════════════════════════════

WIDTH, HEIGHT = 1280, 800
FPS = 60

# 색상
BLACK      = (15, 15, 25)
WHITE      = (240, 240, 245)
GRAY       = (60, 65, 80)
LIGHT_GRAY = (100, 105, 120)
DARK_BG    = (22, 24, 35)
CARD_BG    = (35, 40, 60)
CARD_FACE  = (55, 65, 95)
CARD_MATCH = (40, 160, 90)
CARD_WRONG = (200, 60, 60)

# 테마 색상
ACCENT_BLUE   = (60, 140, 255)
ACCENT_GREEN  = (50, 200, 120)
ACCENT_ORANGE = (255, 160, 40)
ACCENT_RED    = (240, 70, 70)
ACCENT_PURPLE = (160, 100, 240)
ACCENT_CYAN   = (60, 210, 230)

# Tier 색상
TIER_COLORS = {
    1: ACCENT_BLUE,
    2: ACCENT_GREEN,
    3: ACCENT_ORANGE,
    4: ACCENT_PURPLE,
}


# ═══════════════════════════════════════════════
#  ROS2 학습 데이터
# ═══════════════════════════════════════════════

# Tier 1: 기본 개념
MATCH_DATA_T1 = [
    ("Node", "ROS2 최소 실행 단위\n모든 기능의 기본 블록"),
    ("Topic", "비동기 Pub/Sub\n데이터 스트리밍 채널"),
    ("Publisher", "토픽에 메시지를\n발행하는 노드"),
    ("Subscriber", "토픽에서 메시지를\n수신하는 노드"),
    ("Service", "요청-응답 방식\n동기 통신"),
    ("Action", "장시간 작업 + 피드백\n비동기 Goal/Result"),
    ("Message", "노드 간 주고받는\n데이터 구조체"),
    ("Package", "관련 노드/설정을\n묶는 빌드 단위"),
    ("Workspace", "여러 패키지를 포함하는\ncolcon 빌드 공간"),
    ("rclpy", "ROS2 Python\n클라이언트 라이브러리"),
    ("QoS", "통신 품질 정책\nReliable/BestEffort"),
    ("Callback", "메시지 수신 시\n호출되는 함수"),
]

# Tier 2: 시스템 구조
MATCH_DATA_T2 = [
    ("Launch File", "여러 노드를 한 번에\n실행하는 구성 파일"),
    ("Parameter", "런타임에 변경 가능한\n노드 설정값"),
    ("TF2", "좌표계 변환 프레임워크\nbase→camera 등"),
    ("URDF", "로봇 물리 구조 기술\n링크+조인트 XML"),
    ("Xacro", "URDF 매크로 확장\n변수/조건문 지원"),
    ("colcon", "ROS2 빌드 시스템\ncolcon build 명령"),
    ("setup.py", "Python 패키지의\n엔트리포인트 정의"),
    ("ament_cmake", "C++ 패키지용\n빌드 시스템"),
    ("DDS", "분산 통신 미들웨어\nROS2 하부 프로토콜"),
    ("Bag File", "토픽 데이터를\n녹화/재생하는 파일"),
    ("Lifecycle Node", "상태 머신 기반 노드\nConfiguring→Active"),
    ("Component", "단일 프로세스 내\n여러 노드 실행"),
]

# Tier 3: Nav2/SLAM
MATCH_DATA_T3 = [
    ("Nav2", "ROS2 자율 주행\n네비게이션 프레임워크"),
    ("SLAM", "동시 위치추정 및\n지도 작성"),
    ("Costmap", "장애물 정보를 담은\n격자 지도 레이어"),
    ("Global Planner", "출발→목표 전체\n경로 계획"),
    ("Local Planner", "장애물 회피하며\n실시간 경로 조정"),
    ("Recovery", "경로 실패 시\n복구 행동 (회전/후진)"),
    ("Waypoint", "순찰 경로의\n중간 지점 좌표"),
    ("Odometry", "바퀴 회전으로 추정한\n로봇 이동량"),
    ("LiDAR", "레이저 거리 센서\n360° 장애물 스캔"),
    ("map_server", "저장된 맵을\n로드/제공하는 노드"),
    ("AMCL", "적응적 몬테카를로\n위치 추정"),
    ("cmd_vel", "로봇 속도 명령\nlinear+angular"),
]

# Tier 4: 하드웨어
MATCH_DATA_T4 = [
    ("micro-ROS", "MCU에서 동작하는\n경량 ROS2 에이전트"),
    ("ESP32", "micro-ROS 지원\nWiFi MCU 보드"),
    ("image_transport", "카메라 영상 전송\n압축/원본 선택"),
    ("sensor_msgs", "IMU, LaserScan 등\n센서 표준 메시지"),
    ("geometry_msgs", "Pose, Twist 등\n기하학 메시지"),
    ("nav_msgs", "OccupancyGrid 등\n네비게이션 메시지"),
    ("Joint State", "모터 관절의\n위치/속도/토크"),
    ("robot_state_pub", "URDF → TF 변환\n자동 발행 노드"),
    ("rviz2", "3D 시각화 도구\nTF/맵/경로 표시"),
    ("Gazebo", "물리 시뮬레이터\n로봇/환경 테스트"),
    ("ros2_control", "하드웨어 추상화\n모터 인터페이스"),
    ("bridge", "ROS1↔ROS2\n메시지 변환"),
]

# 커맨드 러시 데이터
COMMAND_DATA = [
    # Tier 1: 기본 명령어
    {"tier": 1, "prompt": "노드 목록 확인", "answer": "ros2 node list", "hint": "ros2 node ..."},
    {"tier": 1, "prompt": "토픽 목록 확인", "answer": "ros2 topic list", "hint": "ros2 topic ..."},
    {"tier": 1, "prompt": "토픽 /cmd_vel 메시지 확인", "answer": "ros2 topic echo /cmd_vel", "hint": "ros2 topic echo ..."},
    {"tier": 1, "prompt": "서비스 목록 확인", "answer": "ros2 service list", "hint": "ros2 service ..."},
    {"tier": 1, "prompt": "turtlesim 노드 실행", "answer": "ros2 run turtlesim turtlesim_node", "hint": "ros2 run turtlesim ..."},
    {"tier": 1, "prompt": "토픽 /image_raw 의 타입 확인", "answer": "ros2 topic info /image_raw", "hint": "ros2 topic info ..."},
    {"tier": 1, "prompt": "토픽에 메시지 발행 (cmd_vel)", "answer": "ros2 topic pub /cmd_vel geometry_msgs/msg/Twist", "hint": "ros2 topic pub ..."},
    {"tier": 1, "prompt": "인터페이스 상세 보기 (Twist)", "answer": "ros2 interface show geometry_msgs/msg/Twist", "hint": "ros2 interface show ..."},

    # Tier 2: 시스템
    {"tier": 2, "prompt": "파라미터 목록 확인", "answer": "ros2 param list", "hint": "ros2 param ..."},
    {"tier": 2, "prompt": "파라미터 값 설정", "answer": "ros2 param set /node_name param_name value", "hint": "ros2 param set ..."},
    {"tier": 2, "prompt": "launch 파일 실행", "answer": "ros2 launch package_name launch_file.py", "hint": "ros2 launch ..."},
    {"tier": 2, "prompt": "워크스페이스 빌드", "answer": "colcon build", "hint": "colcon ..."},
    {"tier": 2, "prompt": "빌드 후 환경 소싱", "answer": "source install/setup.bash", "hint": "source install/..."},
    {"tier": 2, "prompt": "패키지 생성 (Python)", "answer": "ros2 pkg create --build-type ament_python pkg_name", "hint": "ros2 pkg create ..."},
    {"tier": 2, "prompt": "bag 파일 녹화", "answer": "ros2 bag record -a", "hint": "ros2 bag ..."},
    {"tier": 2, "prompt": "bag 파일 재생", "answer": "ros2 bag play bag_file", "hint": "ros2 bag play ..."},

    # Tier 3: Nav2
    {"tier": 3, "prompt": "Nav2 bringup 실행", "answer": "ros2 launch nav2_bringup navigation_launch.py", "hint": "ros2 launch nav2_bringup ..."},
    {"tier": 3, "prompt": "SLAM toolbox 실행", "answer": "ros2 launch slam_toolbox online_async_launch.py", "hint": "ros2 launch slam_toolbox ..."},
    {"tier": 3, "prompt": "맵 저장", "answer": "ros2 run nav2_map_server map_saver_cli -f map_name", "hint": "ros2 run nav2_map_server ..."},
    {"tier": 3, "prompt": "TF 트리 확인", "answer": "ros2 run tf2_tools view_frames", "hint": "ros2 run tf2_tools ..."},
    {"tier": 3, "prompt": "rviz2 실행", "answer": "ros2 run rviz2 rviz2", "hint": "ros2 run rviz2 ..."},
    {"tier": 3, "prompt": "Gazebo 시뮬레이터 실행", "answer": "ros2 launch gazebo_ros gazebo.launch.py", "hint": "ros2 launch gazebo_ros ..."},
]

# 노드 빌더 데이터 (Kevin 시나리오)
NODE_BUILDER_LEVELS = [
    {
        "level": 1,
        "title": "카메라 → 감지",
        "description": "카메라 노드에서 영상을 낙상감지 노드로 전달",
        "nodes": ["camera_node", "fall_detector"],
        "topics": ["/image_raw"],
        "correct_connections": [("camera_node", "/image_raw", "fall_detector")],
    },
    {
        "level": 2,
        "title": "센서 + 모터 제어",
        "description": "센서 데이터를 읽고 모터를 제어하는 구성",
        "nodes": ["sensor_node", "motor_controller", "nav2_controller"],
        "topics": ["/sensor_data", "/cmd_vel"],
        "correct_connections": [
            ("sensor_node", "/sensor_data", "nav2_controller"),
            ("nav2_controller", "/cmd_vel", "motor_controller"),
        ],
    },
    {
        "level": 3,
        "title": "Kevin 순찰 시스템",
        "description": "카메라+감지+LLM+모터 전체 연결",
        "nodes": ["camera_node", "fall_detector", "guard_brain", "motor_ctrl", "alert_node"],
        "topics": ["/image_raw", "/detection", "/cmd_vel", "/alert"],
        "correct_connections": [
            ("camera_node", "/image_raw", "fall_detector"),
            ("fall_detector", "/detection", "guard_brain"),
            ("guard_brain", "/cmd_vel", "motor_ctrl"),
            ("guard_brain", "/alert", "alert_node"),
        ],
    },
    {
        "level": 4,
        "title": "Kevin 풀 시스템",
        "description": "Nav2 + SLAM + 감지 + LLM + 대시보드 전체 구성",
        "nodes": ["camera", "lidar", "fall_detect", "face_detect", "guard_brain",
                  "nav2", "motor", "alert", "dashboard"],
        "topics": ["/image_raw", "/scan", "/fall_event", "/face_event",
                   "/cmd_vel", "/alert_msg", "/robot_status"],
        "correct_connections": [
            ("camera", "/image_raw", "fall_detect"),
            ("camera", "/image_raw", "face_detect"),
            ("lidar", "/scan", "nav2"),
            ("fall_detect", "/fall_event", "guard_brain"),
            ("face_detect", "/face_event", "guard_brain"),
            ("guard_brain", "/cmd_vel", "nav2"),
            ("nav2", "/cmd_vel", "motor"),
            ("guard_brain", "/alert_msg", "alert"),
            ("guard_brain", "/robot_status", "dashboard"),
        ],
    },
]


# ═══════════════════════════════════════════════
#  유틸리티
# ═══════════════════════════════════════════════

def draw_rounded_rect(surface, color, rect, radius=12, alpha=255):
    """둥근 모서리 사각형"""
    if alpha < 255:
        s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
        pygame.draw.rect(s, (*color, alpha), (0, 0, rect[2], rect[3]), border_radius=radius)
        surface.blit(s, (rect[0], rect[1]))
    else:
        pygame.draw.rect(surface, color, rect, border_radius=radius)


def draw_text(surface, text, pos, font, color=WHITE, center=False, max_width=0):
    """텍스트 렌더링 (멀티라인 지원)"""
    lines = text.split('\n')
    y = pos[1]
    for line in lines:
        rendered = font.render(line, True, color)
        if center:
            x = pos[0] - rendered.get_width() // 2
        else:
            x = pos[0]
        if max_width > 0 and rendered.get_width() > max_width:
            # 축소
            scale = max_width / rendered.get_width()
            new_w = int(rendered.get_width() * scale)
            new_h = int(rendered.get_height() * scale)
            rendered = pygame.transform.smoothscale(rendered, (new_w, new_h))
            if center:
                x = pos[0] - new_w // 2
        surface.blit(rendered, (x, y))
        y += rendered.get_height() + 2
    return y


def ease_out_back(t):
    """이징 함수"""
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * pow(t - 1, 3) + c1 * pow(t - 1, 2)


# ═══════════════════════════════════════════════
#  게임 상태
# ═══════════════════════════════════════════════

class GameState(Enum):
    MAIN_MENU = auto()
    MODE_SELECT = auto()
    MATCH_GAME = auto()
    MATCH_TIER_SELECT = auto()
    COMMAND_GAME = auto()
    COMMAND_TIER_SELECT = auto()
    NODE_GAME = auto()
    NODE_LEVEL_SELECT = auto()
    RESULT = auto()


class Button:
    """클릭 가능한 버튼"""
    def __init__(self, x, y, w, h, text, color=ACCENT_BLUE, text_color=WHITE,
                 font_size=24, radius=10):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.text_color = text_color
        self.font_size = font_size
        self.radius = radius
        self.hover = False
        self.alpha_anim = 0
        self._font = None

    def draw(self, surface, fonts):
        self.hover = self.rect.collidepoint(pygame.mouse.get_pos())
        c = tuple(min(255, v + 30) for v in self.color) if self.hover else self.color
        draw_rounded_rect(surface, c, self.rect, self.radius)
        if self.hover:
            draw_rounded_rect(surface, WHITE, self.rect, self.radius)
            draw_rounded_rect(surface, c,
                              (self.rect.x + 2, self.rect.y + 2,
                               self.rect.w - 4, self.rect.h - 4), self.radius - 1)

        font = fonts.get(self.font_size, fonts[24])
        draw_text(surface, self.text, (self.rect.centerx, self.rect.centery - font.get_height()//2),
                  font, self.text_color, center=True)

    def clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(event.pos)
        return False


# ═══════════════════════════════════════════════
#  Mode 1: Memory Match
# ═══════════════════════════════════════════════

class MatchCard:
    def __init__(self, x, y, w, h, text, pair_id, is_concept=True):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.pair_id = pair_id
        self.is_concept = is_concept
        self.flipped = False
        self.matched = False
        self.wrong_timer = 0
        self.flip_anim = 0  # 0=face down, 1=face up

    def draw(self, surface, fonts):
        if self.matched:
            draw_rounded_rect(surface, CARD_MATCH, self.rect, 10, 180)
            font = fonts[18]
            draw_text(surface, self.text,
                      (self.rect.centerx, self.rect.centery - 15),
                      font, WHITE, center=True, max_width=self.rect.w - 16)
            return

        if self.wrong_timer > 0:
            draw_rounded_rect(surface, CARD_WRONG, self.rect, 10)
            self.wrong_timer -= 1
        elif self.flipped:
            draw_rounded_rect(surface, CARD_FACE, self.rect, 10)
        else:
            draw_rounded_rect(surface, CARD_BG, self.rect, 10)
            # 물음표
            font = fonts[32]
            draw_text(surface, "?", (self.rect.centerx, self.rect.centery - 16),
                      font, LIGHT_GRAY, center=True)
            return

        # 텍스트
        if self.is_concept:
            font = fonts[20]
            color = ACCENT_CYAN
        else:
            font = fonts[16]
            color = WHITE

        draw_text(surface, self.text,
                  (self.rect.centerx, self.rect.centery - 20),
                  font, color, center=True, max_width=self.rect.w - 14)


class MatchGame:
    def __init__(self, tier_data, tier_num):
        self.tier = tier_num
        self.pairs = random.sample(tier_data, min(8, len(tier_data)))
        self.cards: List[MatchCard] = []
        self.selected: List[MatchCard] = []
        self.matches_found = 0
        self.total_pairs = len(self.pairs)
        self.attempts = 0
        self.start_time = time.time()
        self.finished = False
        self.wait_timer = 0
        self._create_cards()

    def _create_cards(self):
        cards_data = []
        for i, (concept, desc) in enumerate(self.pairs):
            cards_data.append((concept, i, True))
            cards_data.append((desc, i, False))
        random.shuffle(cards_data)

        # 그리드 레이아웃
        n = len(cards_data)
        cols = 4
        rows = math.ceil(n / cols)
        card_w, card_h = 200, 120
        gap = 15
        start_x = (WIDTH - cols * (card_w + gap)) // 2 + gap // 2
        start_y = 120

        for idx, (text, pair_id, is_concept) in enumerate(cards_data):
            r, c = divmod(idx, cols)
            x = start_x + c * (card_w + gap)
            y = start_y + r * (card_h + gap)
            self.cards.append(MatchCard(x, y, card_w, card_h, text, pair_id, is_concept))

    def handle_event(self, event):
        if self.finished or self.wait_timer > 0:
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for card in self.cards:
                if card.rect.collidepoint(event.pos) and not card.flipped and not card.matched:
                    card.flipped = True
                    self.selected.append(card)
                    if len(self.selected) == 2:
                        self.attempts += 1
                        if self.selected[0].pair_id == self.selected[1].pair_id:
                            self.selected[0].matched = True
                            self.selected[1].matched = True
                            self.matches_found += 1
                            self.selected = []
                            if self.matches_found == self.total_pairs:
                                self.finished = True
                        else:
                            self.selected[0].wrong_timer = 30
                            self.selected[1].wrong_timer = 30
                            self.wait_timer = 30
                    break

    def update(self):
        if self.wait_timer > 0:
            self.wait_timer -= 1
            if self.wait_timer == 0:
                for card in self.selected:
                    card.flipped = False
                self.selected = []

    def draw(self, surface, fonts):
        elapsed = time.time() - self.start_time
        # 헤더
        tier_color = TIER_COLORS.get(self.tier, ACCENT_BLUE)
        draw_text(surface, f"Memory Match — Tier {self.tier}",
                  (WIDTH // 2, 20), fonts[28], tier_color, center=True)
        draw_text(surface, f"매칭: {self.matches_found}/{self.total_pairs}   "
                           f"시도: {self.attempts}   "
                           f"시간: {elapsed:.1f}s",
                  (WIDTH // 2, 60), fonts[20], LIGHT_GRAY, center=True)

        for card in self.cards:
            card.draw(surface, fonts)

    def get_result(self):
        elapsed = time.time() - self.start_time
        accuracy = (self.total_pairs / max(1, self.attempts)) * 100
        stars = 3 if accuracy >= 80 else 2 if accuracy >= 60 else 1
        if elapsed < self.total_pairs * 5:
            stars = min(3, stars + 1)
        return {
            "mode": "Memory Match",
            "tier": self.tier,
            "time": elapsed,
            "attempts": self.attempts,
            "accuracy": accuracy,
            "stars": min(3, stars),
        }


# ═══════════════════════════════════════════════
#  Mode 2: Command Rush
# ═══════════════════════════════════════════════

class CommandGame:
    def __init__(self, tier):
        self.tier = tier
        cmds = [c for c in COMMAND_DATA if c["tier"] == tier]
        if not cmds:
            cmds = COMMAND_DATA[:8]
        self.questions = random.sample(cmds, min(8, len(cmds)))
        self.current_idx = 0
        self.input_text = ""
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.start_time = time.time()
        self.question_time = time.time()
        self.time_limit = 30 if tier <= 2 else 45
        self.finished = False
        self.feedback = ""
        self.feedback_timer = 0
        self.feedback_color = WHITE
        self.show_hint = False
        self.hint_used = 0

    def handle_event(self, event):
        if self.finished:
            return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self._check_answer()
            elif event.key == pygame.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
            elif event.key == pygame.K_TAB:
                self.show_hint = True
                self.hint_used += 1
            elif event.key == pygame.K_ESCAPE:
                pass
            else:
                if event.unicode and event.unicode.isprintable():
                    self.input_text += event.unicode

    def _check_answer(self):
        if self.current_idx >= len(self.questions):
            return

        q = self.questions[self.current_idx]
        user = self.input_text.strip().lower()
        answer = q["answer"].lower()

        # 유연한 매칭 (핵심 부분이 포함되면 OK)
        answer_parts = answer.split()
        user_parts = user.split()
        match_count = sum(1 for p in answer_parts if p in user_parts)
        similarity = match_count / len(answer_parts) if answer_parts else 0

        if user == answer or similarity >= 0.8:
            time_bonus = max(0, 10 - (time.time() - self.question_time))
            self.score += 100 + int(time_bonus * 10)
            self.combo += 1
            self.max_combo = max(self.max_combo, self.combo)
            if self.combo >= 3:
                self.score += 50  # 콤보 보너스
            self.feedback = f"정답! +{100 + int(time_bonus * 10)}점"
            self.feedback_color = ACCENT_GREEN
        elif similarity >= 0.5:
            self.score += 50
            self.combo = 0
            self.feedback = f"거의 맞음! +50  정답: {q['answer']}"
            self.feedback_color = ACCENT_ORANGE
        else:
            self.combo = 0
            self.feedback = f"오답!  정답: {q['answer']}"
            self.feedback_color = ACCENT_RED

        self.feedback_timer = 90
        self.input_text = ""
        self.show_hint = False
        self.current_idx += 1
        self.question_time = time.time()

        if self.current_idx >= len(self.questions):
            self.finished = True

    def update(self):
        if self.feedback_timer > 0:
            self.feedback_timer -= 1
        # 시간 제한 체크
        if not self.finished and time.time() - self.question_time > self.time_limit:
            self.feedback = f"시간초과!  정답: {self.questions[self.current_idx]['answer']}"
            self.feedback_color = ACCENT_RED
            self.feedback_timer = 90
            self.combo = 0
            self.input_text = ""
            self.show_hint = False
            self.current_idx += 1
            self.question_time = time.time()
            if self.current_idx >= len(self.questions):
                self.finished = True

    def draw(self, surface, fonts):
        tier_color = TIER_COLORS.get(self.tier, ACCENT_BLUE)

        draw_text(surface, f"Command Rush — Tier {self.tier}",
                  (WIDTH // 2, 20), fonts[28], tier_color, center=True)

        # 스코어 바
        draw_text(surface, f"점수: {self.score}   콤보: {self.combo}x   "
                           f"문제: {self.current_idx + 1}/{len(self.questions)}",
                  (WIDTH // 2, 60), fonts[20], LIGHT_GRAY, center=True)

        if self.finished:
            return

        if self.current_idx < len(self.questions):
            q = self.questions[self.current_idx]

            # 타이머 바
            elapsed = time.time() - self.question_time
            ratio = max(0, 1 - elapsed / self.time_limit)
            bar_w = 600
            bar_color = ACCENT_GREEN if ratio > 0.5 else ACCENT_ORANGE if ratio > 0.2 else ACCENT_RED
            draw_rounded_rect(surface, GRAY, (WIDTH//2 - bar_w//2, 100, bar_w, 12), 6)
            draw_rounded_rect(surface, bar_color,
                              (WIDTH//2 - bar_w//2, 100, int(bar_w * ratio), 12), 6)

            # 문제
            draw_rounded_rect(surface, CARD_BG, (140, 150, WIDTH - 280, 120), 15)
            draw_text(surface, f"미션: {q['prompt']}",
                      (WIDTH // 2, 180), fonts[28], ACCENT_CYAN, center=True)

            # 힌트
            if self.show_hint:
                draw_text(surface, f"힌트: {q['hint']}",
                          (WIDTH // 2, 230), fonts[20], ACCENT_ORANGE, center=True)

            # 입력 필드
            draw_rounded_rect(surface, (40, 45, 65), (140, 320, WIDTH - 280, 60), 10)
            pygame.draw.rect(surface, ACCENT_BLUE, (140, 320, WIDTH - 280, 60), 2, border_radius=10)
            cursor = "│" if int(time.time() * 2) % 2 == 0 else ""
            draw_text(surface, f"$ {self.input_text}{cursor}",
                      (160, 332), fonts[24], ACCENT_GREEN)

            # 안내
            draw_text(surface, "Enter: 제출  |  Tab: 힌트  |  ESC: 건너뛰기",
                      (WIDTH // 2, 400), fonts[16], LIGHT_GRAY, center=True)

        # 피드백
        if self.feedback_timer > 0:
            alpha = min(255, self.feedback_timer * 8)
            draw_rounded_rect(surface, CARD_BG, (140, 450, WIDTH - 280, 60), 10, alpha)
            draw_text(surface, self.feedback,
                      (WIDTH // 2, 465), fonts[22], self.feedback_color, center=True,
                      max_width=WIDTH - 320)

    def get_result(self):
        elapsed = time.time() - self.start_time
        return {
            "mode": "Command Rush",
            "tier": self.tier,
            "time": elapsed,
            "score": self.score,
            "max_combo": self.max_combo,
            "hints": self.hint_used,
            "stars": 3 if self.score >= 700 else 2 if self.score >= 400 else 1,
        }


# ═══════════════════════════════════════════════
#  Mode 3: Node Builder
# ═══════════════════════════════════════════════

class DragNode:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y
        self.w = 140
        self.h = 50
        self.dragging = False
        self.connected = False

    @property
    def rect(self):
        return pygame.Rect(self.x, self.y, self.w, self.h)

    @property
    def center(self):
        return (self.x + self.w // 2, self.y + self.h // 2)

    def draw(self, surface, fonts):
        color = ACCENT_GREEN if self.connected else ACCENT_BLUE
        if self.dragging:
            color = ACCENT_CYAN
        draw_rounded_rect(surface, color, (self.x, self.y, self.w, self.h), 8)
        draw_text(surface, self.name,
                  (self.x + self.w // 2, self.y + self.h // 2 - 10),
                  fonts[16], WHITE, center=True, max_width=self.w - 10)


class DragTopic:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y
        self.w = 120
        self.h = 36
        self.dragging = False

    @property
    def rect(self):
        return pygame.Rect(self.x, self.y, self.w, self.h)

    @property
    def center(self):
        return (self.x + self.w // 2, self.y + self.h // 2)

    def draw(self, surface, fonts):
        color = ACCENT_ORANGE if self.dragging else ACCENT_PURPLE
        draw_rounded_rect(surface, color, (self.x, self.y, self.w, self.h), 18)
        draw_text(surface, self.name,
                  (self.x + self.w // 2, self.y + self.h // 2 - 8),
                  fonts[14], WHITE, center=True, max_width=self.w - 10)


class NodeBuilderGame:
    def __init__(self, level_data):
        self.level = level_data["level"]
        self.title = level_data["title"]
        self.description = level_data["description"]
        self.correct = level_data["correct_connections"]

        self.nodes: List[DragNode] = []
        self.topics: List[DragTopic] = []
        self.connections = []  # [(node1, topic, node2)]
        self.dragging_item = None
        self.drag_offset = (0, 0)
        self.connecting = False
        self.connect_start = None  # 연결 시작 노드
        self.connect_topic = None  # 선택된 토픽
        self.finished = False
        self.start_time = time.time()
        self.score = 0
        self.feedback = ""
        self.feedback_timer = 0

        self._layout(level_data)

    def _layout(self, data):
        # 노드 배치
        n = len(data["nodes"])
        node_area_y = 200
        spacing_x = min(180, (WIDTH - 200) // max(n, 1))
        start_x = (WIDTH - spacing_x * n) // 2

        for i, name in enumerate(data["nodes"]):
            row = i // 5
            col = i % 5
            x = start_x + col * spacing_x
            y = node_area_y + row * 100
            self.nodes.append(DragNode(name, x, y))

        # 토픽 배치
        t_n = len(data["topics"])
        topic_y = node_area_y + (math.ceil(n / 5)) * 100 + 80
        t_spacing = min(160, (WIDTH - 200) // max(t_n, 1))
        t_start_x = (WIDTH - t_spacing * t_n) // 2

        for i, name in enumerate(data["topics"]):
            x = t_start_x + i * t_spacing
            self.topics.append(DragTopic(name, x, topic_y))

    def handle_event(self, event):
        if self.finished:
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos

            # 노드 클릭 → 연결 시작
            for node in self.nodes:
                if node.rect.collidepoint(pos):
                    if self.connect_start is None:
                        self.connect_start = node
                        self.connecting = True
                    elif self.connect_topic is not None and node != self.connect_start:
                        # 연결 완성
                        conn = (self.connect_start.name, self.connect_topic.name, node.name)
                        if conn not in self.connections:
                            self.connections.append(conn)
                        self._check_connections()
                        self.connect_start = None
                        self.connect_topic = None
                        self.connecting = False
                    return

            # 토픽 클릭 (연결 중일 때)
            if self.connecting and self.connect_start:
                for topic in self.topics:
                    if topic.rect.collidepoint(pos):
                        self.connect_topic = topic
                        return

            # 클릭이 빈 곳이면 연결 취소
            self.connect_start = None
            self.connect_topic = None
            self.connecting = False

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            # 우클릭: 연결 삭제
            pos = event.pos
            self.connections = [c for c in self.connections
                                if not self._connection_near(c, pos)]
            self._check_connections()
            self.connect_start = None
            self.connect_topic = None
            self.connecting = False

    def _connection_near(self, conn, pos):
        """연결선 근처 클릭인지 판단"""
        src = self._find_node(conn[0])
        dst = self._find_node(conn[2])
        if not src or not dst:
            return False
        sx, sy = src.center
        dx, dy = dst.center
        # 선분과 점의 거리
        length = math.hypot(dx - sx, dy - sy)
        if length < 1:
            return False
        dist = abs((dy-sy)*pos[0] - (dx-sx)*pos[1] + dx*sy - dy*sx) / length
        return dist < 20

    def _find_node(self, name):
        for n in self.nodes:
            if n.name == name:
                return n
        return None

    def _check_connections(self):
        correct_set = set(tuple(c) for c in self.correct)
        user_set = set(tuple(c) for c in self.connections)

        # 노드 연결 상태 업데이트
        for node in self.nodes:
            node.connected = any(c[0] == node.name or c[2] == node.name
                                 for c in self.connections if tuple(c) in correct_set)

        if correct_set == user_set:
            self.finished = True
            self.score = max(100, 500 - len(self.connections) * 20)

    def update(self):
        if self.feedback_timer > 0:
            self.feedback_timer -= 1

    def draw(self, surface, fonts):
        draw_text(surface, f"Node Builder — Level {self.level}: {self.title}",
                  (WIDTH // 2, 20), fonts[28], ACCENT_PURPLE, center=True)
        draw_text(surface, self.description,
                  (WIDTH // 2, 60), fonts[18], LIGHT_GRAY, center=True)

        # 연결 안내
        if self.connecting:
            info = "노드 선택 → "
            if self.connect_start:
                info += f"[{self.connect_start.name}] → "
            if self.connect_topic:
                info += f"({self.connect_topic.name}) → 대상 노드 클릭"
            else:
                info += "토픽 클릭"
            draw_text(surface, info,
                      (WIDTH // 2, 95), fonts[18], ACCENT_CYAN, center=True)
        else:
            draw_text(surface, "노드 클릭 → 토픽 클릭 → 대상 노드 클릭  |  우클릭: 연결 삭제",
                      (WIDTH // 2, 95), fonts[16], LIGHT_GRAY, center=True)

        # 연결선 그리기
        correct_set = set(tuple(c) for c in self.correct)
        for conn in self.connections:
            src = self._find_node(conn[0])
            dst = self._find_node(conn[2])
            if src and dst:
                color = ACCENT_GREEN if tuple(conn) in correct_set else ACCENT_RED
                pygame.draw.line(surface, color, src.center, dst.center, 3)
                # 토픽 라벨
                mx = (src.center[0] + dst.center[0]) // 2
                my = (src.center[1] + dst.center[1]) // 2
                draw_rounded_rect(surface, DARK_BG, (mx - 50, my - 12, 100, 24), 12)
                draw_text(surface, conn[1], (mx, my - 8), fonts[12], ACCENT_ORANGE, center=True)

        # 진행 중 연결선
        if self.connecting and self.connect_start:
            mouse_pos = pygame.mouse.get_pos()
            pygame.draw.line(surface, ACCENT_CYAN, self.connect_start.center, mouse_pos, 2)

        # 노드, 토픽 그리기
        for node in self.nodes:
            # 선택 중인 노드 하이라이트
            if node == self.connect_start:
                pygame.draw.rect(surface, ACCENT_CYAN,
                                 (node.x - 3, node.y - 3, node.w + 6, node.h + 6),
                                 2, border_radius=10)
            node.draw(surface, fonts)

        for topic in self.topics:
            if topic == self.connect_topic:
                pygame.draw.rect(surface, ACCENT_ORANGE,
                                 (topic.x - 3, topic.y - 3, topic.w + 6, topic.h + 6),
                                 2, border_radius=20)
            topic.draw(surface, fonts)

        # 진행 상태
        correct_count = sum(1 for c in self.connections if tuple(c) in correct_set)
        draw_text(surface, f"올바른 연결: {correct_count}/{len(self.correct)}",
                  (WIDTH // 2, HEIGHT - 60), fonts[20], ACCENT_GREEN, center=True)

    def get_result(self):
        elapsed = time.time() - self.start_time
        correct_set = set(tuple(c) for c in self.correct)
        correct_count = sum(1 for c in self.connections if tuple(c) in correct_set)
        wrong_count = len(self.connections) - correct_count
        stars = 3 if wrong_count == 0 else 2 if wrong_count <= 2 else 1
        return {
            "mode": "Node Builder",
            "level": self.level,
            "title": self.title,
            "time": elapsed,
            "correct": correct_count,
            "total": len(self.correct),
            "wrong": wrong_count,
            "stars": stars,
        }


# ═══════════════════════════════════════════════
#  메인 게임 클래스
# ═══════════════════════════════════════════════

class ROS2Commander:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("ROS2 Commander — Kevin 자율순찰 로봇 학습 게임")
        self.clock = pygame.time.Clock()

        # 폰트
        self.fonts = {}
        for size in [12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 48]:
            try:
                # 한글 지원 폰트
                for font_name in ["NanumGothic", "NanumGothicBold", "Malgun Gothic",
                                  "NotoSansCJK-Regular", "NotoSansKR-Regular",
                                  "D2Coding", "Arial"]:
                    try:
                        self.fonts[size] = pygame.font.SysFont(font_name, size)
                        break
                    except:
                        continue
                if size not in self.fonts:
                    self.fonts[size] = pygame.font.Font(None, size)
            except:
                self.fonts[size] = pygame.font.Font(None, size)

        self.state = GameState.MAIN_MENU
        self.current_game = None
        self.result = None

        # 메뉴 버튼
        self._create_menu_buttons()

        # 별 기록
        self.records = {}

    def _create_menu_buttons(self):
        cx = WIDTH // 2
        self.menu_buttons = {
            "match": Button(cx - 180, 300, 360, 70, "🃏  Memory Match", ACCENT_BLUE, font_size=28),
            "command": Button(cx - 180, 390, 360, 70, "⌨  Command Rush", ACCENT_GREEN, font_size=28),
            "node": Button(cx - 180, 480, 360, 70, "🔗  Node Builder", ACCENT_PURPLE, font_size=28),
            "quit": Button(cx - 100, 580, 200, 50, "종료", GRAY, font_size=22),
        }

        # Tier 선택 버튼
        self.tier_buttons = {}
        for i in range(1, 5):
            color = TIER_COLORS[i]
            labels = {1: "Tier 1: 기본 개념", 2: "Tier 2: 시스템 구조",
                      3: "Tier 3: Nav2/SLAM", 4: "Tier 4: 하드웨어"}
            self.tier_buttons[i] = Button(cx - 200, 200 + i * 80, 400, 60,
                                          labels[i], color, font_size=24)

        self.back_btn = Button(30, HEIGHT - 60, 120, 45, "← 뒤로", GRAY, font_size=20)

        # 노드 레벨 버튼
        self.node_level_buttons = {}
        for i, lvl in enumerate(NODE_BUILDER_LEVELS):
            self.node_level_buttons[i] = Button(
                cx - 220, 220 + i * 80, 440, 60,
                f"Level {lvl['level']}: {lvl['title']}", ACCENT_PURPLE, font_size=22
            )

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self._handle_event(event)

            self._update()
            self._draw()
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()

    def _handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self.state in (GameState.MATCH_GAME, GameState.COMMAND_GAME, GameState.NODE_GAME):
                if self.current_game and self.current_game.finished:
                    self.state = GameState.MAIN_MENU
                    self.current_game = None
                else:
                    self.state = GameState.MAIN_MENU
                    self.current_game = None
                return
            elif self.state == GameState.RESULT:
                self.state = GameState.MAIN_MENU
                return
            elif self.state in (GameState.MATCH_TIER_SELECT, GameState.COMMAND_TIER_SELECT,
                                GameState.NODE_LEVEL_SELECT, GameState.MODE_SELECT):
                self.state = GameState.MAIN_MENU
                return

        if self.state == GameState.MAIN_MENU:
            for key, btn in self.menu_buttons.items():
                if btn.clicked(event):
                    if key == "match":
                        self.state = GameState.MATCH_TIER_SELECT
                    elif key == "command":
                        self.state = GameState.COMMAND_TIER_SELECT
                    elif key == "node":
                        self.state = GameState.NODE_LEVEL_SELECT
                    elif key == "quit":
                        pygame.quit()
                        sys.exit()

        elif self.state == GameState.MATCH_TIER_SELECT:
            if self.back_btn.clicked(event):
                self.state = GameState.MAIN_MENU
                return
            tier_data_map = {1: MATCH_DATA_T1, 2: MATCH_DATA_T2,
                             3: MATCH_DATA_T3, 4: MATCH_DATA_T4}
            for tier, btn in self.tier_buttons.items():
                if btn.clicked(event):
                    self.current_game = MatchGame(tier_data_map[tier], tier)
                    self.state = GameState.MATCH_GAME

        elif self.state == GameState.COMMAND_TIER_SELECT:
            if self.back_btn.clicked(event):
                self.state = GameState.MAIN_MENU
                return
            for tier, btn in self.tier_buttons.items():
                if tier <= 3 and btn.clicked(event):
                    self.current_game = CommandGame(tier)
                    self.state = GameState.COMMAND_GAME

        elif self.state == GameState.NODE_LEVEL_SELECT:
            if self.back_btn.clicked(event):
                self.state = GameState.MAIN_MENU
                return
            for idx, btn in self.node_level_buttons.items():
                if btn.clicked(event):
                    self.current_game = NodeBuilderGame(NODE_BUILDER_LEVELS[idx])
                    self.state = GameState.NODE_GAME

        elif self.state == GameState.MATCH_GAME:
            if self.back_btn.clicked(event):
                self.state = GameState.MAIN_MENU
                self.current_game = None
                return
            if self.current_game:
                self.current_game.handle_event(event)
                if self.current_game.finished:
                    self.result = self.current_game.get_result()
                    self.state = GameState.RESULT

        elif self.state == GameState.COMMAND_GAME:
            if self.current_game:
                self.current_game.handle_event(event)
                if self.current_game.finished:
                    self.result = self.current_game.get_result()
                    self.state = GameState.RESULT

        elif self.state == GameState.NODE_GAME:
            if self.back_btn.clicked(event):
                self.state = GameState.MAIN_MENU
                self.current_game = None
                return
            if self.current_game:
                self.current_game.handle_event(event)
                if self.current_game.finished:
                    self.result = self.current_game.get_result()
                    self.state = GameState.RESULT

        elif self.state == GameState.RESULT:
            if event.type == pygame.KEYDOWN or (
                    event.type == pygame.MOUSEBUTTONDOWN and event.button == 1):
                self.state = GameState.MAIN_MENU
                self.current_game = None

    def _update(self):
        if self.current_game and hasattr(self.current_game, 'update'):
            self.current_game.update()

    def _draw(self):
        self.screen.fill(DARK_BG)

        if self.state == GameState.MAIN_MENU:
            self._draw_main_menu()
        elif self.state in (GameState.MATCH_TIER_SELECT, GameState.COMMAND_TIER_SELECT):
            self._draw_tier_select()
        elif self.state == GameState.NODE_LEVEL_SELECT:
            self._draw_node_level_select()
        elif self.state in (GameState.MATCH_GAME, GameState.COMMAND_GAME, GameState.NODE_GAME):
            if self.current_game:
                self.current_game.draw(self.screen, self.fonts)
            self.back_btn.draw(self.screen, self.fonts)
        elif self.state == GameState.RESULT:
            self._draw_result()

    def _draw_main_menu(self):
        # 타이틀
        draw_text(self.screen, "ROS2 Commander",
                  (WIDTH // 2, 80), self.fonts[48], ACCENT_CYAN, center=True)
        draw_text(self.screen, "Kevin 자율순찰 로봇 — ROS2 학습 게임",
                  (WIDTH // 2, 150), self.fonts[22], LIGHT_GRAY, center=True)

        # 구분선
        pygame.draw.line(self.screen, GRAY, (WIDTH//2 - 200, 200), (WIDTH//2 + 200, 200), 1)

        # 모드 설명
        draw_text(self.screen, "학습 모드를 선택하세요",
                  (WIDTH // 2, 240), self.fonts[20], WHITE, center=True)

        for btn in self.menu_buttons.values():
            btn.draw(self.screen, self.fonts)

        # 하단 안내
        draw_text(self.screen, "ESC: 뒤로  |  Pygame 기반 학습 게임",
                  (WIDTH // 2, HEIGHT - 30), self.fonts[14], LIGHT_GRAY, center=True)

    def _draw_tier_select(self):
        if self.state == GameState.MATCH_TIER_SELECT:
            title = "Memory Match — Tier 선택"
        else:
            title = "Command Rush — Tier 선택"

        draw_text(self.screen, title,
                  (WIDTH // 2, 80), self.fonts[32], ACCENT_CYAN, center=True)
        draw_text(self.screen, "학습할 영역을 선택하세요",
                  (WIDTH // 2, 130), self.fonts[20], LIGHT_GRAY, center=True)

        for tier, btn in self.tier_buttons.items():
            if self.state == GameState.COMMAND_TIER_SELECT and tier == 4:
                continue  # 커맨드 러시는 Tier 4 없음
            btn.draw(self.screen, self.fonts)

        self.back_btn.draw(self.screen, self.fonts)

    def _draw_node_level_select(self):
        draw_text(self.screen, "Node Builder — Level 선택",
                  (WIDTH // 2, 80), self.fonts[32], ACCENT_PURPLE, center=True)
        draw_text(self.screen, "Kevin 시스템 아키텍처를 구성하세요",
                  (WIDTH // 2, 130), self.fonts[20], LIGHT_GRAY, center=True)

        for idx, btn in self.node_level_buttons.items():
            btn.draw(self.screen, self.fonts)
            lvl = NODE_BUILDER_LEVELS[idx]
            draw_text(self.screen, lvl["description"],
                      (WIDTH // 2, btn.rect.bottom + 5), self.fonts[14],
                      LIGHT_GRAY, center=True)

        self.back_btn.draw(self.screen, self.fonts)

    def _draw_result(self):
        if not self.result:
            return

        draw_text(self.screen, "결과",
                  (WIDTH // 2, 80), self.fonts[36], ACCENT_CYAN, center=True)

        # 별
        stars = self.result.get("stars", 0)
        star_text = "★" * stars + "☆" * (3 - stars)
        star_color = ACCENT_ORANGE if stars >= 2 else ACCENT_RED
        draw_text(self.screen, star_text,
                  (WIDTH // 2, 140), self.fonts[48], star_color, center=True)

        # 모드별 결과
        y = 220
        mode = self.result.get("mode", "")

        draw_rounded_rect(self.screen, CARD_BG, (WIDTH//2 - 250, y - 10, 500, 280), 15)

        draw_text(self.screen, f"모드: {mode}",
                  (WIDTH // 2, y), self.fonts[24], WHITE, center=True)
        y += 40

        if "tier" in self.result:
            draw_text(self.screen, f"Tier: {self.result['tier']}",
                      (WIDTH // 2, y), self.fonts[20], LIGHT_GRAY, center=True)
            y += 35

        draw_text(self.screen, f"소요 시간: {self.result['time']:.1f}초",
                  (WIDTH // 2, y), self.fonts[20], LIGHT_GRAY, center=True)
        y += 35

        if "accuracy" in self.result:
            draw_text(self.screen, f"정확도: {self.result['accuracy']:.0f}%  "
                                   f"(시도 {self.result['attempts']}회)",
                      (WIDTH // 2, y), self.fonts[20], ACCENT_GREEN, center=True)
        elif "score" in self.result:
            draw_text(self.screen, f"점수: {self.result['score']}  "
                                   f"최대 콤보: {self.result['max_combo']}x",
                      (WIDTH // 2, y), self.fonts[20], ACCENT_GREEN, center=True)
        elif "correct" in self.result:
            draw_text(self.screen, f"올바른 연결: {self.result['correct']}/{self.result['total']}  "
                                   f"오답: {self.result['wrong']}",
                      (WIDTH // 2, y), self.fonts[20], ACCENT_GREEN, center=True)
        y += 50

        # 격려 메시지
        if stars == 3:
            msg = "완벽합니다! ROS2 마스터에 한 발 더!"
        elif stars == 2:
            msg = "좋은 성적이에요! 한 번 더 도전해 보세요!"
        else:
            msg = "시작이 반! 반복하면 실력이 쑥쑥 올라요!"

        draw_text(self.screen, msg,
                  (WIDTH // 2, y), self.fonts[22], ACCENT_ORANGE, center=True)

        # 안내
        draw_text(self.screen, "아무 키나 클릭하면 메뉴로 돌아갑니다",
                  (WIDTH // 2, HEIGHT - 60), self.fonts[18], LIGHT_GRAY, center=True)


# ═══════════════════════════════════════════════
#  실행
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    game = ROS2Commander()
    game.run()
