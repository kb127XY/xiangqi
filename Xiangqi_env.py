import random
from typing import Any, Dict, List, Tuple

class XiangqiEnv:
    def __init__(self):
        self.state = None
        self.current_player = None  # 'red' or 'black'
        self.step_limit = 300   # 最大步数上限（可根据需要改）
        self.step_count = 0
    def board(self)  -> List[list[str]]:
        return [
            list("rnbakabnr"),  # 黑方底线
            list("........."),
            list(".c.....c."),
            list("p.p.p.p.p"),
            list("........."),
            list("........."),
            list("P.P.P.P.P"),
            list(".C.....C."),
            list("........."),
            list("RNBAKABNR"),  # 红方底线
             #车马相士帅
        ]
    def reset(self) -> str:
        # 重置到初始局面，返回可哈希的状态（建议用 FEN 字符串或压缩后的特征）
        self.current_player = 'red'
        self.state = self.encode_fen()  
        self.step_count = 0
        return self.state
    def encode_fen(self)-> str:
        parts=[]
        for row in self.board():
            cnt=0
            s=""
            for x in row :
                if x =='.':
                    cnt+=1
                else:
                    if cnt>0:
                        s+=str(cnt)
                        cnt=0
                    s+=x
            if cnt>0:
                s+=str(cnt)
            parts.append(s)
        side = 'w' if self.current_player=='red' else 'b'
        return '/'.join(parts) + f' {side} - - 0 1'
    
    def _from_uci(self, s: str):
        c0 = ord(s[0]) - ord('a'); r0 = int(s[1])
        c1 = ord(s[2]) - ord('a'); r1 = int(s[3])
        return r0, c0, r1, c1

    def _in_bounds(self, r, c): return 0 <= r < 10 and 0 <= c < 9
    def _is_red(self, pc): return pc.isupper()
    def _is_black(self, pc): return pc.islower()
    def _is_empty(self, pc): return pc == '.'
    def _same_side(self, a, b):
        if self._is_empty(a) or self._is_empty(b): return False
        return (self._is_red(a) and self._is_red(b)) or (self._is_black(a) and self._is_black(b))

    def _clear_path(self, r0, c0, r1, c1):
        if r0 == r1:
            step = 1 if c1 > c0 else -1
            for c in range(c0+step, c1, step):
                if self.board()[r0][c] != '.': return False
            return True
        if c0 == c1:
            step = 1 if r1 > r0 else -1
            for r in range(r0+step, r1, step):
                if self.board()[r][c0] != '.': return False
            return True
        return False

    def _knight_leg_clear(self, r0, c0, r1, c1):
        dr, dc = r1-r0, c1-c0
        if abs(dr) == 2 and abs(dc) == 1:
            leg = (r0 + (1 if dr>0 else -1), c0)
        elif abs(dr) == 1 and abs(dc) == 2:
            leg = (r0, c0 + (1 if dc>0 else -1))
        else:
            return False
        return self.board()[leg[0]][leg[1]] == '.'

    def _elephant_eye_clear(self, r0, c0, r1, c1):
        if abs(r1-r0) != 2 or abs(c1-c0) != 2: return False
        eye = (r0 + (1 if r1>r0 else -1), c0 + (1 if c1>c0 else -1))
        return self.board()[eye[0]][eye[1]] == '.'
    def _legal_for_piece(self, r0, c0, r1, c1) -> bool:
        if not self._in_bounds(r1, c1): return False
        me = self.board()[r0][c0]
        you = self.board()[r1][c1]
        if self._is_empty(me): return False
        if self._same_side(me, you): return False  # 不能吃自己人

        p = me.upper()
        dr, dc = r1 - r0, c1 - c0

        if p == 'R':  # 车
            return (r0 == r1 or c0 == c1) and self._clear_path(r0, c0, r1, c1)

        if p == 'N':  # 马
            return (abs(dr), abs(dc)) in [(2,1),(1,2)] and self._knight_leg_clear(r0, c0, r1, c1)

        if p == 'C':  # 炮
            return (r0 == r1 or c0 == c1) and self._clear_path(r0, c0, r1, c1)

        if p == 'K':  # 将
            return (abs(dr), abs(dc)) in [(1,0),(0,1)]

        if p == 'A':  # 士
            return abs(dr) == 1 and abs(dc) == 1

        if p == 'B':  # 相
            return self._elephant_eye_clear(r0, c0, r1, c1)

        if p == 'P':  # 兵
            forward = 1 if self._is_red(me) else -1
            return dr == forward and dc == 0

        return False
    def legal_actions(self, state=None):
        acts = []
        red_turn = (self.current_player == 'red')
        for r in range(10):
            for c in range(9):
                pc = self.board()[r][c]
                if pc == '.': continue
                if red_turn and not self._is_red(pc): continue
                if (not red_turn) and not self._is_black(pc): continue
            # 穷举可达格（粗暴但好用）：整盘所有格尝试一次
                for rr in range(10):
                    for cc in range(9):
                        if self._legal_for_piece(r, c, rr, cc):
                            acts.append(f"{chr(ord('a')+c)}{r}{chr(ord('a')+cc)}{rr}")
        return acts
    def _apply_move_inplace(self, r0, c0, r1, c1):
        self.board()[r1][c1] = self.board()[r0][c0]  # 目的地放我
        self.board()[r0][c0] = '.'

    def step(self, action: str):
        r0, c0, r1, c1 = self._from_uci(action)

    # 兜底校验：是否在可行集合里
        legal_list = self.legal_actions()
        if action not in legal_list:
        # 非法步：直接判这方负（也可选择不给奖励只忽略）
            rew = -1.0 if self.current_player == 'red' else 1.0
            return self.encode_fen(), rew, True, {"result": "illegal_move"}

    # 执行
        captured = self.board()[r1][c1]
        self._apply_move_inplace(r0, c0, r1, c1)

    # 判终：将/帅被吃（简化）
        red_king_alive = any('K' in row for row in self.board())
        black_king_alive = any('k' in row for row in self.board())
        done = False
        reward = 0.0
        info = {}

        if not red_king_alive or not black_king_alive:
            done = True
            if red_king_alive and not black_king_alive:  # 红胜
                return self.encode_fen(), 1.0, True, {"result": "win"}
            elif black_king_alive and not red_king_alive:  # 黑胜
                return self.encode_fen(), -1.0, True, {"result": "loss"}
            else:
                return self.encode_fen(), 0.0, True, {"result": "draw"}

    # 轮转走子方 & 返回状态
        if not done:
            self.current_player = 'black' if self.current_player == 'red' else 'red'

        next_legal = self.legal_actions()
        if not next_legal:
        # 上一步行动方是胜者：如果现在轮到黑走，说明上一手是红；反之亦然
            last_was_red = (self.current_player == 'black')
            if last_was_red:
                return self.encode_fen(), 1.0, True, {"result": "win"}
            else:
                return self.encode_fen(), -1.0, True, {"result": "loss"}
    
        self.step_count += 1
        if self.step_count >= self.step_limit:  # 在 __init__ 里设：self.step_limit = 300; self.step_count = 0
            return self.encode_fen(), 0.0, True, {"result": "draw"}
        return self.encode_fen(), 0.0,False,{}

    def is_red_to_move(self) -> bool:
        return self.current_player == 'red'
def opponent_policy(env: XiangqiEnv, state, Q, epsilon=0.2):
        actions = env.legal_actions(state)
        if not actions:
            return None
        if random.random() < epsilon:
            return random.choice(actions)
    # 选择 Q 值最小的一方可理解为对手在与我对抗；也可以共享 “最大” 模式（更稳定）
    # 这里采用“最大”共享以简化（自博弈）
        best = max(actions, key=lambda a: Q.get((state, a), 0.0))
        return best
'''
if __name__ == "__main__":
    env = XiangqiEnv()
    state = env.reset()
    print("初始局面 FEN：")
    print(state)
    # 1) reset 一致性
    s1 = env.reset()
    s2 = env.reset()
    assert s1 == s2, "reset() 每次都应回到相同初始局面"
    print("✓ reset 一致性通过")

    # 2) 初始动作用例非空
    acts = env.legal_actions(s1)
    assert isinstance(acts, list) and len(acts) > 0, "开局动作应非空"
    print("✓ 开局 legal_actions 非空通过")

    # 3) 执行一个动作后：FEN 走子方切换 / 状态变化
    a = acts[0]
    s3, r, done, info = env.step(a)
    assert " w" in s1 or " b" in s1
    assert " w" in s3 or " b" in s3
    assert s3 != s1, "一步之后状态应变化"
    print("✓ step 状态变化通过")

    # 4) 轮换走子方
    side1 = "w" if env.current_player == "red" else "b"  # 当前侧与 FEN 一致校验
    assert s3.endswith(" b") or s3.endswith(" w")
    print("✓ 走子方切换通过")

    # 5) 非法步兜底（构造一个超界步/无来源子）
    bad = "a9a9"  # 同格或不在 legal 列表
    s4, r2, done2, info2 = env.step(bad)
    assert done2 is True and r2 in (-1.0, 1.0), "非法步应被终止并惩罚（或按你的策略）"
    print("✓ 非法步兜底通过")

    print("\n🚀 Phase-2 冒烟测试通过")
'''