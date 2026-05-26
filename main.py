"""
main.py — SmartRoute v3 FastAPI Server
WebSocket-driven simulation backend

Chạy: python main.py
Mở:   http://localhost:8080
"""

import asyncio
import json
import math
import os
import warnings
from typing import Optional

warnings.filterwarnings('ignore', category=DeprecationWarning)


from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from graph import Graph
from aco_engine import ACOEngine
from qlearning import QLearningAgent, STATES, ACTION_PARAMS, STATE_LABELS
from environment import TSPEnvironment, DDPG_CONSTANTS
from ddpg_agent import DDPGAgent
from trainer import HeadlessTrainer
from evaluator import Evaluator

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
NUM_NODES    = 20
NUM_ANTS     = 30
MS_PER_GEN   = 0.12   # seconds between generations
CANVAS_W     = 800
CANVAS_H     = 580

app = FastAPI(title='SmartRoute v3')



# ─────────────────────────────────────────────────────────────────────────────
# Simulation State (global single-session)
# ─────────────────────────────────────────────────────────────────────────────
class SimState:
    def __init__(self):
        self.graph:      Optional[Graph]           = None
        self.aco:        Optional[ACOEngine]        = None
        self.ql_agent:   Optional[QLearningAgent]   = None
        self.ddpg_agent: Optional[DDPGAgent]        = None
        self.env:        Optional[TSPEnvironment]   = None

        self.mode                = 'ql'   # 'ql' | 'ddpg'
        self.is_running          = False
        self.generation          = 0
        self.best_path: Optional[list[int]] = None
        self.best_cost           = math.inf
        self.gen_since_last_train = 0
        self.is_processing       = False

        # Last QL step result (for UI)
        self._last_ql_result: Optional[dict] = None
        # Last DDPG step result (for UI)
        self._last_ddpg_info: Optional[dict] = None

sim = SimState()


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket Connection Manager
# ─────────────────────────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, msg: dict):
        text = json.dumps(msg, default=_json_default)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def send_to(self, ws: WebSocket, msg: dict):
        try:
            await ws.send_text(json.dumps(msg, default=_json_default))
        except Exception:
            self.disconnect(ws)


def _json_default(obj):
    if isinstance(obj, float) and math.isinf(obj):
        return None
    return str(obj)


manager = ConnectionManager()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — Serialization
# ─────────────────────────────────────────────────────────────────────────────
def _graph_data() -> dict:
    """Serialize graph state for the frontend renderer."""
    g = sim.graph
    if not g:
        return {}
    return {
        'nodes':         g.nodes,
        'pheromones':    g.pheromones,
        'blocked_edges': list(g.blocked_edges),
        'size':          g.size,
    }


def _build_gen_update() -> dict:
    msg: dict = {
        'type':       'gen_update',
        'generation': sim.generation,
        'best_cost':  sim.best_cost if math.isfinite(sim.best_cost) else None,
        'best_path':  sim.best_path,
        'graph':      _graph_data(),
        'mode':       sim.mode,
        'ql':         None,
        'ddpg':       None,
    }

    if sim.mode == 'ql' and sim.ql_agent:
        r  = sim._last_ql_result or {}
        ag = sim.ql_agent
        msg['ql'] = {
            'state':        ag.current_state,
            'state_label':  STATE_LABELS.get(ag.current_state, 'BÌNH THƯỜNG'),
            'action':       ag.current_action,
            'action_label': ACTION_PARAMS[ag.current_action]['label'],
            'alpha':        ACTION_PARAMS[ag.current_action]['alpha'],
            'rho':          ACTION_PARAMS[ag.current_action]['rho'],
            'epsilon':      ag.epsilon,
            'reward':       r.get('reward', 0),
            'q_table':      ag.q_table,
        }

    elif sim.mode == 'ddpg' and sim.ddpg_agent and sim.env:
        da  = sim.ddpg_agent
        env = sim.env
        di  = sim._last_ddpg_info or {}
        msg['ddpg'] = {
            'alpha':       env.current_alpha,
            'rho':         env.current_rho,
            'reward':      env.reward_history[-1] if env.reward_history else 0,
            'critic_loss': da.last_critic_loss,
            'actor_loss':  da.last_actor_loss,
            'buffer_size': da.buffer_size,
            'noise_level': da.noise_level,
            'state_vec':   env.get_state(),
        }

    return msg


# ─────────────────────────────────────────────────────────────────────────────
# Simulation Init
# ─────────────────────────────────────────────────────────────────────────────
def init_simulation():
    sim.graph    = Graph(NUM_NODES, CANVAS_W, CANVAS_H)
    sim.aco      = ACOEngine(sim.graph, num_ants=NUM_ANTS)
    sim.ql_agent = QLearningAgent()

    sim.graph.generate()

    sim.generation           = 0
    sim.best_path            = None
    sim.best_cost            = math.inf
    sim.gen_since_last_train  = 0
    sim.is_processing        = False
    sim._last_ql_result      = None
    sim._last_ddpg_info      = None

    if sim.mode == 'ddpg' and sim.ddpg_agent:
        sim.env = TSPEnvironment(sim.graph, sim.aco)
        sim.env.reset()
        sim.ddpg_agent.reset_noise()
    else:
        sim.env = None


# ─────────────────────────────────────────────────────────────────────────────
# One Generation
# ─────────────────────────────────────────────────────────────────────────────
async def run_generation():
    if not sim.is_running or sim.is_processing:
        return
    sim.is_processing = True

    try:
        if sim.mode == 'ql':
            _step_ql()
        else:
            _step_ddpg()

        await manager.broadcast(_build_gen_update())
    except Exception as exc:
        print(f'[runGeneration] Error: {exc}')
    finally:
        sim.is_processing = False


def _step_ql():
    """One generation of Q-Learning ACO."""
    if sim.generation == 0:
        q_res = {
            'action': 0,
            'params': ACTION_PARAMS[0],
            'state':  STATES['IMPROVING'],
            'reward': 0,
            'epsilon': sim.ql_agent.epsilon,
        }
    else:
        q_res = sim.ql_agent.step(sim.best_cost)
        sim.aco.set_params(q_res['params']['alpha'], q_res['params']['rho'])
        if q_res['action'] == 1 and q_res['state'] == STATES['DEGRADED']:
            sim.graph.reset_pheromones()

    sim._last_ql_result = q_res
    result = sim.aco.run_iteration()

    if result['best_path'] and result['best_cost'] < sim.best_cost:
        sim.best_cost = result['best_cost']
        sim.best_path = result['best_path']

    sim.generation += 1


def _step_ddpg():
    """One generation of DDPG ACO."""
    if not sim.ddpg_agent or not sim.env:
        return

    state       = sim.env.get_state()
    action_info = sim.ddpg_agent.select_action(state)
    alpha       = action_info['alpha']
    rho         = action_info['rho']
    raw_action  = action_info['raw_action']

    sim.aco.set_params(alpha, rho)
    if sim.env.stuck_counter >= 12:
        sim.graph.reset_pheromones()

    result     = sim.aco.run_iteration()
    round_cost = result['best_cost']

    if result['best_path'] and round_cost < sim.best_cost:
        sim.best_cost = round_cost
        sim.best_path = result['best_path']

    sim.generation += 1

    step_res   = sim.env.step(round_cost, alpha, rho)
    next_state = step_res['next_state']

    sim.ddpg_agent.remember(state, raw_action, step_res['reward'], next_state)

    sim.gen_since_last_train += 1
    if sim.gen_since_last_train >= 2:
        sim.gen_since_last_train = 0
        sim.ddpg_agent.train_step()


# ─────────────────────────────────────────────────────────────────────────────
# Background Simulation Loop
# ─────────────────────────────────────────────────────────────────────────────
async def simulation_loop():
    while True:
        if sim.is_running:
            await run_generation()
        await asyncio.sleep(MS_PER_GEN)


# ─────────────────────────────────────────────────────────────────────────────
# Command Handler
# ─────────────────────────────────────────────────────────────────────────────
async def handle_command(ws: WebSocket, msg: dict):
    action = msg.get('type', '')

    # ── start ────────────────────────────────────────────────────────────────
    if action == 'start':
        sim.is_running = True
        await manager.broadcast({'type': 'toast', 'message': '▶ Mô phỏng đang chạy...'})

    # ── pause ────────────────────────────────────────────────────────────────
    elif action == 'pause':
        sim.is_running = False
        await manager.broadcast({'type': 'toast', 'message': '⏸ Đã tạm dừng'})

    # ── reset ────────────────────────────────────────────────────────────────
    elif action == 'reset':
        sim.is_running = False
        init_simulation()
        await manager.broadcast({
            'type': 'init',
            'graph': _graph_data(),
            'mode': sim.mode,
        })
        await manager.broadcast({'type': 'toast', 'message': '🔄 Đã khởi tạo lại'})

    # ── switch_mode ───────────────────────────────────────────────────────────
    elif action == 'switch_mode':
        new_mode       = msg.get('mode', 'ql')
        sim.is_running = False
        sim.mode       = new_mode

        if new_mode == 'ddpg':
            if not sim.ddpg_agent:
                await manager.broadcast({'type': 'toast',
                                         'message': '⏳ Đang khởi tạo DDPG...'})
                sim.ddpg_agent = DDPGAgent()
            sim.env = TSPEnvironment(sim.graph, sim.aco)
            sim.env.reset()
            sim.ddpg_agent.reset_noise()
            sim.gen_since_last_train = 0
        else:
            sim.ql_agent = QLearningAgent()
            sim.env      = None

        sim.generation = 0
        sim.best_path  = None
        sim.best_cost  = math.inf

        await manager.broadcast({'type': 'mode_switch', 'mode': new_mode})
        label = 'ACO + DDPG' if new_mode == 'ddpg' else 'ACO + Q-Learning'
        await manager.broadcast({'type': 'toast',
                                 'message': f'✅ Chuyển sang {label} Mode'})
        await manager.broadcast(_build_gen_update())

    # ── block_edge ────────────────────────────────────────────────────────────
    elif action == 'block_edge':
        i, j = msg.get('i'), msg.get('j')
        if sim.graph and i is not None and j is not None:
            blocked = sim.graph.block_edge(i, j)
            if blocked:
                if sim.mode == 'ql' and sim.ql_agent:
                    sim.ql_agent.trigger_traffic_event()
                elif sim.mode == 'ddpg' and sim.env:
                    sim.env.trigger_traffic_event()
                sim.best_cost = math.inf
                await manager.broadcast({
                    'type': 'toast',
                    'message': f'⚠️ Tắc đường: cạnh {i}↔{j} (×{sim.graph.TRAFFIC_MULTIPLIER})',
                })
                await manager.broadcast(_build_gen_update())
            else:
                await manager.send_to(ws, {'type': 'toast',
                                           'message': 'Cạnh này đã bị tắc đường rồi!'})

    # ── clear_traffic ─────────────────────────────────────────────────────────
    elif action == 'clear_traffic':
        if sim.graph:
            sim.graph.clear_all_blocked_edges()
            if sim.mode == 'ql' and sim.ql_agent:
                sim.ql_agent.trigger_traffic_event()
            elif sim.mode == 'ddpg' and sim.env:
                sim.env.trigger_traffic_event()
            sim.best_cost = math.inf
            await manager.broadcast({'type': 'toast',
                                     'message': 'Đã gỡ bỏ toàn bộ tắc đường!'})
            await manager.broadcast(_build_gen_update())

    # ── train_fast ────────────────────────────────────────────────────────────
    elif action == 'train_fast':
        if sim.mode != 'ddpg':
            await manager.send_to(ws, {'type': 'toast',
                                       'message': 'Chỉ có thể Train ở chế độ DDPG!'})
            return

        sim.is_running = False
        if not sim.ddpg_agent:
            await manager.broadcast({'type': 'toast',
                                     'message': '⏳ Đang khởi tạo DDPG...'})
            sim.ddpg_agent = DDPGAgent()

        await manager.broadcast({'type': 'train_start'})

        trainer = HeadlessTrainer(sim.graph, sim.ddpg_agent)

        async def progress(gen, best, a_loss, noise):
            await manager.broadcast({
                'type':      'train_progress',
                'gen':       gen,
                'best_cost': best,
            })

        await trainer.train(2000, progress)
        await manager.broadcast({'type': 'train_done'})
        await manager.broadcast({'type': 'toast', 'message': '✅ Train hoàn tất!'})

    # ── run_test_suite ────────────────────────────────────────────────────────
    elif action == 'run_test_suite':
        if not sim.ddpg_agent:
            await manager.send_to(ws, {'type': 'toast',
                                       'message': 'Cần khởi tạo DDPG agent trước!'})
            return

        sim.is_running = False
        await manager.broadcast({'type': 'toast',
                                 'message': '⏳ Đang chạy Test Suite...'})

        evaluator = Evaluator()

        async def progress(msg_str, pct):
            await manager.broadcast({
                'type':    'test_progress',
                'message': msg_str,
                'pct':     pct,
            })

        results = await evaluator.run_test_suite(sim.ddpg_agent, progress)
        await manager.broadcast({'type': 'test_results', 'results': results})

    # ── export_model ──────────────────────────────────────────────────────────
    elif action == 'export_model':
        if sim.ddpg_agent:
            data = sim.ddpg_agent.save_model()
            await manager.send_to(ws, {
                'type':     'model_export',
                'data':     data,
                'filename': 'smartroute_ddpg.pt',
            })
            await manager.broadcast({'type': 'toast', 'message': '✅ Đã xuất mô hình!'})
        else:
            await manager.send_to(ws, {'type': 'toast',
                                       'message': 'Chưa có mô hình DDPG!'})

    # ── import_model ──────────────────────────────────────────────────────────
    elif action == 'import_model':
        data = msg.get('data')
        if data:
            try:
                if not sim.ddpg_agent:
                    sim.ddpg_agent = DDPGAgent()
                sim.ddpg_agent.load_model(data)
                await manager.broadcast({'type': 'toast',
                                         'message': '✅ Tải mô hình thành công!'})
            except Exception as exc:
                await manager.send_to(ws, {
                    'type': 'toast',
                    'message': f'❌ Lỗi tải mô hình: {exc}',
                })


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket Endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket('/ws')
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)

    # Send initial state
    await manager.send_to(ws, {
        'type':  'init',
        'graph': _graph_data(),
        'mode':  sim.mode,
    })

    try:
        while True:
            data = await ws.receive_text()
            msg  = json.loads(data)
            await handle_command(ws, msg)
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as exc:
        print(f'[WS Error] {exc}')
        manager.disconnect(ws)


# ─────────────────────────────────────────────────────────────────────────────
# Static Files
# ─────────────────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get('/')
async def index():
    return FileResponse(os.path.join(_BASE_DIR, 'index.html'))


@app.get('/favicon.ico')
async def favicon():
    return FileResponse(os.path.join(_BASE_DIR, 'favicon.ico'),
                        status_code=204) if os.path.exists(
        os.path.join(_BASE_DIR, 'favicon.ico')) else FileResponse(
        os.path.join(_BASE_DIR, 'index.html'), status_code=204)


# Mount static directories
_css_dir = os.path.join(_BASE_DIR, 'css')
_js_dir  = os.path.join(_BASE_DIR, 'js')
if os.path.isdir(_css_dir):
    app.mount('/css', StaticFiles(directory=_css_dir), name='css')
if os.path.isdir(_js_dir):
    app.mount('/js', StaticFiles(directory=_js_dir), name='js')




# ─────────────────────────────────────────────────────────────────────────────
# Startup event
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event('startup')
async def startup_event():
    init_simulation()
    asyncio.create_task(simulation_loop())
    print('=== SmartRoute v3 (Python/FastAPI) ===')
    print('    http://localhost:8080')
    print('    Press Ctrl+C to stop')
    print('=======================================')

# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8080, reload=False)