from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

def set_cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), hex_color)
    shd.set(qn('w:val'), 'clear')
    tcPr.append(shd)

def add_heading(doc, text, level=1, color=None):
    p = doc.add_heading(text, level=level)
    if color:
        for run in p.runs:
            run.font.color.rgb = RGBColor(*bytes.fromhex(color))
    return p

def add_table(doc, headers, rows, header_color='1F3864'):
    table = doc.add_table(rows=1+len(rows), cols=len(headers))
    table.style = 'Table Grid'
    # Header row
    hdr = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = h
        set_cell_bg(cell, header_color)
        for run in cell.paragraphs[0].runs:
            run.font.bold = True
            run.font.color.rgb = RGBColor(255,255,255)
            run.font.size = Pt(10)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    # Data rows
    for r_idx, row_data in enumerate(rows):
        row = table.rows[r_idx+1]
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.text = val
            cell.paragraphs[0].runs[0].font.size = Pt(9.5)
            if r_idx % 2 == 0:
                set_cell_bg(cell, 'EBF3FB')
    return table

doc = Document()

# === PAGE MARGINS ===
for section in doc.sections:
    section.top_margin    = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin   = Cm(2.5)
    section.right_margin  = Cm(2.5)

# ============================================================
# TITLE PAGE
# ============================================================
doc.add_paragraph()
doc.add_paragraph()
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run('SmartRoute v3')
run.font.size = Pt(32)
run.font.bold = True
run.font.color.rgb = RGBColor(0x1A, 0x73, 0xE8)

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
run2 = sub.add_run('Hệ thống Điều phối Logistics Thích ứng\nDynamic TSP với ACO + Q-Learning + DDPG (Deep RL)')
run2.font.size = Pt(14)
run2.font.color.rgb = RGBColor(80, 80, 80)

doc.add_paragraph()
date_p = doc.add_paragraph()
date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run3 = date_p.add_run(f'Ngày tạo: {datetime.date.today().strftime("%d/%m/%Y")}')
run3.font.size = Pt(11)
run3.font.color.rgb = RGBColor(120,120,120)

doc.add_page_break()

# ============================================================
# 1. TỔNG QUAN
# ============================================================
add_heading(doc, '1. Tổng quan dự án', 1, '1A73E8')
doc.add_paragraph(
    'SmartRoute v3 là một ứng dụng web mô phỏng bài toán Người Bán Hàng Động (Dynamic TSP) '
    'trong lĩnh vực điều phối logistics. Hệ thống sử dụng thuật toán tối ưu đàn kiến (ACO) '
    'làm nền tảng, kết hợp với hai bộ não AI có thể chuyển đổi qua lại trong thời gian thực:\n'
    '• Q-Learning (Tabular) — 3 trạng thái × 3 hành động rời rạc\n'
    '• DDPG (Deep Deterministic Policy Gradient) — điều chỉnh liên tục tham số Alpha/Rho'
)

doc.add_paragraph(
    'Điểm đặc biệt của dự án là hỗ trợ "Tắc đường động" — người dùng có thể chặn bất kỳ '
    'cạnh nào trên bản đồ trong khi thuật toán đang chạy, và AI sẽ tự động phản ứng bằng cách '
    'tìm tuyến đường thay thế tối ưu.'
)

add_heading(doc, '1.1. Công nghệ sử dụng', 2)
add_table(doc,
    ['Hạng mục', 'Công nghệ', 'Phiên bản', 'Mục đích'],
    [
        ['Frontend',    'HTML5 / Vanilla CSS / ES Modules', '—',    'Giao diện web'],
        ['AI / ML',     'TensorFlow.js',                   '4.15', 'Mạng Neural DDPG (Actor-Critic)'],
        ['Build Tool',  'Vite',                            '5.2',  'Dev server, HMR, bundler'],
        ['Runtime',     'Node.js',                         '≥18',  'Chạy Vite và npm'],
        ['Rendering',   'HTML5 Canvas API',                '—',    'Vẽ bản đồ, pheromone, particles'],
        ['Charts',      'Chart.js',                        '4.x',  'Biểu đồ hội tụ real-time'],
        ['Ngôn ngữ',    'JavaScript (ES2022)',              '—',    'Toàn bộ logic frontend'],
    ]
)

# ============================================================
# 2. KIẾN TRÚC
# ============================================================
doc.add_paragraph()
add_heading(doc, '2. Kiến trúc hệ thống', 1, '1A73E8')

add_heading(doc, '2.1. Cấu trúc thư mục', 2)
code_block = doc.add_paragraph()
code_block.style = 'No Spacing'
run_code = code_block.add_run(
    'Ant_Colony_QL_DQN/\n'
    '├── index.html          # Entry point — giao diện web chính\n'
    '├── package.json        # Cấu hình npm (TF.js + Vite)\n'
    '├── vite.config.js      # Cấu hình Vite dev server\n'
    '├── css/\n'
    '│   └── style.css       # Design system (glassmorphism, dark theme)\n'
    '├── js/\n'
    '│   ├── graph.js        # [DATA]  Đồ thị: nodes, distances, pheromone matrix\n'
    '│   ├── aco.js          # [ALGO]  ACO Engine: đàn kiến, elitist strategy\n'
    '│   ├── qlearning.js    # [AI-1]  Q-Learning Agent (tabular, 3×3)\n'
    '│   ├── environment.js  # [AI-2]  Môi trường DDPG: state vector 10D, reward\n'
    '│   ├── ddpg_agent.js   # [AI-2]  DDPG: Actor-Critic, Replay Buffer, OU Noise\n'
    '│   ├── trainer.js      # [AI-2]  Headless Trainer: train nhanh không render\n'
    '│   ├── test_suite.js   # [EVAL]  5 Test Cases + Brute-force solver\n'
    '│   ├── evaluator.js    # [EVAL]  Chạy test tự động, so sánh QL vs DDPG\n'
    '│   ├── renderer.js     # [VIEW]  Canvas renderer\n'
    '│   ├── chart_manager.js# [VIEW]  Biểu đồ hội tụ real-time\n'
    '│   └── app.js          # [CTRL]  Orchestrator: vòng lặp, events, UI\n'
    '└── server.py           # (Cũ) Python server — không dùng ở v3'
)
run_code.font.name = 'Courier New'
run_code.font.size = Pt(9)

add_heading(doc, '2.2. Sơ đồ luồng dữ liệu mỗi thế hệ (Generation)', 2)
doc.add_paragraph(
    'Mỗi thế hệ (generation) trong cả 2 chế độ thực hiện theo trình tự:\n\n'
    '  [1] Agent chọn hành động → cập nhật Alpha (α) và Rho (ρ)\n'
    '  [2] ACO Engine chạy: 30 kiến xây dựng lộ trình dựa trên pheromone + heuristic\n'
    '  [3] Pheromone bay hơi (×(1-ρ)) và cập nhật theo lộ trình tốt nhất\n'
    '  [4] Môi trường tính Reward → Agent học / cập nhật Q-Table hoặc Replay Buffer\n'
    '  [5] Renderer vẽ lại Canvas và cập nhật biểu đồ'
)

# ============================================================
# 3. THUẬT TOÁN ACO
# ============================================================
doc.add_paragraph()
add_heading(doc, '3. Thuật toán ACO (Ant Colony Optimization)', 1, '1A73E8')
doc.add_paragraph(
    'ACO Engine là nền tảng tìm đường của toàn bộ hệ thống. Cả Q-Learning và DDPG đều '
    'điều khiển ACO thông qua 2 tham số chính: Alpha (α) và Rho (ρ).'
)

add_heading(doc, '3.1. Tham số ACO', 2)
add_table(doc,
    ['Tham số', 'Ký hiệu', 'Ý nghĩa', 'Dải giá trị'],
    [
        ['Alpha',    'α', 'Trọng số pheromone — càng cao kiến càng bám theo mùi cũ',    '0.5 – 2.5'],
        ['Beta',     'β', 'Trọng số heuristic (khoảng cách) — cố định',                  '2.0 (cố định)'],
        ['Rho',      'ρ', 'Tốc độ bay hơi — càng cao pheromone phai càng nhanh',         '0.02 – 0.50'],
        ['Kiến/Gen', 'n', 'Số lượng kiến mỗi thế hệ',                                   '30 (cố định)'],
    ]
)

add_heading(doc, '3.2. Công thức chọn đường đi', 2)
doc.add_paragraph(
    'Xác suất kiến chọn cạnh (i → j):\n\n'
    '    P(i→j) ∝ τ(i,j)^α × η(i,j)^β\n\n'
    'Trong đó τ(i,j) là mức pheromone, η(i,j) = 1/cost(i,j) là heuristic khoảng cách.\n\n'
    'Cập nhật pheromone sau mỗi iteration:\n\n'
    '    τ(i,j) = τ(i,j) × (1 - ρ) + Δτ\n\n'
    'Elitist strategy: top 30% kiến tốt nhất + kiến có lộ trình ngắn nhất toàn thời gian '
    'được phép deposit pheromone, với trọng số cao hơn kiến thường.'
)

# ============================================================
# 4. Q-LEARNING
# ============================================================
doc.add_paragraph()
add_heading(doc, '4. Q-Learning Agent', 1, '1A73E8')
doc.add_paragraph(
    'Bộ não thế hệ đầu — Tabular Q-Learning với không gian trạng thái và hành động rời rạc. '
    'Đơn giản, hội tụ nhanh, ổn định trên bản đồ nhỏ.'
)

add_heading(doc, '4.1. Không gian trạng thái (State Space)', 2)
add_table(doc,
    ['State', 'Tên', 'Điều kiện kích hoạt'],
    [
        ['S0', 'IMPROVING', 'Best distance đang giảm đều qua các gen'],
        ['S1', 'STUCK',     '≥10 gen liên tiếp không có kỷ lục mới'],
        ['S2', 'DEGRADED',  'Tắc đường xảy ra hoặc distance tăng đột ngột'],
    ]
)

add_heading(doc, '4.2. Không gian hành động (Action Space)', 2)
add_table(doc,
    ['Action', 'Tên', 'Alpha (α)', 'Rho (ρ)', 'Chiến lược'],
    [
        ['A0', 'Duy trì (MAINTAIN)', '1.0', '0.10', 'Bình thường, giữ ổn định'],
        ['A1', 'Khám phá (EXPLORE)', '0.5', '0.80', 'Xóa mùi cũ, mở rộng tìm kiếm'],
        ['A2', 'Khai thác (EXPLOIT)', '2.0', '0.05', 'Bám chặt lộ trình tốt nhất'],
    ]
)

add_heading(doc, '4.3. Hàm phần thưởng (Reward)', 2)
add_table(doc,
    ['Reward', 'Điều kiện'],
    [
        ['+10', 'Đạt kỷ lục mới (best distance giảm)'],
        ['+1',  'Duy trì ổn định, không bị kẹt'],
        ['-5',  'Vẫn bị kẹt sau khi đã STUCK'],
        ['-10', 'Distance tệ hơn đáng kể'],
    ]
)

# ============================================================
# 5. DDPG
# ============================================================
doc.add_paragraph()
add_heading(doc, '5. DDPG Agent (Deep Deterministic Policy Gradient)', 1, '1A73E8')
doc.add_paragraph(
    'Bộ não thế hệ mới — sử dụng mạng Neural Network để điều chỉnh liên tục (continuous) '
    'hai tham số ACO, thay vì nhảy giữa 3 mức cố định như Q-Learning. '
    'Cho phép tinh chỉnh mịn hơn và phản ứng linh hoạt hơn với tắc đường động.'
)

add_heading(doc, '5.1. Kiến trúc mạng', 2)
add_table(doc,
    ['Mạng', 'Kiến trúc', 'Input', 'Output', 'Số tham số'],
    [
        ['Actor (Online)',  'Dense(256)→Dense(256)→Dense(128)→tanh', 'State 10D', 'Action 2D (α, ρ)', '~101,762'],
        ['Actor (Target)',  'Giống Actor Online',                    'State 10D', 'Action 2D',         '~101,762'],
        ['Critic (Online)', 'Dense(256)→Dense(256)→Dense(128)→1',   '[State∥Action] 12D', 'Q-value 1D', '~102,145'],
        ['Critic (Target)', 'Giống Critic Online',                   '[State∥Action] 12D', 'Q-value 1D', '~102,145'],
    ]
)

add_heading(doc, '5.2. State Vector 10 chiều', 2)
add_table(doc,
    ['Chiều', 'Tên feature', 'Mô tả', 'Dải'],
    [
        ['[0]', 'normalized_best_cost',   'best_cost / reference_cost',                '[0, 3]'],
        ['[1]', 'improvement_rate',       '(prev_cost - curr_cost) / prev_cost',        '[-1, 1]'],
        ['[2]', 'stuck_counter_norm',     'Số gen kẹt / STUCK_THRESHOLD',               '[0, 1]'],
        ['[3]', 'avg_pheromone',          'Trung bình pheromone / PHEROMONE_MAX',        '[0, 1]'],
        ['[4]', 'std_pheromone',          'Độ lệch chuẩn pheromone / PHEROMONE_MAX',    '[0, 1]'],
        ['[5]', 'blocked_edge_ratio',     'Số cạnh tắc / tổng số cạnh',                 '[0, 1]'],
        ['[6]', 'current_alpha_norm',     '(alpha - 0.5) / 2.0',                        '[0, 1]'],
        ['[7]', 'current_rho_norm',       '(rho - 0.02) / 0.48',                        '[0, 1]'],
        ['[8]', 'generation_progress',    '(generation % 100) / 100',                   '[0, 1]'],
        ['[9]', 'pheromone_entropy',      'Entropy chuẩn hoá của phân phối pheromone', '[0, 1]'],
    ]
)

add_heading(doc, '5.3. Action Space', 2)
doc.add_paragraph(
    'Actor xuất ra vector 2D ∈ [-1, 1]² (qua hàm tanh), sau đó được scale tuyến tính:\n\n'
    '    alpha = 0.5 + (raw[0] + 1) / 2 × 2.0   →   alpha ∈ [0.5, 2.5]\n'
    '    rho   = 0.02 + (raw[1] + 1) / 2 × 0.48  →   rho   ∈ [0.02, 0.50]'
)

add_heading(doc, '5.4. Hàm phần thưởng DDPG (3 thành phần)', 2)
doc.add_paragraph(
    '1. Kết quả ACO:\n'
    '   • +20 nếu cải thiện ≥ 5%\n'
    '   • +10 nếu cải thiện > 0%\n'
    '   • -5  nếu kẹt ≥ STUCK_THRESHOLD gen\n'
    '   • +2  nếu đang giữ phong độ tốt (≤ 2% của allTimeBest)\n'
    '   • -1  nếu lang thang không hội tụ\n\n'
    '2. Sức khoẻ tham số (Bell-curve):\n'
    '   • Alpha lý tưởng ≈ 1.0: thưởng khi ở giữa dải, phạt khi quá thấp hoặc quá cao\n'
    '   • Rho lý tưởng ≈ 0.10: thưởng khi ở mức trung bình, phạt khi bay hơi quá nhanh hoặc quá chậm\n'
    '   • Điểm thêm từ [-2, +2]\n\n'
    '3. Entropy pheromone:\n'
    '   • -3 nếu entropy < 0.15 (kiến bị "lock-in" 1 lối duy nhất)\n'
    '   • +1 nếu entropy trong [0.25, 0.75] (vùng khám phá lành mạnh)'
)

add_heading(doc, '5.5. Hyperparameters', 2)
add_table(doc,
    ['Tham số', 'Giá trị', 'Ý nghĩa'],
    [
        ['Batch size',        '64',     'Số mẫu lấy từ Replay Buffer mỗi lần train'],
        ['Replay Buffer size','5,000',  'Số transitions tối đa lưu trong bộ nhớ'],
        ['Min buffer',        '200',    'Số mẫu tối thiểu trước khi bắt đầu train'],
        ['Actor LR',          '1e-4',   'Tốc độ học của Actor (Adam optimizer)'],
        ['Critic LR',         '1e-3',   'Tốc độ học của Critic (Adam optimizer)'],
        ['Gamma (γ)',          '0.95',   'Discount factor (tầm nhìn dài hạn)'],
        ['Tau (τ)',            '0.005',  'Hệ số Polyak soft-update target networks'],
        ['OU Sigma init',     '0.3',    'Nhiễu khám phá ban đầu'],
        ['OU Sigma min',      '0.05',   'Nhiễu tối thiểu (sau khi đã train đủ)'],
        ['OU Sigma decay',    '0.9995', 'Tốc độ giảm nhiễu mỗi bước train'],
    ]
)

# ============================================================
# 6. HEADLESS TRAINER & EVALUATOR
# ============================================================
doc.add_paragraph()
add_heading(doc, '6. Headless Trainer & Test Suite Evaluator', 1, '1A73E8')

add_heading(doc, '6.1. Headless Trainer', 2)
doc.add_paragraph(
    '"Headless" nghĩa là chạy không có giao diện đồ họa (không render Canvas). '
    'Khi bấm ⚡ Train DDPG (Fast), hệ thống bỏ qua toàn bộ bước render '
    'và chạy thuần túy vòng lặp ACO + DDPG học trong JavaScript. '
    '2,000 thế hệ hoàn thành trong 2–5 giây thay vì ~4 phút nếu render đầy đủ.'
)

add_heading(doc, '6.2. Test Suite (5 bài kiểm tra chuẩn)', 2)
add_table(doc,
    ['Test Case', 'Nodes', 'Loại', 'Mô tả'],
    [
        ['TC1', 'N=8',  'Bình thường',  'Bản đồ ngẫu nhiên không tắc đường'],
        ['TC2', 'N=10', 'Tắc cứng ×1', '1 cạnh bị chặn từ đầu (chi phí ×100)'],
        ['TC3', 'N=10', 'Tắc cứng ×2', '2 cạnh bị chặn từ đầu'],
        ['TC4', 'N=10', 'Tắc động ×1', '1 cạnh xuất hiện tắc đường ở gen 50'],
        ['TC5', 'N=10', 'Tắc động ×2', '2 cạnh: cạnh 1 tắc ở gen 50, cạnh 2 tắc ở gen 100'],
    ]
)
doc.add_paragraph(
    'Với mỗi test case, Brute-force (duyệt toàn bộ N! hoán vị) tính ra khoảng cách Optimal tuyệt đối. '
    'Cả QL-ACO và DDPG đều chạy 500 gen trên cùng đồ thị để so sánh Error % so với Optimal.'
)

# ============================================================
# 7. EXPORT / IMPORT MODEL
# ============================================================
doc.add_paragraph()
add_heading(doc, '7. Export / Import Model (DDPG)', 1, '1A73E8')
doc.add_paragraph(
    'DDPG Agent hỗ trợ lưu và tải lại trọng số Neural Network thông qua API của TensorFlow.js:\n\n'
    '• Export: Actor model được lưu thành 2 file:\n'
    '   - ddpg-actor-model.json  (cấu trúc mạng)\n'
    '   - ddpg-actor-model.weights.bin  (trọng số)\n\n'
    '• Import: Chọn cả 2 file trên cùng lúc. Trọng số được nạp vào Actor và đồng bộ sang Actor Target.\n\n'
    'Tính năng này cho phép lưu lại model tốt sau khi train, và khôi phục lại ngay lập tức '
    'ở phiên làm việc tiếp theo mà không cần train lại từ đầu.'
)

# ============================================================
# 8. HƯỚNG DẪN CÀI ĐẶT
# ============================================================
doc.add_paragraph()
add_heading(doc, '8. Hướng dẫn cài đặt & Chạy', 1, '1A73E8')

add_heading(doc, '8.1. Yêu cầu phần mềm', 2)
add_table(doc,
    ['Phần mềm', 'Phiên bản', 'Link'],
    [
        ['Node.js (LTS)', '≥ v18', 'https://nodejs.org'],
        ['Trình duyệt',   'Chrome / Edge / Firefox mới nhất', '—'],
    ]
)

add_heading(doc, '8.2. Các bước', 2)
doc.add_paragraph(
    'Bước 1 — Cài Node.js LTS từ https://nodejs.org, sau đó khởi động lại máy tính.\n\n'
    'Bước 2 — Mở PowerShell và cấp quyền chạy script (chỉ cần làm 1 lần):\n'
    '    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned\n\n'
    'Bước 3 — Cài thư viện (chỉ cần làm 1 lần):\n'
    '    cd "E:\\Python Projects\\Ant_Colony_QL_DQN"\n'
    '    npm install\n\n'
    'Bước 4 — Chạy ứng dụng:\n'
    '    npm run dev\n'
    '    → Mở trình duyệt tại http://localhost:5173/'
)

# ============================================================
# 9. TÍNH NĂNG UI
# ============================================================
doc.add_paragraph()
add_heading(doc, '9. Tính năng giao diện', 1, '1A73E8')
add_table(doc,
    ['Nút / Tính năng', 'Mô tả'],
    [
        ['▶ Bắt đầu / ⏸ Tạm dừng',  'Khởi động hoặc tạm dừng vòng lặp mô phỏng'],
        ['🚧 Tạo Tắc Đường',         'Kích hoạt chế độ chặn đường, click cạnh để tắc (×100 chi phí)'],
        ['❌ Xoá Tắc Đường',         'Gỡ toàn bộ điểm tắc, khôi phục chi phí gốc, AI reset baseline'],
        ['🔄 Khởi tạo lại',          'Tạo bản đồ ngẫu nhiên mới, reset toàn bộ'],
        ['🧠 Mode Switch (header)',   'Chuyển đổi nóng giữa Q-Learning ↔ DDPG'],
        ['⚡ Train DDPG (Fast)',      'Headless training: 2,000 gen không render, hoàn thành trong 2–5 giây'],
        ['💾 Export',                'Tải trọng số DDPG Actor xuống máy (JSON + BIN)'],
        ['📂 Import',                'Tải lại trọng số đã lưu (chọn cả 2 file)'],
        ['📊 Run Test Suite',        'Chạy 5 bài test chuẩn, so sánh QL vs DDPG vs Optimal Brute-force'],
        ['Canvas pheromone',         'Màu sắc cạnh đậm nhạt theo mức pheromone real-time'],
        ['Particle kiến',            'Hiệu ứng kiến chạy dọc lộ trình tốt nhất'],
        ['State Vector panel',       '10 thanh mini-bar visualize state DDPG cập nhật liên tục'],
        ['Biểu đồ hội tụ',          'Best Distance + Reward trên cùng chart, cập nhật mỗi gen'],
    ]
)

# ============================================================
# 10. SO SÁNH QL vs DDPG
# ============================================================
doc.add_paragraph()
add_heading(doc, '10. So sánh Q-Learning và DDPG', 1, '1A73E8')
add_table(doc,
    ['Tiêu chí', 'Q-Learning (Tabular)', 'DDPG (Deep RL)'],
    [
        ['Loại hành động',    'Rời rạc (3 action cố định)',          'Liên tục (vô hạn giá trị)'],
        ['Độ mịn điều chỉnh','Thô — nhảy giữa 3 mức preset',        'Mịn — tinh chỉnh từng 0.001'],
        ['Tốc độ hội tụ',    'Nhanh (không cần train trước)',        'Cần train headless 2,000 gen trước'],
        ['Bộ nhớ',           'Q-Table 3×3 = 9 giá trị',             'Neural Network ~400,000 tham số'],
        ['Khả năng tổng quát','Kém hơn (chỉ 3 trạng thái)',          'Tốt hơn (state 10D liên tục)'],
        ['Phản ứng tắc đường','Ngay lập tức (switch action)',        'Sau 1-2 gen (update reward)'],
        ['Export/Import',    'Không hỗ trợ',                         'Hỗ trợ (JSON + BIN)'],
        ['Ưu điểm chính',    'Đơn giản, nhanh, ổn định',            'Tinh tế, linh hoạt, scalable'],
    ]
)

# ============================================================
# LƯU FILE
# ============================================================
output_path = r'e:\Python Projects\Ant_Colony_QL_DQN\SmartRoute_v3_Overview.docx'
doc.save(output_path)
print(f'[OK] Saved: {output_path}')
