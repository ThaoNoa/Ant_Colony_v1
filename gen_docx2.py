from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

# ── Helpers ───────────────────────────────────────────────────────────────────
def set_cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), hex_color)
    shd.set(qn('w:val'), 'clear')
    tcPr.append(shd)

def heading(doc, text, level=1, rgb=(26,115,232)):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.color.rgb = RGBColor(*rgb)
    return p

def para(doc, text, size=11, bold=False, color=None, mono=False, indent=False):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.left_indent = Cm(0.8)
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)
    if mono:
        run.font.name = 'Courier New'
    return p

def code(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.8)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(4)
    run = p.add_run(text)
    run.font.name = 'Courier New'
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x1a, 0x1a, 0x2e)
    # light grey bg via shading on paragraph
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), 'F0F0F0')
    shd.set(qn('w:val'), 'clear')
    pPr.append(shd)
    return p

def table(doc, headers, rows, hdr_color='1F3864'):
    t = doc.add_table(rows=1+len(rows), cols=len(headers))
    t.style = 'Table Grid'
    hdr_row = t.rows[0]
    for i, h in enumerate(headers):
        c = hdr_row.cells[i]
        c.text = h
        set_cell_bg(c, hdr_color)
        for run in c.paragraphs[0].runs:
            run.font.bold  = True
            run.font.color.rgb = RGBColor(255,255,255)
            run.font.size  = Pt(10)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for ri, row_data in enumerate(rows):
        row = t.rows[ri+1]
        for ci, val in enumerate(row_data):
            c = row.cells[ci]
            c.text = val
            c.paragraphs[0].runs[0].font.size = Pt(9.5)
            if ri % 2 == 0:
                set_cell_bg(c, 'EBF3FB')
    return t

def divider(doc):
    doc.add_paragraph('─' * 80).runs[0].font.color.rgb = RGBColor(180,180,180)

# ── Document ──────────────────────────────────────────────────────────────────
doc = Document()
for sec in doc.sections:
    sec.top_margin    = Cm(2.0)
    sec.bottom_margin = Cm(2.0)
    sec.left_margin   = Cm(2.5)
    sec.right_margin  = Cm(2.5)

# ══════════════════════════════════════════
# TRANG BÌA
# ══════════════════════════════════════════
doc.add_paragraph()
doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run('SmartRoute v3')
r.font.size  = Pt(34)
r.font.bold  = True
r.font.color.rgb = RGBColor(0x1A, 0x73, 0xE8)

p2 = doc.add_paragraph()
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r2 = p2.add_run('Tổng quan mô hình ACO+QL và ACO+DDPG\nQuy trình hoạt động & So sánh cải tiến')
r2.font.size = Pt(14)
r2.font.color.rgb = RGBColor(80,80,80)

doc.add_paragraph()
pd = doc.add_paragraph()
pd.alignment = WD_ALIGN_PARAGRAPH.CENTER
pd.add_run(f'Ngày: {datetime.date.today().strftime("%d/%m/%Y")}').font.color.rgb = RGBColor(140,140,140)

doc.add_page_break()

# ══════════════════════════════════════════
# 1. BÀI TOÁN GỐC
# ══════════════════════════════════════════
heading(doc, '1. Bài toán: Dynamic TSP', 1)
para(doc,
    'Bài toán Người Bán Hàng Động (Dynamic TSP) yêu cầu tìm lộ trình ngắn nhất '
    'đi qua tất cả N điểm giao hàng và quay về Kho — nhưng trong khi đang tìm đường, '
    'bản đồ có thể thay đổi đột ngột (xuất hiện tắc đường). '
    'Thuật toán phải vừa tìm đường tối ưu, vừa liên tục thích ứng với thay đổi trong thời gian thực.'
)
para(doc,
    'Điểm khác biệt so với TSP tĩnh: không thể tính trước một lần rồi dùng mãi. '
    'Mỗi khi có tắc đường, lộ trình cũ có thể không còn hợp lệ và hệ thống phải '
    'tìm lại đường thay thế nhanh nhất có thể — trong khi kiến vẫn đang chạy.'
)

# ══════════════════════════════════════════
# 2. NỀN TẢNG CHUNG: ACO ENGINE
# ══════════════════════════════════════════
doc.add_paragraph()
heading(doc, '2. Nền tảng chung: ACO Engine', 1)
para(doc,
    'Cả 2 mô hình đều dùng chung 1 lõi ACO. Mỗi thế hệ, 30 con kiến ảo xuất phát '
    'từ Kho hàng (Depot) và xây dựng lộ trình theo xác suất:'
)
code(doc,
    'P(i → j)  ∝  τ(i,j)^α  ×  η(i,j)^β\n\n'
    '  τ(i,j)  : mức pheromone trên cạnh — "mùi" từ kiến trước để lại\n'
    '  η(i,j)  : heuristic = 1 / cost(i,j)  — ưu tiên cạnh ngắn hơn\n'
    '  α        : trọng số pheromone  ← AI điều chỉnh\n'
    '  β        : trọng số heuristic = 2.0  (cố định)'
)
para(doc, 'Cập nhật pheromone sau mỗi thế hệ:')
code(doc,
    'τ(i,j)  ←  τ(i,j) × (1 - ρ)  +  Δτ\n\n'
    '  ρ   : tốc độ bay hơi  ← AI điều chỉnh\n'
    '  Δτ  : lượng pheromone top 30% kiến tốt nhất deposit (Elitist Strategy)'
)
para(doc,
    'Nhiệm vụ duy nhất của AI (dù là QL hay DDPG): Quyết định α và ρ '
    'tối ưu nhất cho từng tình huống.'
)

# ══════════════════════════════════════════
# 3. MÔ HÌNH 1: ACO + Q-LEARNING
# ══════════════════════════════════════════
doc.add_paragraph()
heading(doc, '3. Mô hình 1: ACO + Q-Learning (Tabular)', 1, rgb=(0,120,60))

heading(doc, '3.1. Ý tưởng thiết kế', 2, rgb=(0,120,60))
para(doc,
    'Q-Learning giải quyết bài toán điều khiển ACO bằng cách rời rạc hoá mọi thứ. '
    'Thay vì xử lý vô số tình huống liên tục, hệ thống ép toàn bộ thực tế vào '
    '3 trạng thái và 3 hành động — đơn giản nhưng hiệu quả trên bản đồ nhỏ.'
)

heading(doc, '3.2. Không gian trạng thái', 2, rgb=(0,120,60))
table(doc,
    ['State', 'Tên', 'Điều kiện kích hoạt'],
    [
        ['S0', 'IMPROVING', 'Best distance đang giảm qua các thế hệ'],
        ['S1', 'STUCK',     '≥10 gen liên tiếp không có kỷ lục mới'],
        ['S2', 'DEGRADED',  'Tắc đường xảy ra / distance tăng đột ngột'],
    ], '145A32'
)

heading(doc, '3.3. Không gian hành động', 2, rgb=(0,120,60))
table(doc,
    ['Action', 'Tên', 'α (Alpha)', 'ρ (Rho)', 'Chiến lược'],
    [
        ['A0', 'MAINTAIN (Duy trì)',  '1.0', '0.10', 'Bình thường, giữ ổn định'],
        ['A1', 'EXPLORE  (Khám phá)', '0.5', '0.80', 'Xóa mùi cũ, mở rộng tìm kiếm'],
        ['A2', 'EXPLOIT  (Khai thác)', '2.0','0.05', 'Bám chặt lộ trình tốt nhất'],
    ], '145A32'
)

heading(doc, '3.4. Quy trình hoạt động mỗi thế hệ', 2, rgb=(0,120,60))
code(doc,
    '[1]  Đọc trạng thái S hiện tại (IMPROVING / STUCK / DEGRADED)\n'
    '[2]  Tra Q-Table → chọn Action A có Q-value cao nhất (ε-greedy)\n'
    '[3]  Áp α, ρ từ Action A vào ACO Engine\n'
    '[4]  30 kiến xây lộ trình → tìm best_cost vòng này\n'
    '[5]  Cập nhật pheromone theo lộ trình tốt nhất (Elitist)\n'
    '[6]  Tính Reward:\n'
    '       +10  nếu kỷ lục mới\n'
    '       +1   nếu ổn định\n'
    '       -5   nếu vẫn kẹt\n'
    '       -10  nếu tệ hơn\n'
    '[7]  Cập nhật Q-Table:\n'
    '       Q(S,A) ← Q(S,A) + lr × [R + γ × max Q(S\') - Q(S,A)]\n'
    '[8]  Chuyển sang trạng thái mới S\', lặp lại'
)

heading(doc, '3.5. Ưu điểm & Hạn chế', 2, rgb=(0,120,60))
table(doc,
    ['Ưu điểm', 'Hạn chế'],
    [
        ['Hội tụ cực nhanh (không cần train trước)',
         'Chỉ 3 mức α/ρ cứng nhắc — không tinh chỉnh mịn'],
        ['Phản ứng ngay khi tắc đường (switch action 1 gen)',
         '3 trạng thái quá thô — không phân biệt "kẹt nhẹ" vs "kẹt nặng"'],
        ['Đơn giản, ổn định, dễ debug (xem thẳng Q-Table)',
         'Không tổng quát hóa được giữa bản đồ khác nhau'],
        ['Không cần GPU, chạy ngay trên CPU browser',
         'Bộ nhớ chỉ 9 số — không học được pattern phức tạp'],
    ]
)

# ══════════════════════════════════════════
# 4. MÔ HÌNH 2: ACO + DDPG
# ══════════════════════════════════════════
doc.add_paragraph()
heading(doc, '4. Mô hình 2: ACO + DDPG (Deep Reinforcement Learning)', 1, rgb=(180,0,180))

heading(doc, '4.1. Cải tiến so với Q-Learning', 2, rgb=(180,0,180))
table(doc,
    ['Vấn đề của Q-Learning', 'Giải pháp DDPG'],
    [
        ['Chỉ 3 mức α/ρ cứng nhắc',
         'Continuous action: α∈[0.5,2.5], ρ∈[0.02,0.50] — vô hạn giá trị'],
        ['3 trạng thái thô',
         'State vector 10 chiều liên tục — chi tiết, phong phú hơn nhiều'],
        ['Không học từ lịch sử xa',
         'Replay Buffer 5,000 transitions — học từ kinh nghiệm ngẫu nhiên'],
        ['Q-Table không tổng quát hóa',
         'Neural Network — tổng quát hóa giữa các tình huống tương tự'],
        ['Chỉ phản ứng 3 mức nhảy bậc',
         'Điều chỉnh liên tục từng 0.001 — cực kỳ mịn và chính xác'],
    ], '6B0080'
)

heading(doc, '4.2. Kiến trúc 4 mạng Neural (Actor-Critic)', 2, rgb=(180,0,180))
para(doc, 'DDPG dùng 4 mạng để ổn định quá trình học — 2 mạng online + 2 mạng target:')
code(doc,
    'Actor  (online) : state(10D) → Dense256 → Dense256 → Dense128 → tanh → action(2D)\n'
    'Actor  (target) : bản sao mềm của Actor online\n'
    'Critic (online) : [state ∥ action](12D) → Dense256 → Dense256 → Dense128 → Q(1D)\n'
    'Critic (target) : bản sao mềm của Critic online\n\n'
    'Soft update (Polyak averaging, τ=0.005):\n'
    '  θ_target ← τ × θ_online + (1-τ) × θ_target\n'
    '  → Cập nhật chậm, tránh diverge, ổn định học'
)

heading(doc, '4.3. State Vector 10 chiều', 2, rgb=(180,0,180))
table(doc,
    ['Chiều', 'Feature', 'Ý nghĩa', 'Dải'],
    [
        ['[0]', 'normalized_best_cost',  'best_cost / reference_cost — đang tốt hay xấu?',       '[0, 3]'],
        ['[1]', 'improvement_rate',      '(prev-curr)/prev — đang tiến bộ hay không?',            '[-1,1]'],
        ['[2]', 'stuck_counter_norm',    'Số gen kẹt / 15 — kẹt bao lâu rồi?',                   '[0, 1]'],
        ['[3]', 'avg_pheromone',         'mean(τ)/τ_max — mùi đậm hay nhạt?',                     '[0, 1]'],
        ['[4]', 'std_pheromone',         'std(τ)/τ_max — mùi đều hay tập trung 1 vài cạnh?',     '[0, 1]'],
        ['[5]', 'blocked_edge_ratio',    'cạnh tắc / tổng cạnh — tắc đường bao nhiêu %?',        '[0, 1]'],
        ['[6]', 'alpha_norm',            '(α - 0.5) / 2.0 — đang dùng α bao nhiêu?',             '[0, 1]'],
        ['[7]', 'rho_norm',              '(ρ - 0.02) / 0.48 — đang dùng ρ bao nhiêu?',           '[0, 1]'],
        ['[8]', 'generation_progress',   '(gen % 100) / 100 — vị trí trong chu kỳ 100 gen',      '[0, 1]'],
        ['[9]', 'pheromone_entropy',     'entropy(phero) — kiến đa dạng hay bị lock-in 1 lối?',  '[0, 1]'],
    ], '6B0080'
)

heading(doc, '4.4. Quy trình hoạt động mỗi thế hệ', 2, rgb=(180,0,180))
code(doc,
    '[1]  getState() → lấy state vector 10D từ môi trường\n\n'
    '[2]  Actor.predict(state) → raw_action ∈ [-1, 1]²\n'
    '     + Ornstein-Uhlenbeck Noise (giảm dần σ: 0.3 → 0.05)\n'
    '     → action = clip(raw + noise, -1, 1)\n\n'
    '[3]  Scale action → {α, ρ} thực tế:\n'
    '       α = 0.5 + (action[0]+1)/2 × 2.0     →  α ∈ [0.5, 2.5]\n'
    '       ρ = 0.02 + (action[1]+1)/2 × 0.48   →  ρ ∈ [0.02, 0.50]\n\n'
    '[4]  ACO chạy với α, ρ vừa chọn → best_cost vòng này\n\n'
    '[5]  Tính Reward (3 thành phần — xem mục 4.5)\n\n'
    '[6]  Lưu transition (s, a, r, s\') vào Replay Buffer (5,000 slots)\n\n'
    '[7]  Nếu buffer ≥ 200 mẫu → TRAIN:\n'
    '     a) Sample ngẫu nhiên 64 transitions từ buffer\n'
    '     b) TD target:  y = r + γ × Critic_target(s\', Actor_target(s\'))\n'
    '     c) Update Critic: minimize MSE(Critic(s,a), y)\n'
    '     d) Update Actor:  maximize E[Critic(s, Actor(s))]  ← gradient ascent\n'
    '     e) Soft-update target networks (τ=0.005)\n\n'
    '[8]  Lặp lại — noise σ giảm dần → Agent khai thác nhiều hơn'
)

heading(doc, '4.5. Hàm phần thưởng — 3 thành phần', 2, rgb=(180,0,180))
para(doc, 'Thiết kế 3 thành phần cộng lại để tránh Agent khai thác lỗ hổng (Reward Exploitation):')
code(doc,
    'R1 — Kết quả ACO:\n'
    '  +20  nếu cải thiện ≥ 5%          (tìm đường tốt hơn nhiều)\n'
    '  +10  nếu cải thiện > 0%          (cải thiện nhỏ)\n'
    '  -15  nếu tắc đường + tệ hơn      (tắc đường gây thiệt hại nặng)\n'
    '   +2  nếu giữ phong độ tốt        (≤ 2% so với allTimeBest)\n'
    '   -5  nếu kẹt ≥ 15 gen            (stuck nặng)\n'
    '   -1  nếu lang thang              (không tiến, không kẹt)\n\n'
    'R2 — Sức khoẻ tham số (Bell-curve Gaussian — phạt cả 2 cực đoan):\n'
    '  alphaHealth = exp(-((αNorm - 0.35) / 0.25)²)   đỉnh tại α ≈ 1.0\n'
    '  rhoHealth   = exp(-((ρNorm - 0.20) / 0.25)²)   đỉnh tại ρ ≈ 0.10\n'
    '  paramScore  = (alphaHealth + rhoHealth - 1.0) × 2   ∈ [-2, +2]\n\n'
    'R3 — Entropy pheromone:\n'
    '  -3  nếu entropy < 0.15            (lock-in: kiến chỉ đi 1 lối duy nhất)\n'
    '  +1  nếu entropy ∈ [0.25, 0.75]   (vùng khám phá lành mạnh)\n\n'
    'Tổng: reward = R1 + R2 + R3'
)

heading(doc, '4.6. Tại sao cần 3 thành phần?', 2, rgb=(180,0,180))
table(doc,
    ['Nếu chỉ có...', 'Agent sẽ khai thác lỗ hổng bằng cách...', 'Hậu quả'],
    [
        ['R1 (kết quả ACO)',
         'Giữ α thấp + ρ cao → kiến lang thang, không bao giờ "kẹt" → +2 mãi mãi',
         'Pheromone loãng, không hội tụ'],
        ['R1 + R2 (thêm bell-curve)',
         'Đẩy α cao + ρ thấp → đúng vùng bell-curve, nhưng lock-in 1 đường duy nhất',
         'Không phản ứng được khi tắc đường'],
        ['R1 + R2 + R3 (đầy đủ)',
         'Không có lỗ hổng nào để khai thác — entropy thấp bị phạt ngay',
         '✓ Hành vi lành mạnh: hội tụ + linh hoạt'],
    ]
)

heading(doc, '4.7. Headless Training', 2, rgb=(180,0,180))
para(doc,
    '"Headless" = chạy không có giao diện đồ họa (bỏ qua bước render Canvas). '
    'Khi bấm ⚡ Train DDPG (Fast), hệ thống chạy thuần túy vòng lặp ACO + DDPG học '
    'trong JavaScript. 2,000 thế hệ hoàn thành trong 2–5 giây thay vì ~4 phút nếu render đầy đủ.'
)
code(doc,
    'Render bình thường:   1 gen = ~120ms  →  2,000 gen ≈ 4 phút\n'
    'Headless training:    1 gen = ~2ms    →  2,000 gen ≈ 4 giây\n'
    '                                         (nhanh hơn 60 lần)'
)

# ══════════════════════════════════════════
# 5. SO SÁNH TỔNG THỂ
# ══════════════════════════════════════════
doc.add_paragraph()
heading(doc, '5. So sánh tổng thể', 1)
table(doc,
    ['Tiêu chí', 'ACO + Q-Learning', 'ACO + DDPG'],
    [
        ['Loại action',         '3 giá trị rời rạc cố định',        'Liên tục — vô hạn giá trị'],
        ['Độ mịn điều chỉnh',  'Thô — nhảy giữa 3 mức preset',     'Mịn — tinh chỉnh từng 0.001'],
        ['Không gian state',    '3 trạng thái thô',                  '10 chiều liên tục, phong phú'],
        ['Bộ nhớ',             'Q-Table 9 số',                      'Neural Network ~400,000 tham số'],
        ['Khởi động',          'Ngay lập tức, không cần train',      'Cần Headless Train ~2,000 gen'],
        ['Tốc độ hội tụ',      'Nhanh (~50 gen)',                    'Trung bình (~100-200 gen)'],
        ['Phản ứng tắc đường', 'Ngay (1 gen)',                       '1-2 gen'],
        ['Tổng quát hóa',      'Kém (chỉ 3 tình huống)',             'Tốt hơn (state 10D)'],
        ['Export model',       '✗ Không hỗ trợ',                    '✓ JSON + BIN (TF.js)'],
        ['Phù hợp với',        'Map nhỏ, cần phản ứng nhanh',       'Map phức tạp, cần độ tinh tế'],
    ]
)

para(doc,
    'Kết luận: QL-ACO là lựa chọn mặc định tốt — nhanh, đơn giản, ổn định. '
    'DDPG-ACO phù hợp khi cần điều chỉnh mịn và học được chiến lược phức tạp '
    'qua nhiều lần tương tác với môi trường thay đổi liên tục.',
    bold=True
)

# ══════════════════════════════════════════
# LƯU FILE
# ══════════════════════════════════════════
out = r'e:\Python Projects\Ant_Colony_QL_DQN\SmartRoute_v3_Model_Overview.docx'
doc.save(out)
print(f'[OK] Saved: {out}')
