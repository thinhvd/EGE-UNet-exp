'''
Workbook for analysis/small_lesion_errors.py.

Only TP / FP / FN (and the two boundary shares) are stored as values: every derived column - areas,
recall, precision, DSC, error type - and every summary number is an Excel formula on those cells,
and the thresholds live in input cells on the first sheet. Changing a threshold re-sorts the error
types and updates the summary without re-running anything.
'''
import numpy as np
from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from analysis.small_lesion_errors import ERROR_TYPES, TYPE_GOOD

FONT = 'Arial'
F_BODY = Font(name=FONT, size=10)
F_BOLD = Font(name=FONT, size=10, bold=True)
F_TITLE = Font(name=FONT, size=13, bold=True)
F_H2 = Font(name=FONT, size=11, bold=True)
F_INPUT = Font(name=FONT, size=10, color='0000FF', bold=True)
F_NOTE = Font(name=FONT, size=9, italic=True, color='52514E')
FILL_HEAD = PatternFill('solid', fgColor='D6E4F0')
FILL_INPUT = PatternFill('solid', fgColor='FFF2CC')
FILL_LOW = PatternFill('solid', fgColor='F8D0CC')
FILL_REF = PatternFill('solid', fgColor='EEF3F8')
THIN = Side(style='thin', color='C3C2B7')
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP = Alignment(wrap_text=True, vertical='top')
CENTER = Alignment(horizontal='center', vertical='center', wrap_text=True)

GUIDE = 'Cách đọc'
SUMMARY = 'Tổng hợp'
WIDE = 'DSC từng ảnh'

# threshold input cells on the guide sheet
TH_ROWS = {'low': 5, 'high': 6, 'detect': 7, 'cover': 8, 'clean': 9}
LABEL_ROW0 = 13          # row of "Tốt"; the four error types follow


def q(sheet):
    return "'" + sheet.replace("'", "''") + "'"


def th(key):
    return f"{q(GUIDE)}!$B${TH_ROWS[key]}"


def label_ref(i):
    '''i = 0 for "Tốt", 1..4 for the error types.'''
    return f"{q(GUIDE)}!$A${LABEL_ROW0 + i}"


def put(ws, row, col, value, font=F_BODY, fill=None, fmt=None, align=None, border=True):
    c = ws.cell(row=row, column=col, value=value)
    c.font = font
    if fill:
        c.fill = fill
    if fmt:
        c.number_format = fmt
    if align:
        c.alignment = align
    if border:
        c.border = BOX
    return c


def header(ws, row, titles, widths=None):
    for j, t in enumerate(titles, 1):
        put(ws, row, j, t, F_BOLD, FILL_HEAD, align=CENTER)
    if widths:
        for j, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(j)].width = w


def cpu_sentence(c):
    if c['n_differ'] == 0:
        return (f'Hai cột "sát viền" và các ảnh minh hoạ dùng mask dự đoán chạy lại trên CPU; cả '
                f'{c["n_pairs"]} cặp ảnh–model cho TP/FP/FN trùng khớp tuyệt đối với lượt đánh giá chính thức.')
    s = (f'Hai cột "sát viền" và các ảnh minh hoạ dùng mask dự đoán chạy lại trên CPU. '
         f'{c["n_pairs"] - c["n_differ"]}/{c["n_pairs"]} cặp ảnh–model trùng khớp tuyệt đối với lượt đánh giá '
         f'chính thức; {c["n_differ"]} cặp còn lại lệch vài pixel nằm sát ngưỡng 0,5 do khác phần cứng '
         f'(GPU so với CPU) — tối đa {c["max_px"]} px, trung vị {c["median_px"]:g} px, DSC lệch tối đa '
         f'{c["max_dsc"]:.4f}. ')
    s += ('Diện tích ground truth trùng khớp 100%. ' if c['gt_all_equal'] else
          'CẢNH BÁO: có ảnh lệch cả diện tích ground truth. ')
    if c['type_changes'] == 0 and c['bucket_changes'] == 0:
        s += 'Không cặp nào bị đổi kiểu lỗi hay đổi nhóm DSC. '
    else:
        s += (f'{c["type_changes"]} cặp sẽ đổi kiểu lỗi và {c["bucket_changes"]} cặp sẽ đổi nhóm DSC nếu dùng '
              f'mask CPU. ')
    return s + 'Chi tiết: file kiem_tra_mask_cpu_vs_chinh_thuc.csv.'


def guide_sheet(wb, n, edges, a, hw, cpu_check):
    ws = wb.active
    ws.title = GUIDE
    ws.column_dimensions['A'].width = 30
    ws.column_dimensions['B'].width = 12
    ws.column_dimensions['C'].width = 95

    put(ws, 1, 1, 'Phân tích lỗi trên nhóm tổn thương NHỎ — EXP-04 (cross-stage fusion)', F_TITLE, border=False)
    put(ws, 2, 1, f'{n} ảnh val có diện tích tổn thương < {edges[0] * 100:.4f}% ảnh '
                  f'(tertile nhỏ nhất của 650 ảnh — đúng nhóm NHỎ trong báo cáo).', F_NOTE, border=False)

    put(ws, 4, 1, 'Ngưỡng (ô xanh có thể sửa — các sheet khác tự cập nhật)', F_H2, border=False)
    rows = [('low', 'DSC thấp', a.dsc_low, 'Ảnh có DSC < ngưỡng này được đếm là "DSC thấp".'),
            ('high', 'DSC cao (ảnh Tốt)', a.dsc_high,
             'Ảnh có DSC ≥ ngưỡng này được đếm là "DSC cao" và xếp vào nhóm Tốt (không phân loại lỗi).'),
            ('detect', 'Recall tối thiểu (tìm thấy)', a.detect,
             'Recall < ngưỡng này: vùng tô gần như không chồng lên tổn thương.'),
            ('cover', 'Recall phủ gần hết', a.cover,
             'Recall ≥ ngưỡng này: vùng tô phủ gần hết tổn thương.'),
            ('clean', 'Precision gọn', a.clean,
             'Precision ≥ ngưỡng này: vùng tô gần như nằm trọn trong tổn thương.')]
    for key, name, val, note in rows:
        r = TH_ROWS[key]
        put(ws, r, 1, name)
        put(ws, r, 2, val, F_INPUT, FILL_INPUT, fmt='0.00')
        put(ws, r, 3, note, align=WRAP)

    put(ws, LABEL_ROW0 - 1, 1, 'Kiểu lỗi (xét lần lượt từ trên xuống, chỉ cho ảnh chưa Tốt)', F_H2, border=False)
    rules = [(TYPE_GOOD, 'DSC ≥ ngưỡng DSC cao.'),
             (ERROR_TYPES[0], 'Recall < ngưỡng tìm thấy. Model tô ra chỗ khác hoặc gần như không tô gì — '
                              'vùng tô và vùng đúng gần như không chồng nhau.'),
             (ERROR_TYPES[1], 'Recall ≥ ngưỡng phủ và Precision < ngưỡng gọn. Model tìm đúng chỗ, phủ gần hết '
                              'tổn thương, nhưng tô lan ra cả vùng da lành xung quanh.'),
             (ERROR_TYPES[2], 'Các trường hợp còn lại: sai ở CẢ HAI phía đường viền (vừa tô thừa, vừa bỏ sót) — '
                              'đúng chỗ nhưng đường viền lệch.'),
             (ERROR_TYPES[3], 'Precision ≥ ngưỡng gọn và Recall < ngưỡng phủ. Vùng tô nằm trong tổn thương '
                              'nhưng quá nhỏ — bỏ sót phần rìa (kiểu này không có trong 3 kiểu ban đầu).')]
    for i, (lab, rule) in enumerate(rules):
        put(ws, LABEL_ROW0 + i, 1, lab, F_BOLD)
        ws.merge_cells(start_row=LABEL_ROW0 + i, start_column=2, end_row=LABEL_ROW0 + i, end_column=3)
        put(ws, LABEL_ROW0 + i, 2, rule, align=WRAP)
        ws.row_dimensions[LABEL_ROW0 + i].height = 28

    r = LABEL_ROW0 + 6
    put(ws, r, 1, 'Định nghĩa (mỗi ảnh, tại độ phân giải 256×256)', F_H2, border=False)
    defs = [('TP', 'Pixel tổn thương được tô đúng.'),
            ('FP', 'Pixel da lành bị tô nhầm (tô thừa).'),
            ('FN', 'Pixel tổn thương bị bỏ sót.'),
            ('Recall', 'TP / (TP + FN) — tỉ lệ tổn thương được tô trúng.'),
            ('Precision', 'TP / (TP + FP) — tỉ lệ vùng tô là tổn thương thật. Để trống nếu model không tô gì.'),
            ('DSC', '2·TP / (2·TP + FP + FN).'),
            ('FP sát viền (%)', f'Phần trăm pixel FP nằm trong {a.band} px ngay ngoài đường viền ground truth. '
                                'Cao = tô thừa chủ yếu là viền nở ra; thấp = tô lan ra xa.'),
            ('FN sát viền (%)', f'Phần trăm pixel FN nằm trong {a.band} px ngay trong đường viền ground truth. '
                                'Cao = bỏ sót chủ yếu ở rìa; thấp = bỏ sót cả phần lõi.')]
    for i, (k, v) in enumerate(defs, 1):
        put(ws, r + i, 1, k, F_BOLD)
        ws.merge_cells(start_row=r + i, start_column=2, end_row=r + i, end_column=3)
        put(ws, r + i, 2, v, align=WRAP)

    r = r + len(defs) + 2
    put(ws, r, 1, 'Nguồn số liệu và kiểm tra', F_H2, border=False)
    src = [
        'TP / FP / FN của từng ảnh lấy nguyên từ lượt đánh giá chính thức (file per_image_metrics_full.csv '
        'của mỗi run, chạy trên GPU) — đúng số liệu đứng sau báo cáo, không tính lại.',
        'Đã kiểm tra trước khi dùng: TP+FP+FN+TN = 65.536 ở mọi ảnh; TP+FN đúng bằng diện tích ground truth; '
        'DSC lưu trong file đúng bằng 2TP/(2TP+FP+FN); DSC trung bình nhóm nhỏ của từng model khớp bảng so sánh '
        'trong báo cáo; mIoU tính lại từ số đếm từng ảnh tái tạo đúng test_results.json.',
        'Ảnh được resize về 256×256, pixel được tô khi xác suất ≥ 0,5, ground truth là mask resize ≥ 0,5 — '
        'giống hệt pipeline đánh giá.',
        cpu_sentence(cpu_check),
    ]
    for i, s in enumerate(src, 1):
        ws.merge_cells(start_row=r + i, start_column=1, end_row=r + i, end_column=3)
        put(ws, r + i, 1, s, align=WRAP, border=False)
        ws.row_dimensions[r + i].height = 30 if len(s) < 260 else 58
    ws.freeze_panes = None


MODEL_COLS = ['Ảnh', 'Diện tích GT (px)', 'Diện tích GT (% ảnh)', 'Diện tích dự đoán (px)',
              'TP', 'FP', 'FN', 'Recall', 'Precision', 'DSC', 'Kiểu lỗi',
              'FP sát viền (%)', 'FN sát viền (%)']
MODEL_WIDTHS = [16, 11, 11, 12, 8, 8, 8, 9, 10, 9, 24, 11, 11]


def model_sheet(wb, label, recs, hw):
    ws = wb.create_sheet(label)
    header(ws, 1, MODEL_COLS, MODEL_WIDTHS)
    ws.row_dimensions[1].height = 32
    for i, r in enumerate(recs):
        row = i + 2
        put(ws, row, 1, r['image'])
        put(ws, row, 2, f'=E{row}+G{row}', fmt='#,##0')
        put(ws, row, 3, f'=B{row}/{hw}', fmt='0.00%')
        put(ws, row, 4, f'=E{row}+F{row}', fmt='#,##0')
        put(ws, row, 5, r['tp'], fmt='#,##0')
        put(ws, row, 6, r['fp'], fmt='#,##0')
        put(ws, row, 7, r['fn'], fmt='#,##0')
        put(ws, row, 8, f'=E{row}/(E{row}+G{row})', fmt='0.000')
        put(ws, row, 9, f'=IF(E{row}+F{row}=0,"",E{row}/(E{row}+F{row}))', fmt='0.000')
        put(ws, row, 10, f'=2*E{row}/(2*E{row}+F{row}+G{row})', fmt='0.000')
        put(ws, row, 11,
            f'=IF(J{row}>={th("high")},{label_ref(0)},'
            f'IF(H{row}<{th("detect")},{label_ref(1)},'
            f'IF(AND(H{row}>={th("cover")},I{row}<{th("clean")}),{label_ref(2)},'
            f'IF(AND(I{row}>={th("clean")},H{row}<{th("cover")}),{label_ref(4)},{label_ref(3)}))))')
        for col, key in ((12, 'fp_near_boundary_pct'), (13, 'fn_near_boundary_pct')):
            v = r[key]
            put(ws, row, col, None if v is None else round(v, 4), fmt='0.0')
    last = len(recs) + 1
    ws.conditional_formatting.add(f'J2:J{last}', CellIsRule(operator='lessThan', formula=['0.5'], fill=FILL_LOW))
    ws.freeze_panes = 'B2'
    ws.auto_filter.ref = f'A1:{get_column_letter(len(MODEL_COLS))}{last}'
    return last


def summary_sheet(wb, labels, last):
    ws = wb.create_sheet(SUMMARY, 1)
    put(ws, 1, 1, 'Kết quả trên nhóm tổn thương NHỎ, theo từng model', F_TITLE, border=False)
    put(ws, 2, 1, 'Mọi ô là công thức tính trên sheet của từng model; ngưỡng lấy từ sheet "Cách đọc". '
                  'Các giá trị trung bình là trung bình theo ảnh (mỗi ảnh có trọng số như nhau).',
        F_NOTE, border=False)
    titles = ['Model', 'Số ảnh', 'DSC trung bình', 'Recall trung bình', 'Precision trung bình',
              'DSC trung vị',
              f'="Số ảnh DSC < "&{th("low")}',
              f'="Số ảnh "&{th("low")}&" ≤ DSC < "&{th("high")}',
              f'="Số ảnh DSC ≥ "&{th("high")}',
              f'={label_ref(1)}', f'={label_ref(2)}', f'={label_ref(3)}', f'={label_ref(4)}',
              'Số ảnh model không tô gì']
    widths = [15, 8, 11, 11, 11, 10, 12, 14, 12, 15, 13, 13, 13, 12]
    header(ws, 4, titles, widths)
    ws.row_dimensions[4].height = 44
    for i, lab in enumerate(labels):
        row = 5 + i
        s = q(lab)
        J, H, I, K, D, A = (f'{s}!${c}$2:${c}${last}' for c in 'JHIKDA')
        fill = FILL_REF if i == 0 else None
        put(ws, row, 1, lab, F_BOLD if i == 0 else F_BODY, fill)
        put(ws, row, 2, f'=COUNTA({A})', fill=fill, fmt='0')
        put(ws, row, 3, f'=AVERAGE({J})', fill=fill, fmt='0.0000')
        put(ws, row, 4, f'=AVERAGE({H})', fill=fill, fmt='0.0000')
        put(ws, row, 5, f'=AVERAGE({I})', fill=fill, fmt='0.0000')
        put(ws, row, 6, f'=MEDIAN({J})', fill=fill, fmt='0.0000')
        put(ws, row, 7, f'=COUNTIF({J},"<"&{th("low")})', fill=fill, fmt='0')
        put(ws, row, 8, f'=COUNTIFS({J},">="&{th("low")},{J},"<"&{th("high")})', fill=fill, fmt='0')
        put(ws, row, 9, f'=COUNTIF({J},">="&{th("high")})', fill=fill, fmt='0')
        for k in range(4):
            put(ws, row, 10 + k, f'=COUNTIF({K},{label_ref(k + 1)})', fill=fill, fmt='0')
        put(ws, row, 14, f'=COUNTIF({D},0)', fill=fill, fmt='0')
    put(ws, 6 + len(labels), 1, 'Hàng tô nền: model gốc (mốc so sánh).', F_NOTE, border=False)
    ws.freeze_panes = 'B5'


def wide_sheet(wb, labels, last):
    ws = wb.create_sheet(WIDE, 2)
    first = q(labels[0])
    titles = ['Ảnh', 'Diện tích GT (px)', 'Diện tích GT (% ảnh)'] + [f'DSC {lab}' for lab in labels]
    header(ws, 1, titles, [16, 11, 11] + [11] * len(labels))
    ws.row_dimensions[1].height = 32
    for row in range(2, last + 1):
        put(ws, row, 1, f'={first}!A{row}')
        put(ws, row, 2, f'={first}!B{row}', fmt='#,##0')
        put(ws, row, 3, f'={first}!C{row}', fmt='0.00%')
        for j, lab in enumerate(labels):
            put(ws, row, 4 + j, f'={q(lab)}!J{row}', fmt='0.000')
    end_col = get_column_letter(3 + len(labels))
    ws.conditional_formatting.add(f'D2:{end_col}{last}',
                                  CellIsRule(operator='lessThan', formula=['0.5'], fill=FILL_LOW))
    ws.freeze_panes = 'B2'
    ws.auto_filter.ref = f'A1:{end_col}{last}'
    put(ws, last + 2, 1, 'Ô tô đỏ: DSC < 0,5. Cùng một hàng là cùng một ảnh ở tất cả model.', F_NOTE, border=False)


def write_workbook(path, labels, records, edges, n, a, hw, cpu_check):
    wb = Workbook()
    guide_sheet(wb, n, edges, a, hw, cpu_check)
    last = None
    for lab in labels:
        last = model_sheet(wb, lab, records[lab], hw)
    summary_sheet(wb, labels, last)
    wide_sheet(wb, labels, last)
    wb.save(path)
    print(f'wrote {path}')
