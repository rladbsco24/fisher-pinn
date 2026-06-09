from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
BLACK = RGBColor(0, 0, 0)
FONT = "Malgun Gothic"
BODY_SIZE = Pt(10)


def _set_rfonts(obj, font_name: str = FONT) -> None:
    rpr = obj._element.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), font_name)
    rfonts.set(qn("w:hAnsi"), font_name)
    rfonts.set(qn("w:eastAsia"), font_name)
    rfonts.set(qn("w:cs"), font_name)


def _format_run(run, *, bold: bool = False, italic: bool = False) -> None:
    run.font.name = FONT
    run.font.size = BODY_SIZE
    run.font.color.rgb = BLACK
    run.font.bold = bold
    run.font.italic = italic
    _set_rfonts(run)


def _set_paragraph_style(paragraph, *, before: float = 0.0, after: float = 4.0, line_spacing: float = 1.05) -> None:
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.line_spacing = line_spacing


def _configure_document(document: Document, title: str) -> None:
    section = document.sections[0]
    section.top_margin = Inches(0.62)
    section.bottom_margin = Inches(0.62)
    section.left_margin = Inches(0.68)
    section.right_margin = Inches(0.68)
    section.header_distance = Inches(0.35)
    section.footer_distance = Inches(0.35)

    styles = document.styles
    for style_name in ("Normal", "Heading 1", "Heading 2", "Heading 3"):
        style = styles[style_name]
        style.font.name = FONT
        style.font.size = BODY_SIZE
        style.font.color.rgb = BLACK
        _set_rfonts(style)
        style.paragraph_format.line_spacing = 1.05
    styles["Normal"].paragraph_format.space_after = Pt(4)
    for heading in ("Heading 1", "Heading 2", "Heading 3"):
        styles[heading].font.bold = True
        styles[heading].paragraph_format.space_before = Pt(4)
        styles[heading].paragraph_format.space_after = Pt(4)

    header = section.header.paragraphs[0]
    header.text = title
    header.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_paragraph_style(header, after=0)
    for run in header.runs:
        _format_run(run, bold=False)


def _add_title(document: Document, title: str, subtitle: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_paragraph_style(paragraph, after=2)
    run = paragraph.add_run(title)
    _format_run(run, bold=True)

    sub = document.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_paragraph_style(sub, after=8)
    run = sub.add_run(subtitle)
    _format_run(run)


def _add_heading(document: Document, text: str, level: int = 1) -> None:
    paragraph = document.add_paragraph(style=f"Heading {level}")
    _set_paragraph_style(paragraph, before=4, after=4)
    run = paragraph.add_run(text)
    _format_run(run, bold=True)


def _add_para(document: Document, text: str, *, after: float = 4.0) -> None:
    paragraph = document.add_paragraph()
    _set_paragraph_style(paragraph, after=after)
    run = paragraph.add_run(text)
    _format_run(run)


def _set_cell_text(cell, text: str, *, bold: bool = False, align: WD_ALIGN_PARAGRAPH = WD_ALIGN_PARAGRAPH.LEFT) -> None:
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    paragraph = cell.paragraphs[0]
    paragraph.alignment = align
    _set_paragraph_style(paragraph, before=0, after=0, line_spacing=1.0)
    paragraph.text = ""
    run = paragraph.add_run(text)
    _format_run(run, bold=bold)


def _set_cell_margins(table) -> None:
    tbl_pr = table._tbl.tblPr
    margins = tbl_pr.find(qn("w:tblCellMar"))
    if margins is None:
        margins = OxmlElement("w:tblCellMar")
        tbl_pr.append(margins)
    for tag, value in (
        ("top", "90"),
        ("bottom", "90"),
        ("left", "140"),
        ("right", "140"),
        ("start", "140"),
        ("end", "140"),
    ):
        node = margins.find(qn(f"w:{tag}"))
        if node is None:
            node = OxmlElement(f"w:{tag}")
            margins.append(node)
        node.set(qn("w:w"), value)
        node.set(qn("w:type"), "dxa")


def _set_table_geometry(table, widths: list[float]) -> None:
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    total_dxa = int(round(sum(widths) * 1440))
    width_dxa = [int(round(width * 1440)) for width in widths]

    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:type"), "dxa")
    tbl_w.set(qn("w:w"), str(total_dxa))

    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:type"), "dxa")
    tbl_ind.set(qn("w:w"), "0")

    layout = tbl_pr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")

    old_grid = tbl.find(qn("w:tblGrid"))
    if old_grid is not None:
        tbl.remove(old_grid)
    grid = OxmlElement("w:tblGrid")
    for width in width_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    tbl.insert(0, grid)

    for row in table.rows:
        for idx, cell in enumerate(row.cells):
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:type"), "dxa")
            tc_w.set(qn("w:w"), str(width_dxa[idx]))


def _add_table(document: Document, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.style = "Table Grid"
    _set_cell_margins(table)
    _set_table_geometry(table, widths)
    for idx, header in enumerate(headers):
        table.columns[idx].width = Inches(widths[idx])
        _set_cell_text(table.rows[0].cells[idx], header, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    for row in rows:
        cells = table.add_row().cells
        for idx, value in enumerate(row):
            cells[idx].width = Inches(widths[idx])
            _set_cell_text(cells[idx], value)
    _set_table_geometry(table, widths)
    spacer = document.add_paragraph()
    _set_paragraph_style(spacer, after=4)


def _add_reference(document: Document, text: str) -> None:
    paragraph = document.add_paragraph()
    _set_paragraph_style(paragraph, before=0, after=3, line_spacing=1.0)
    paragraph.paragraph_format.first_line_indent = Inches(-0.18)
    paragraph.paragraph_format.left_indent = Inches(0.18)
    run = paragraph.add_run(text)
    _format_run(run)


PINN_PAGES: list[tuple[str, list[str], tuple[list[str], list[list[str]], list[float]] | None]] = [
    (
        "1. 구현 범위와 핵심 구조",
        [
            "현재 저장소는 기본 Fisher-KPP PINN과 한국 산림청 pine-wilt PINN을 하나의 반응-확산 모델 계열로 구성한다. 모델은 Fisher-KPP 방정식, known initial condition, moving-front 보정 손실, 계수 추정, RK4 기준해 비교를 통합해 forward field와 front geometry를 함께 학습한다.",
            "기본 계열은 정규화된 2차원 Fisher-KPP forward 문제를 대상으로 한다. 코드상 기본 실행은 `scripts/run_inverse_origin.py`의 `geo_spectral_forward` 프로파일이며, 이전 inverse-origin 데모보다 forward 해석, known initial condition, front-aware loss, adaptive sampling, RK4 비교를 더 명시적으로 포함한다.",
            "산림청 계열은 같은 반응-확산 구조를 한국 pine-wilt 관측 밀도 격자에 적용한다. 구현은 land mask, sea exclusion, 조치 전 월별 시간축, 물리 계수 변환, 관측 밀도 가중치를 동일한 forward PINN/RK4 비교 체계 안에서 처리한다.",
        ],
        (
            ["항목", "현재 구현 기준"],
            [
                ["기본문제", "Fisher-KPP reaction-diffusion, 필요 시 advection을 제거한 forward PINN"],
                ["기본 PINN", "PirateNet/RWF 스타일 backbone, Fourier/front feature, hard IC, front-aware losses"],
                ["산림청 PINN", "land-only collocation, sea penalty, D/r 학습, 공간 계수장, 월별 조치 전 비교"],
            ],
            [1.45, 5.60],
        ),
    ),
    (
        "2. 기본 Fisher-KPP 문제와 기준해",
        [
            "기본 PDE는 u_t = D Laplacian(u) + r u(1-u)이다. Gaussian seed 문제에서는 정규화된 단위 정사각형에서 초기 국소 감염원이 퍼지는 형태를 사용하고, Ablowitz-Zeppetella 계열 검증에서는 u(x,t) = (1 + exp((x - c t - x0)/sqrt(6)))^-2, c = 5/sqrt(6)인 알려진 travelling wave를 기준으로 삼는다.",
            "정규화는 코드가 공간을 [0, 1] 좌표로 다루기 때문에 중요하다. 예를 들어 AZ 1D 기준은 x in [-20, 20]을 단위 좌표로 변환하면서 D_norm = 1 / 40^2로 들어간다. 이 때문에 수치적으로 보이는 diffusion 값은 물리식의 D=1과 다르지만, 변환을 거친 PDE는 같은 front speed와 wave profile을 나타낸다.",
            "기본 PINN과 RK4의 오차 차이는 두 방법의 계산 대상에서 발생한다. RK4는 알려진 초기장과 PDE를 고정 격자에서 직접 시간 적분하고, PINN은 연속 좌표 함수 u(x,y,t)를 신경망으로 근사하면서 PDE residual, 관측, 초기조건, front geometry를 동시에 만족시킨다. RK4는 forward reference solver로 높은 격자 정확도를 제공하고, PINN은 inverse/irregular data/연속 surrogate 기능을 제공한다.",
        ],
        (
            ["기준", "의미"],
            [
                ["AZ exact wave", "front phase와 low-level ring을 검증하기 위한 정확한 travelling-wave 기준"],
                ["Gaussian seed", "국소 발병원이 2D에서 확산되는 synthetic forward benchmark"],
                ["RK4 reference", "PINN 최종장과 front metrics를 비교하는 같은 문제의 수치 기준"],
            ],
            [1.45, 5.60],
        ),
    ),
    (
        "3. 기본 PINN의 모델 구조",
        [
            "기본 forward PINN은 `OriginPINN`을 사용하며 `geo_spectral_forward` 프로파일에서 architecture는 pirate, Fourier sigma는 완화된 값, random weight factorization은 켜진다. 입력은 x, y, t와 함께 seed-relative geometry, travelling-wave 좌표, front-local Fourier feature, KPP front envelope, 공간 계수장을 사용한다.",
            "hard initial condition은 t=0 부근에서 네트워크 출력이 초기장을 쉽게 무시하지 못하도록 한다. 이전 결과에서 PINN이 처음부터 퍼진 haze처럼 보였던 현상은 sparse front 문제에서 MSE와 PDE residual만으로는 대부분의 영역을 낮은 값으로 흐리게 만드는 해가 벌점을 적게 받을 수 있기 때문에 발생했다. 현재 구현은 known IC loss와 hard IC를 동시에 써서 이 현상을 줄인다.",
            "D와 r은 기본 forward 프로파일에서 학습 가능하다. 학습 계수는 physics parameter anchor, 공간 계수 regularization, front-speed 관련 loss로 안정화된다. 이 설계는 산림청 zip에서 D_tilde와 r_tilde를 PINN으로 추정한 뒤 D_phys = D_tilde * S_km^2 / DT, r_phys = r_tilde / DT로 해석하던 원리를 확장한다.",
        ],
        (
            ["구성", "역할"],
            [
                ["Pirate/RWF backbone", "front와 residual이 강한 비선형장에서도 gradient 흐름을 안정화"],
                ["front-local features", "예상 front 주변의 moving coordinate를 네트워크 입력에 추가"],
                ["spatial coefficients", "D(x,y), r(x,y)의 약한 보정장을 허용하되 regularization으로 제한"],
            ],
            [1.65, 5.40],
        ),
    ),
    (
        "4. front-aware 손실과 moving-front 보정",
        [
            "front-level-set alignment loss는 expected low-level ring 주변에서 u의 level, 안쪽/바깥쪽 순서 관계, front-normal slope를 함께 제약한다. Fisher-KPP leading edge가 만드는 등치선 위치를 soft geometric regularizer로 사용해 front phase와 front thickness를 직접 정렬한다.",
            "front profile loss는 expected front corridor에서 선형화된 KPP leading-edge profile을 맞추도록 한다. low-level area loss만 쓰면 넓은 저농도 haze가 통과할 수 있으므로, support Tversky, contrast, mass floor, mass balance를 함께 둔다. 이는 대부분을 0으로 예측하는 해와 넓게 흐린 해를 모두 비용이 큰 방향으로 밀어내기 위한 장치다.",
            "학습은 time marching과 time slab curriculum을 포함한다. 초반에는 쉬운 시간 구간과 초기조건을 먼저 고정하고, 뒤로 갈수록 전체 시간축, front loss, PDE residual을 확대한다. RAR와 front-aware sampler는 residual, u(1-u), gradient 정보를 결합해 active front 주변 collocation을 보강한다.",
        ],
        (
            ["손실/샘플링", "보정 대상"],
            [
                ["level-set/profile", "front phase가 늦거나 빠지고 halo가 생기는 문제"],
                ["support/contrast/mass", "all-zero 해 또는 넓은 저농도 haze 해"],
                ["time marching/RAR", "parabolic PDE의 장시간 누적 오차와 front-local residual 누락"],
            ],
            [1.75, 5.30],
        ),
    ),
    (
        "5. 기본 PINN 검증과 관찰 해석",
        [
            "현재 코드의 full run은 기본적으로 1200 epochs이며, quick run은 60 또는 120 epochs로 빠른 회귀 검증을 수행한다. README에 기록된 최근 60-epoch 검증은 `geo_spectral_forward().quick()`에서 weak RK4 teacher와 front-profile 조합을 사용했을 때 final_time_relative_l2 = 0.2682, validation_observation_mse = 5.49e-4, front_area_010_mae = 0.0119, mass_mae = 0.0025를 보였다.",
            "같은 설정의 RK4 reference는 rk4_final_time_relative_l2 = 0.00465를 제공한다. 이 수치는 forward 적분 reference의 격자 정확도이며, PINN 결과는 continuous/inverse-capable surrogate가 관측, PDE residual, 계수, front geometry를 동시에 만족시키는 최적화 정확도다. 두 값은 같은 문제를 서로 다른 계산 형식으로 평가한다.",
            "주요 관찰은 네 가지다. 첫째, front area와 mass metric은 field L2보다 front failure를 더 선명하게 드러낸다. 둘째, PINN error map은 active front 주변에서 가장 높은 정보를 제공한다. 셋째, support, contrast, mass floor, IC는 all-zero 해와 diffuse haze 해를 효과적으로 억제한다. 넷째, D/r 학습은 forward field와 front speed의 물리 일관성을 함께 추정하는 핵심 구성이다.",
        ],
        (
            ["지표", "해석"],
            [
                ["final-time L2", "전체장 forward 정확도, RK4와 직접 비교 가능"],
                ["front_area / active-front", "moving front 위치와 폭이 맞는지 보는 핵심 지표"],
                ["mass trajectory", "감염 총량이 사라지거나 과도하게 흐려지는지 확인"],
            ],
            [1.55, 5.50],
        ),
    ),
    (
        "6. 한국 산림청 데이터와 문제 구성",
        [
            "산림청 계열은 compact GitHub-safe 데이터와 raw CSV 선택 경로를 모두 지원한다. compact 데이터는 2016-2023 감염목 좌표와 연도, manifest, province GeoJSON을 포함한다. raw CSV가 있으면 `조사일자`와 `방제완료여부`를 읽어 조치 전 월별 시간축을 재구성할 수 있다.",
            "공간 좌표는 raw bundle의 EPSG:5181 좌표를 EPSG:5179로 변환하고, 격자 계산에서는 x,y를 [0, 1]로 정규화한다. land mask는 province GeoJSON으로 만들며, observation density는 바다 셀에서 0으로 둔다. RK4는 land/sea 경계에서 masked no-flux diffusion을 쓰고, PINN은 land-only collocation과 sea exclusion penalty를 쓴다.",
            "조치 전 비교는 현재 기본 proxy로 누적 infected and completed records가 50,000을 넘는 첫 날짜를 대규모 조치 시작으로 본다. 제공 raw bundle에서는 이 날짜가 2016-10-02로 추정되며, 기본 비교는 2016-01부터 2016-09까지 월별 관측, RK4, PINN을 같은 elapsed-year grid에서 비교한다.",
        ],
        (
            ["데이터 축", "현재 처리"],
            [
                ["annual compact", "2016-2023 연도별 감염 위치를 GitHub에 포함"],
                ["pre-action month", "raw CSV가 있을 때 조치 시작 전 월별 확산만 비교"],
                ["land/sea", "관측, RK4, PINN 모두 바다 확산을 금지하는 제약 포함"],
            ],
            [1.55, 5.50],
        ),
    ),
    (
        "7. 산림청 PINN과 D/r 해석",
        [
            "산림청 PINN은 `fit_korea_pine_wilt_pinn`에서 동작한다. 모델은 기본 PINN과 같은 `OriginPINN` 계열을 쓰지만, seed-front feature와 travelling-wave exact feature는 끄고, 관측 격자와 land mask에 맞춘 data loss, initial condition loss, land-only PDE residual, sea penalty, weak boundary loss를 사용한다.",
            "계수는 normalized diffusion과 reaction을 학습한다. 물리 해석은 grid의 kilometer scale을 통해 D_phys = D_norm * L_km^2로 되돌린다. x/y 방향의 실제 지도 폭과 높이가 다르기 때문에 normalized PDE residual에서는 D_x = D_phys / width_km^2, D_y = D_phys / height_km^2인 anisotropic scaling을 사용한다.",
            "이 구조는 기존 산림청 zip의 D/r 추정 원리를 일반화한다. 기존 zip이 계수 산정과 forward PINN을 중심으로 구성되었다면, 현재 모델은 land mask, sea penalty, 공간 계수장, monthly action-time comparison, RK4/PINN 공동 시각화를 포함한다. 산림청 결과의 D/r은 관측 밀도, 방제 이력, 신고 체계, 벌목 효과가 결합된 effective spread coefficient로 정리된다.",
        ],
        (
            ["계수", "물리 해석"],
            [
                ["D_norm", "[0,1] 정규화 좌표에서 PINN이 학습하는 기본 diffusion"],
                ["D_phys", "선택한 L_km로 환산한 km^2/year diffusion"],
                ["r", "정규화 시간 단위가 year이므로 reaction_per_year로 해석"],
            ],
            [1.55, 5.50],
        ),
    ),
    (
        "8. 적용 범위와 참고문헌",
        [
            "현재 구현은 Fisher-KPP family를 중심으로 forward field, moving front, 계수 추정, land-constrained spread를 함께 다루는 기준 모델이다. 기본 synthetic 문제에서는 front-aware PINN이 moving front를 안정적으로 추적하고, 산림청 문제에서는 관측 과정과 방제 조치가 반영된 밀도장을 effective reaction-diffusion 계수로 정리한다.",
            "확장 방향은 multi-seed full run, 조치 시점 이후의 control term 명시화, 실제 월별 또는 일별 고해상도 관측 확보, front metric 기반 ablation의 통계 반복, D/r posterior와 uncertainty reporting이다. 현재 산출물은 validation figure, GIF, metrics JSON을 통해 field error, front error, mass trajectory, learned physics를 일관된 기준으로 확인한다.",
        ],
        None,
    ),
]

PINN_REFERENCES = [
    "Fisher, Ronald A. 1937. “The Wave of Advance of Advantageous Genes.” Annals of Eugenics 7 (4): 355–369.",
    "Kolmogorov, A. N., I. G. Petrovsky, and N. S. Piskunov. 1937. “A Study of the Equation of Diffusion with Increase in the Quantity of Matter, and Its Application to a Biological Problem.” Bulletin of Moscow University, Mathematics and Mechanics 1: 1–25.",
    "Ablowitz, Mark J., and Anthony Zeppetella. 1979. “Explicit Solutions of Fisher’s Equation for a Special Wave Speed.” Bulletin of Mathematical Biology 41: 835–840.",
    "Raissi, Maziar, Paris Perdikaris, and George Em Karniadakis. 2019. “Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations.” Journal of Computational Physics 378: 686–707.",
    "Wang, Sifan, Yujun Teng, and Paris Perdikaris. 2021. “Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks.” SIAM Journal on Scientific Computing 43 (5): A3055–A3081.",
    "Krishnapriyan, Aditi S., Amir Gholami, Shandian Zhe, Robert M. Kirby, and Michael W. Mahoney. 2021. “Characterizing Possible Failure Modes in Physics-Informed Neural Networks.” Advances in Neural Information Processing Systems 34.",
    "Jagtap, Ameya D., and George Em Karniadakis. 2020. “Extended Physics-Informed Neural Networks (XPINNs): A Generalized Space-Time Domain Decomposition Based Deep Learning Framework for Nonlinear Partial Differential Equations.” Communications in Computational Physics 28 (5): 2002–2041.",
    "Moseley, Ben, Andrew Markham, and Tarje Nissen-Meyer. 2023. “Finite Basis Physics-Informed Neural Networks (FBPINNs): A Scalable Domain Decomposition Approach for Solving Differential Equations.” Advances in Computational Mathematics 49: 62.",
]


RK4_PAGES: list[tuple[str, list[str], tuple[list[str], list[list[str]], list[float]] | None]] = [
    (
        "1. PDE와 비교 기준",
        [
            "RK4와 PDE 수치해석 계열은 u_t = D Laplacian(u) + r u(1-u)를 기준 방정식으로 사용한다. 기본 synthetic 문제와 산림청 문제는 같은 Fisher-KPP family로 정리되며, 기본 문제에서는 Gaussian seed 또는 Ablowitz-Zeppetella travelling wave를 사용하고, 산림청 문제에서는 gridded observation density를 초기장으로 사용한다.",
            "Ablowitz-Zeppetella 기준은 c = 5/sqrt(6)에서 명시적 travelling wave가 존재하므로, forward solver와 PINN이 front phase를 얼마나 따라가는지 확인하기 좋다. 1D 물리 좌표 x in [-20,20]은 단위 좌표로 변환되고, 따라서 코드상 diffusion은 D_norm = 1/40^2로 들어간다.",
            "공정 비교는 같은 PDE, 같은 초기조건, 같은 final time, 같은 공간 grid, 같은 metric을 사용한다. RK4, explicit Euler, backward Euler, trapezoidal은 시간 적분법만 다르게 두고 나머지 조건을 고정한다. same-problem RK4는 현재 PINN 문제 설정과 동일한 초기장, 계수, boundary condition, 평가 지표를 사용한다.",
        ],
        (
            ["비교 요소", "고정 기준"],
            [
                ["PDE", "동일한 Fisher-KPP D, r, boundary condition"],
                ["초기조건", "Gaussian seed, AZ exact profile, 또는 산림청 첫 관측 밀도"],
                ["평가지표", "final L2, max abs error, front area, active-front, mass"],
            ],
            [1.55, 5.50],
        ),
    ),
    (
        "2. RK4 구현과 안정성",
        [
            "RK4는 method-of-lines 방식이다. 먼저 공간 Laplacian을 finite difference로 이산화하고, resulting ODE system을 네 단계 Runge-Kutta로 적분한다. 기본 2D square 문제에서는 no-flux Neumann boundary를 edge padding으로 구현한다. AZ 2D 검증은 exact travelling-wave profile을 x 방향으로 반복하고 boundary에는 exact Dirichlet 값을 적용한다.",
            "명시적 RK4도 확산 CFL 제약을 피할 수 없다. 현재 안정성 점검은 dt_diff_limit = 0.69 dx^2 / (dim D)와 reaction scale 1/r 중 작은 값을 safety factor와 비교한다. 산림청 masked RK4도 같은 취지로 dt가 diffusion/reaction practical limit을 넘으면 에러를 내며, land/sea 경계에서는 masked no-flux flux를 사용한다.",
            "RK4는 정확한 초기장과 계수가 주어진 forward problem에서 낮은 격자 오차를 제공한다. PINN은 연속 함수와 계수, residual, 관측, front 제약을 동시에 최적화한다. 따라서 RK4는 reference solver로, PINN은 inverse와 sparse-data 상황까지 포괄하는 continuous surrogate로 배치된다.",
        ],
        (
            ["구현 함수", "역할"],
            [
                ["forward_fisher_kpp_rk4", "Gaussian seed 2D same-problem reference"],
                ["forward_ablowitz_zeppetella_rk4", "AZ travelling wave exact-boundary reference"],
                ["simulate_density_rk4_at_times", "산림청 월별/연도별 arbitrary output time reference"],
            ],
            [2.05, 5.00],
        ),
    ),
    (
        "3. 산림청 PDE/RK4와 조치 전 월별 축",
        [
            "산림청 RK4는 첫 관측 밀도를 초기조건으로 놓고, land mask 내부에서만 반응-확산을 진행한다. 바다 셀은 0으로 유지되며, diffusion은 land/sea 경계에서 바다로 흐르지 않는다. 물리 prior는 기본적으로 D = 15.5 km^2/year, r = 0.70 1/year이고, front speed는 2 sqrt(D r) = 6.5879 km/year로 계산된다.",
            "좌표 정규화 때문에 D는 두 단계로 해석된다. 물리 diffusion D_phys는 grid scale L_km에 의해 D_norm = D_phys / L_km^2로 바뀌고, 실제 PDE residual이나 RK4에서는 x/y extent 차이에 맞춰 D_x = D_phys / width_km^2, D_y = D_phys / height_km^2가 쓰인다. reaction은 시간 단위가 year이므로 r_per_year로 그대로 들어간다.",
            "조치 전 월별 모드는 raw CSV의 조사일자와 방제완료여부를 사용한다. 기본 proxy는 누적 infected and completed count가 50,000을 넘는 첫 날짜이고, 제공 raw bundle에서는 2016-10-02가 나온다. 그래서 기본 비교는 2016-01부터 2016-09까지의 observed density, RK4 density, PINN density를 같은 elapsed-year time grid에서 비교한다.",
        ],
        (
            ["값", "현재 기본값"],
            [
                ["D_phys", "15.5 km^2/year"],
                ["r_phys", "0.70 1/year"],
                ["action proxy", "cumulative infected-completed count > 50,000, default 2016-10-02"],
            ],
            [1.55, 5.50],
        ),
    ),
    (
        "4. 공정 비교와 참고문헌",
        [
            "RK4와 implicit/explicit 계열 수치해석법의 공정 비교는 accuracy, stability, solve cost를 함께 평가한다. explicit methods는 안정성 조건을 직접 반영하고 구현이 단순하다. backward Euler와 trapezoidal은 큰 dt에서도 안정적인 시간 적분을 제공하며 step마다 선형 또는 비선형 solve를 수행한다. RK4는 네 개의 stage를 사용해 같은 grid와 smooth solution에서 높은 정확도와 구현 균형을 제공한다.",
            "산림청 문제의 PDE/RK4 계열은 관측된 감염 밀도를 시작점으로 land-constrained Fisher-KPP spread를 계산한다. 방제, 신고 지연, 벌목, 기주 분포, 매개충 이동 효과는 effective D/r과 관측 밀도장에 반영되며, RK4 결과는 PINN과 같은 조건에서 비교되는 forward numerical reference로 사용된다.",
        ],
        None,
    ),
]

RK4_REFERENCES = [
    "Fisher, Ronald A. 1937. “The Wave of Advance of Advantageous Genes.” Annals of Eugenics 7 (4): 355–369.",
    "Kolmogorov, A. N., I. G. Petrovsky, and N. S. Piskunov. 1937. “A Study of the Equation of Diffusion with Increase in the Quantity of Matter, and Its Application to a Biological Problem.” Bulletin of Moscow University, Mathematics and Mechanics 1: 1–25.",
    "Ablowitz, Mark J., and Anthony Zeppetella. 1979. “Explicit Solutions of Fisher’s Equation for a Special Wave Speed.” Bulletin of Mathematical Biology 41: 835–840.",
    "Butcher, John C. 2016. Numerical Methods for Ordinary Differential Equations. 3rd ed. Chichester: Wiley.",
    "Morton, K. W., and D. F. Mayers. 2005. Numerical Solution of Partial Differential Equations: An Introduction. 2nd ed. Cambridge: Cambridge University Press.",
]


def _build_pages(document: Document, pages, references: list[str] | None = None) -> None:
    for idx, (heading, paragraphs, table) in enumerate(pages):
        _add_heading(document, heading)
        for paragraph in paragraphs:
            _add_para(document, paragraph)
        if table is not None:
            headers, rows, widths = table
            _add_table(document, headers, rows, widths)
        if references and idx == len(pages) - 1:
            _add_heading(document, "참고문헌", level=2)
            for reference in references:
                _add_reference(document, reference)
        if idx != len(pages) - 1:
            document.add_page_break()


def _audit_all_runs(document: Document) -> None:
    for paragraph in document.paragraphs:
        for run in paragraph.runs:
            _format_run(run, bold=bool(run.bold), italic=bool(run.italic))
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        _format_run(run, bold=bool(run.bold), italic=bool(run.italic))


def build_pinn_doc(path: Path) -> None:
    document = Document()
    _configure_document(document, "기본 PINN 및 산림청 PINN 구현 설명")
    _add_title(document, "기본 PINN 및 산림청 PINN 구현 설명", "Fisher-KPP 및 Korea pine-wilt PINN 구현 사양")
    _build_pages(document, PINN_PAGES, PINN_REFERENCES)
    _audit_all_runs(document)
    document.save(path)


def build_rk4_doc(path: Path) -> None:
    document = Document()
    _configure_document(document, "RK4 및 Fisher-KPP PDE 구현 설명")
    _add_title(document, "RK4 및 Fisher-KPP PDE 구현 설명", "Fisher-KPP PDE 수치해석 구현 사양")
    _build_pages(document, RK4_PAGES, RK4_REFERENCES)
    _audit_all_runs(document)
    document.save(path)


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    build_pinn_doc(DOCS_DIR / "fisher_kpp_pinn_and_korea_pinn_technical_note.docx")
    build_rk4_doc(DOCS_DIR / "fisher_kpp_rk4_pde_technical_note.docx")


if __name__ == "__main__":
    main()
