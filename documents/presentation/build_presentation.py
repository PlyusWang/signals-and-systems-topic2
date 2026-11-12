from __future__ import annotations

import json
import csv
import re
import zipfile
import argparse
import posixpath
from copy import deepcopy
from pathlib import Path

import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls
from pptx.util import Inches, Pt
from lxml import etree


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "documents/presentation" / "topic2_presentation.pptx"
W, H = 13.333, 7.5
INK = "1C292D"
MUTED = "53656A"
TEAL = "148A78"
MINT = "DCEFE9"
AMBER = "D8904C"
CORAL = "C65F4B"
PAPER = "F6F8F6"
WHITE = "FFFFFF"
PALE = "E8EFEC"

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
A14_NS = "http://schemas.microsoft.com/office/drawing/2010/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"


def _r(text: str, italic: bool = False):
    el = etree.Element(f"{{{M_NS}}}r")
    if italic:
        props = etree.SubElement(el, f"{{{M_NS}}}rPr")
        style = etree.SubElement(props, f"{{{M_NS}}}sty")
        style.set(f"{{{M_NS}}}val", "i")
    t = etree.SubElement(el, f"{{{M_NS}}}t")
    if text.startswith(" ") or text.endswith(" "):
        t.set(f"{{{XML_NS}}}space", "preserve")
    t.text = text
    return el


def _sub(base, sub):
    el = etree.Element(f"{{{M_NS}}}sSub")
    etree.SubElement(el, f"{{{M_NS}}}sSubPr")
    e = etree.SubElement(el, f"{{{M_NS}}}e")
    e.append(base)
    s = etree.SubElement(el, f"{{{M_NS}}}sub")
    s.append(sub)
    return el


def _sup(base, sup):
    el = etree.Element(f"{{{M_NS}}}sSup")
    etree.SubElement(el, f"{{{M_NS}}}sSupPr")
    e = etree.SubElement(el, f"{{{M_NS}}}e")
    e.append(base)
    s = etree.SubElement(el, f"{{{M_NS}}}sup")
    s.append(sup)
    return el


def _frac(num, den):
    el = etree.Element(f"{{{M_NS}}}f")
    props = etree.SubElement(el, f"{{{M_NS}}}fPr")
    typ = etree.SubElement(props, f"{{{M_NS}}}type")
    typ.set(f"{{{M_NS}}}val", "bar")
    n = etree.SubElement(el, f"{{{M_NS}}}num")
    n.append(num)
    d = etree.SubElement(el, f"{{{M_NS}}}den")
    d.append(den)
    return el


def _sqrt(expression):
    el = etree.Element(f"{{{M_NS}}}rad")
    props = etree.SubElement(el, f"{{{M_NS}}}radPr")
    hidden = etree.SubElement(props, f"{{{M_NS}}}degHide")
    hidden.set(f"{{{M_NS}}}val", "1")
    etree.SubElement(el, f"{{{M_NS}}}deg")
    content = etree.SubElement(el, f"{{{M_NS}}}e")
    content.append(expression)
    return el


def _nary(symbol: str, lower, upper, expression):
    el = etree.Element(f"{{{M_NS}}}nary")
    props = etree.SubElement(el, f"{{{M_NS}}}naryPr")
    char = etree.SubElement(props, f"{{{M_NS}}}chr")
    char.set(f"{{{M_NS}}}val", symbol)
    lim = etree.SubElement(props, f"{{{M_NS}}}limLoc")
    lim.set(f"{{{M_NS}}}val", "undOvr")
    if lower is None:
        hidden = etree.SubElement(props, f"{{{M_NS}}}subHide")
        hidden.set(f"{{{M_NS}}}val", "1")
    if upper is None:
        hidden = etree.SubElement(props, f"{{{M_NS}}}supHide")
        hidden.set(f"{{{M_NS}}}val", "1")
    sub = etree.SubElement(el, f"{{{M_NS}}}sub")
    if lower is not None:
        sub.append(lower)
    sup = etree.SubElement(el, f"{{{M_NS}}}sup")
    if upper is not None:
        sup.append(upper)
    content = etree.SubElement(el, f"{{{M_NS}}}e")
    content.append(expression)
    return el


def _group(*children):
    el = etree.Element(f"{{{M_NS}}}e")
    for child in children:
        if isinstance(child, str):
            el.append(_r(child))
        else:
            el.append(child)
    return el


def repair_snr_equation(shape):
    """Replace the SNR equation with a complete native log function and argument."""
    element = shape._element if hasattr(shape, "_element") else shape
    old = element.find(f".//{{{M_NS}}}oMath")
    if old is None:
        raise ValueError("SNR shape has no native Office equation")
    a_ns = "http://schemas.openxmlformats.org/drawingml/2006/main"
    font = old.find(f".//{{{a_ns}}}rPr")
    expression = etree.Element(f"{{{M_NS}}}oMath")
    expression.append(_r("SNR = 10 "))
    function = etree.SubElement(expression, f"{{{M_NS}}}func")
    etree.SubElement(function, f"{{{M_NS}}}funcPr")
    name = etree.SubElement(function, f"{{{M_NS}}}fName")
    name.append(_sub(_r("log"), _r("10")))
    argument = etree.SubElement(function, f"{{{M_NS}}}e")
    delimiter = etree.SubElement(argument, f"{{{M_NS}}}d")
    etree.SubElement(delimiter, f"{{{M_NS}}}dPr")
    content = etree.SubElement(delimiter, f"{{{M_NS}}}e")
    content.append(_frac(_sub(_r("E", True), _r("s", True)),
                         _sub(_r("E", True), _r("e", True))))
    expression.append(_r(" dB"))
    for run in expression.iter(f"{{{M_NS}}}r"):
        props = run.find(f"{{{M_NS}}}rPr")
        italic = props is not None
        if props is None:
            props = etree.Element(f"{{{M_NS}}}rPr")
            etree.SubElement(props, f"{{{M_NS}}}sty").set(f"{{{M_NS}}}val", "p")
            run.insert(0, props)
        if font is not None:
            drawing_font = deepcopy(font)
            drawing_font.set("lang", "en-US")
            drawing_font.set("i", "1" if italic else "0")
            run.insert(1, drawing_font)
    old.getparent().replace(old, expression)


def formula_tree(name: str):
    n, N, k = _r("n", True), _r("N", True), _r("k", True)
    if name == "fft":
        return _group(
            _sub(_r("X", True), k), _r(" = "),
            _nary("∑", _group(_r("n"), _r("="), _r("0")),
                  _group(_r("N"), _r("−"), _r("1")),
                  _group(_sub(_r("x", True), n), _sup(_r("e"),
                      _group(_r("−j2πkn/N"))))),
        )
    if name == "stft":
        return _group(_sub(_r("X", True), _group(_r("m, ω"))), _r(" = "),
            _nary("∑", _r("n"), None,
                            _group(_sub(_r("x", True), n), _r("w[n−mR]"),
                                    _sup(_r("e"), _r("−jωn")))))
    if name == "autocorr":
        return _group(_sub(_r("r", True), _group(_r("xx, ℓ"))), _r(" = "),
                      _nary("∑", _r("n"), None,
                            _group(_sub(_r("x", True), n), _sub(_r("x", True), _group(_r("n−ℓ"))))))
    if name == "crosscorr":
        return _group(_sub(_r("r", True), _group(_r("xy, ℓ"))), _r(" = "),
                      _nary("∑", _r("n"), None,
                            _group(_sub(_r("x", True), n), _sub(_r("y", True), _group(_r("n−ℓ"))))))
    if name == "cwt":
        return _group(_sub(_r("W", True), _group(_r("x; a,b"))), _r(" = "),
                      _frac(_r("1"), _sqrt(_r("|a|"))),
                      _nary("∫", None, None,
                            _group(_r("x(t)"), _sup(_r("ψ"), _r("*")),
                                   _r("((t−b)/a) dt"))))
    if name == "threshold":
        return _group(_r("λ = σ̂ √(2 ln N)"))
    if name == "params":
        return _group(_sub(_r("f", True), _r("0")), _r(" = 1.5 kHz    "),
                      _sub(_r("f", True), _r("1")), _r(" = 2.5 kHz    "),
                      _sub(_r("f", True), _r("s")), _r(" = 16 kHz    "),
                      _sub(_r("R", True), _r("b")), _r(" = 1 kbit/s"))
    if name == "fsk_phase":
        return _group(_r("φ[n] = φ[n−1] + 2π f_b[n] / f_s,    "),
                      _r("s[n] = A cos φ[n]"))
    if name == "average":
        return _group(_sub(_r("ŝ", True), n), _r(" = "), _frac(_r("1"), _r("M")),
                      _nary("∑", _r("i=1"), _r("M"), _sub(_r("r", True), _group(_r("i,n")))))
    if name == "snr":
        lower = _group(_r("n"), _r("="), _r("0"))
        upper = _group(_r("N"), _r("−"), _r("1"))
        numerator = _nary("∑", lower, upper, _sup(_r("s[n]"), _r("2")))
        denominator = _nary("∑", lower, upper, _sup(_r("s[n]−ŝ[n]"), _r("2")))
        return _group(_r("SNR = 10 log₁₀ "), _frac(numerator, denominator), _r(" dB"))
    if name == "ber":
        errors = _nary("∑", _r("i=1"), _r("N_b"), _r("1{bᵢ ≠ b̂ᵢ}"))
        return _group(_r("BER = "), _frac(errors, _r("N_b")))
    raise KeyError(name)


def add_math(slide, name: str, x: float, y: float, w: float, h: float, *, size: int = 22):
    formulas = {
        "fft": r"X_k=\sum_(n=0)^(N-1) x_n e^(-j2\pi kn/N)",
        "stft": r"X_(m,\omega)=\sum_n x_n w[n-mR]e^(-j\omega n)",
        "autocorr": r"r_(xx)[\ell]=\sum_n x_n x_(n-\ell)",
        "crosscorr": r"r_(xy)[\ell]=\sum_n x_n y_(n-\ell)",
        "cwt": r"W_x(a,b)=1/\sqrt(|a|) \int x(t)\psi^*((t-b)/a)dt",
        "threshold": "λ=σ\u0302√(2 ln N)",
        "params": r"f_0=1.5 kHz, f_1=2.5 kHz, f_s=16 kHz, R_b=1 kbit/s",
        "fsk_phase": r"\phi[n]=\phi[n-1]+2\pi f_(b[n])/f_s,\ s[n]=A cos(\phi[n])",
        "average": "s\u0302_n=(1/M) \\sum_(i=1)^M r_(i,n)",
        "snr": r"SNR=10 log_(10)(E_s/E_e) dB",
        "ber": r"BER=N_e/N_b",
    }
    if name not in formulas:
        raise KeyError(name)
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    shape.name = f"Math_{name}"
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = False
    tf.margin_left = tf.margin_right = Inches(0.02)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    paragraph = tf.paragraphs[0]
    paragraph.alignment = PP_ALIGN.CENTER
    run = paragraph.add_run()
    run.text = formulas[name]
    run.font.name = "Cambria Math"
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(INK)
    return shape


def _convert_equations_with_powerpoint(path: Path, equation_count: int) -> None:
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    app = win32com.client.DispatchEx("PowerPoint.Application")
    presentation = None
    try:
        app.Visible = True
        app.DisplayAlerts = 0
        presentation = app.Presentations.Open(str(path.resolve()), False, False, True)
        window = presentation.Windows.Item(1)
        window.WindowState = 2  # Work in the minimized presentation window.
        converted = 0
        for slide in presentation.Slides:
            for shape in slide.Shapes:
                if not str(shape.Name).startswith("Math_"):
                    continue
                text_range = shape.TextFrame.TextRange
                linear_text = str(text_range.Text)
                window.View.GotoSlide(int(slide.SlideIndex))
                text_range.Characters(1, int(text_range.Length)).Select()
                app.CommandBars.ExecuteMso("InsertBuildingBlocksEquationsGallery")
                shape.TextFrame.TextRange.Text = linear_text
                app.CommandBars.ExecuteMso("EquationProfessional")
                if shape.TextFrame2.TextRange.MathZones.Count < 1:
                    raise RuntimeError(f"PowerPoint did not create an editable equation: {shape.Name}")
                converted += 1
        if converted != equation_count:
            raise RuntimeError(f"Converted {converted} of {equation_count} native equations")
        presentation.Save()
        print(f"powerpoint_native_equations={converted}")
    finally:
        if presentation is not None:
            presentation.Close()
        if app.Presentations.Count == 0:
            app.Quit()
        pythoncom.CoUninitialize()


def set_background(slide, color=PAPER):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor.from_string(color)


def add_text(slide, text, x, y, w, h, *, size=18, color=INK, bold=False,
             align=PP_ALIGN.LEFT, font="Aptos", margin=0.02, valign=MSO_ANCHOR.MIDDLE):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    for i, line in enumerate(str(text).split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.space_after = Pt(4)
        for run in p.runs:
            run.font.name = font
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = RGBColor.from_string(color)
    return shape


def add_title(slide, title, eyebrow=None):
    add_text(slide, title, 0.62, 0.48, 12.1, 0.62, size=28, color=INK, bold=True)


def add_image(slide, relative, x, y, w=None, h=None):
    path = ROOT / relative
    if w is not None and h is not None:
        from PIL import Image
        with Image.open(path) as image:
            source_w, source_h = image.size
        scale = min(w / source_w, h / source_h)
        fitted_w, fitted_h = source_w * scale, source_h * scale
        slide.shapes.add_picture(str(path), Inches(x + (w - fitted_w) / 2),
                                 Inches(y + (h - fitted_h) / 2),
                                 width=Inches(fitted_w), height=Inches(fitted_h))
    elif w is not None:
        slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
    elif h is not None:
        slide.shapes.add_picture(str(path), Inches(x), Inches(y), height=Inches(h))


def add_rule(slide, x1, y1, x2, y2, color=TEAL, width=2):
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    line.line.color.rgb = RGBColor.from_string(color)
    line.line.width = Pt(width)
    return line


def add_tag(slide, text, x, y, w, color=TEAL):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(0.38))
    shape._element.spPr.append(parse_xml(f'<a:effectLst {nsdecls("a")}/>'))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string(MINT if color == TEAL else "F7EBDD")
    shape.line.fill.background()
    tf = shape.text_frame
    tf.clear()
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = PP_ALIGN.CENTER
    for r in p.runs:
        r.font.name = "Aptos"
        r.font.size = Pt(14)
        r.font.bold = True
        r.font.color.rgb = RGBColor.from_string(color)
    return shape


def add_metric(slide, label, value, x, y, w=2.1, value_color=TEAL):
    add_text(slide, value, x, y, w, 0.58, size=28, bold=True, color=value_color, align=PP_ALIGN.CENTER)
    add_text(slide, label, x, y + 0.58, w, 0.6, size=15, color=MUTED, align=PP_ALIGN.CENTER)


def add_node(slide, text, x, y, w, h, fill=MINT, line=TEAL, size=17):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape._element.spPr.append(parse_xml(f'<a:effectLst {nsdecls("a")}/>'))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string(fill)
    shape.line.color.rgb = RGBColor.from_string(line)
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = PP_ALIGN.CENTER
    for r in p.runs:
        r.font.name = "Microsoft YaHei UI"
        r.font.size = Pt(size)
        r.font.bold = True
        r.font.color.rgb = RGBColor.from_string(INK)
    return shape


def add_cell_text(cell, text, *, size=15, color=INK, bold=False, align=PP_ALIGN.LEFT):
    cell.text = str(text)
    cell.margin_left = cell.margin_right = Inches(0.10)
    cell.margin_top = cell.margin_bottom = Inches(0.06)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    for p in cell.text_frame.paragraphs:
        p.alignment = align
        for run in p.runs:
            run.font.name = "Aptos"
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = RGBColor.from_string(color)


def _add_math_namespace_decls(prs):
    # python-pptx serializes extension namespace prefixes from this map.
    from pptx.oxml.ns import _nsmap
    _nsmap.setdefault("a14", A14_NS)
    _nsmap.setdefault("m", M_NS)


def _slide(prs, title=None, eyebrow=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_background(slide)
    if title:
        add_title(slide, title, eyebrow)
    return slide


def build():
    # Reuse the equations already authored and validated by PowerPoint.
    native_equations = {}
    if OUT.exists():
        with zipfile.ZipFile(OUT) as source:
            for member in source.namelist():
                if not re.fullmatch(r"ppt/slides/slide\d+\.xml", member):
                    continue
                root = etree.fromstring(source.read(member))
                namespaces = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
                              "m": M_NS}
                for element in root.xpath(".//p:sp[.//m:oMath]", namespaces=namespaces):
                    name = element.find("p:nvSpPr/p:cNvPr", namespaces).get("name")
                    native_equations[name] = deepcopy(element)
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    _add_math_namespace_decls(prs)
    prs.core_properties.title = "信号与系统 Topic 2：复杂噪声通信信号恢复"
    prs.core_properties.subject = "时频、小波、相关与重复接收实验"
    prs.core_properties.author = ""

    evaluation = json.loads((ROOT / "task2/experiments/evaluation/seed_sweep_summary.json").read_text(encoding="utf-8"))
    repeat_rows = list(csv.DictReader((ROOT / "task2/experiments/repeat_average/repeat_average_metrics.csv").open(encoding="utf-8-sig")))
    demo = json.loads((ROOT / "task2/experiments/representative/message_demo.json").read_text(encoding="utf-8"))

    # 1. Research question and real analyzer capture.
    s = _slide(prs)
    add_text(s, "复杂噪声中的\n声学通信信号恢复", 0.75, 0.72, 5.6, 1.65, size=34, bold=True)
    add_text(s, "时频分析与 2-FSK 接收实验", 0.78, 2.54, 5.4, 0.55, size=20, color=MUTED)
    add_text(s, "成员 1\n成员 2\n成员 3\n成员 4",
             0.8, 3.55, 5.1, 1.8, size=17, color=INK)
    add_image(s, "task2/figures/noise_waveforms_presentation.png", 6.35, 1.0, 6.5, 5.7)

    # 2. Working analyzer.
    s = _slide(prs, "同一选区，切换不同表示", "SignalScope")
    add_image(s, "task1/screenshots/gui_zh_fft.png", 0.72, 1.48, 8.5, 4.92)
    add_text(s, "16 kHz\n\n1.856 s\n\n可拖动选择\n\n完整 / 选区 FFT", 9.55, 1.83, 2.9, 3.5, size=18, color=INK, bold=True)
    add_text(s, "选区联动 STFT、CWT、自相关和互相关。\n\n后台计算与波形包络抽稀保持界面响应。", 9.55, 5.35, 2.9, 1.2, size=15, color=MUTED)

    # 3. Fourier and time-frequency.
    s = _slide(prs, "窗口长度决定相近频率能否分开", "FFT · STFT · CWT")
    add_image(s, "task2/figures/window_resolution_presentation.png", 0.65, 1.35, 8.15, 5.8)
    add_text(s, "双音：1.00 kHz 与 1.08 kHz\n\nHann 主瓣间距约为 4/T\n\n零填充细化频点采样，\n不改变主瓣宽度。\n\n短 STFT 窗利于定位时间，\n长窗利于区分相近频率。",
             9.05, 1.85, 3.55, 4.55, size=19, color=INK)

    # 4. Editable Fourier/CWT equations.
    s = _slide(prs, "频域与时频表示", "傅里叶与时频分析")
    add_text(s, "离散 Fourier 变换", 0.9, 1.38, 3.4, 0.45, size=18, bold=True)
    add_math(s, "fft", 0.75, 1.82, 11.8, 0.82, size=22)
    add_text(s, "短时 Fourier 变换", 0.9, 2.93, 3.4, 0.45, size=18, bold=True)
    add_math(s, "stft", 0.75, 3.34, 11.8, 0.82, size=22)
    add_text(s, "连续小波变换", 0.9, 4.53, 3.4, 0.45, size=18, bold=True)
    add_math(s, "cwt", 0.75, 4.93, 11.8, 0.82, size=22)
    add_text(s, "三者使用相同的信号与时间选区，强调的结构不同。", 0.9, 6.3, 11.2, 0.45, size=17, color=MUTED)

    # 5. Correlation and synchronization.
    s = _slide(prs, "相关把重复结构转化为检测统计量", "相关分析")
    add_image(s, "task2/figures/correlation_sync_presentation.png", 0.72, 1.55, 6.35, 4.6)
    add_text(s, "自相关", 7.25, 1.5, 2.0, 0.4, size=18, bold=True)
    add_math(s, "autocorr", 6.95, 1.94, 5.6, 0.75, size=20)
    add_text(s, "互相关", 7.25, 3.0, 2.0, 0.4, size=18, bold=True)
    add_math(s, "crosscorr", 6.95, 3.42, 5.6, 0.75, size=20)
    add_text(s, "正滞后表示第一个输入相对参考模板延后。\n相关峰用于帧起点估计和对齐。", 7.15, 4.45, 5.2, 1.0, size=16, color=MUTED)

    # 6. FSK system flow.
    s = _slide(prs, "2-FSK 帧与接收链路", "声学通信")
    add_node(s, "随机载荷\n单帧 192 byte\n重复 96 byte", 0.7, 1.55, 2.0, 1.02, size=15)
    add_node(s, "CRC-32\n+ 32 bit 前导码", 3.0, 1.55, 2.15, 1.02, fill="F7EBDD", line=AMBER)
    add_node(s, "相位连续\n2-FSK 调制", 5.55, 1.55, 2.1, 1.02)
    add_node(s, "复杂噪声\nN1–N4 组合", 8.05, 1.55, 2.1, 1.02, fill="F7EBDD", line=AMBER)
    add_node(s, "互相关同步\n能量判决", 10.55, 1.55, 2.1, 1.02)
    for x in (2.72, 5.2, 7.72, 10.22):
        add_rule(s, x, 2.06, x + 0.22, 2.06, color=AMBER, width=2)
    add_math(s, "fsk_phase", 1.1, 3.2, 11.0, 0.78, size=20)
    add_math(s, "params", 1.2, 4.15, 10.9, 0.65, size=20)
    add_text(s, "接收端只使用公开前导码与载波参数；有效载荷仅参与 BER / CRC 评分。", 1.2, 5.2, 10.9, 0.7,
             size=17, color=MUTED, align=PP_ALIGN.CENTER)

    # 7. Noise families and observed map.
    s = _slide(prs, "噪声类型决定合适的分析尺度", "噪声模型")
    add_image(s, "task2/figures/time_frequency_clean_noisy_presentation.png", 0.65, 1.45, 7.2, 4.85)
    rows = [("N1", "持续 AWGN 底噪"), ("N2", "1777 Hz 窄带干扰"),
            ("N3", "局部脉冲簇"), ("N4", "变强度有色噪声 + 扫频")]
    for i, (tag, desc) in enumerate(rows):
        add_tag(s, tag, 8.3, 1.68 + i * 0.88, 0.85, TEAL if i % 2 == 0 else AMBER)
        add_text(s, desc, 9.35, 1.65 + i * 0.88, 3.1, 0.48, size=17)
    add_text(s, "各方法使用相同的含噪输入。", 8.3, 5.5, 4.2, 0.7, size=17, color=MUTED)

    # 8. Wavelet route.
    s = _slide(prs, "小波阈值对通信信号的影响", "小波去噪")
    add_image(s, "task2/figures/method_snr_N1_N2_presentation.png", 0.75, 1.62, 6.25, 4.35)
    add_math(s, "threshold", 7.15, 1.92, 5.0, 0.85, size=23)
    add_text(s, "DWT 分解后，以最细尺度细节系数的 MAD 估计噪声标准差，再对各层细节系数进行软阈值收缩并逆变换。", 7.15, 3.05, 5.1, 1.3, size=17)
    sync_diagnostic = next(r for r in evaluation if r["case"] == "N1_N2" and r["target_snr_db"] == 0 and r["method"] == "wavelet_soft")
    add_text(s, f"N1+N2、0 dB：小波输出 SNR {sync_diagnostic['output_snr_mean_db']:.2f} dB。\n相关同步 BER {sync_diagnostic['ber_mean']:.3f}；真实帧起点诊断 BER {sync_diagnostic['ber_known_start_mean']:.3f}。\n\n给定正确起点后载荷误码仍高，表明阈值处理同时影响同步与能量判决。",
             7.15, 4.48, 5.25, 2.05, size=15, color=CORAL, bold=True)

    # 9. Correlation-assisted Wiener.
    s = _slide(prs, "噪声功率估计用于短时频谱增强", "PSD-Wiener")
    add_node(s, "噪声专用观测段", 0.75, 1.52, 2.3, 0.68)
    add_node(s, "噪声 STFT 功率估计", 3.55, 1.52, 2.55, 0.68)
    add_node(s, "短时频谱增益", 6.45, 1.52, 2.35, 0.68, fill="F7EBDD", line=AMBER)
    add_node(s, "逆 STFT 重构", 9.35, 1.52, 2.25, 0.68)
    for x in (3.1, 6.0, 8.9):
        add_rule(s, x, 1.86, x + 0.35, 1.86, color=AMBER, width=2)
    psd_result = next(r for r in evaluation if r["case"] == "N1_N2" and r["target_snr_db"] == 0 and r["method"] == "psd_wiener")
    notch_result = next(r for r in evaluation if r["case"] == "N1_N2" and r["target_snr_db"] == 0 and r["method"] == "notch_bandpass")
    auto_result = next(r for r in evaluation if r["case"] == "N1_N2" and r["target_snr_db"] == 0 and r["method"] == "notch_bandpass_auto")
    add_text(s, "噪声观测", 0.95, 3.0, 3.2, 0.5, size=22, bold=True)
    add_text(s, "接收帧前 0.256 秒的噪声段，\n与含噪帧采用相同的窗和帧移。\n跨帧平均 STFT 幅度平方，\n估计各频率的噪声功率。", 0.95, 3.72, 5.1, 2.25, size=21)
    add_text(s, "增益与重构", 7.05, 3.0, 4.4, 0.5, size=22, bold=True)
    add_text(s, "按估计噪声功率收缩短时频谱，\n保留增益下限，再逆 STFT 重构。\n\n窄带干扰另由噪声谱峰估频，\n设置陷波中心频率。", 7.05, 3.72, 5.5, 2.45, size=21)

    # 10. Independent-seed SNR and BER.
    s = _slide(prs, "SNR 与 BER 给出不同评价结论", "独立测试")
    add_image(s, "task2/figures/snr_sweep_N1_N2_presentation.png", 0.65, 1.45, 6.15, 4.35)
    add_image(s, "task2/figures/wavelet_sync_comparison_presentation.png", 6.9, 1.45, 5.8, 4.35)
    add_math(s, "snr", 0.8, 5.9, 5.9, 0.65, size=21)
    add_math(s, "ber", 7.0, 5.9, 5.3, 0.65, size=21)
    add_text(s, "N1+N2；5 个测试种子。右图比较相关同步与真实起点下的小波 BER。",
             0.85, 6.75, 11.65, 0.4, size=16, color=MUTED)

    # 11. Multi-frame averaging.
    s = _slide(prs, "互相关对齐后，独立噪声随帧数下降", "重复接收")
    add_image(s, "task2/figures/repeat_noise_structure_presentation.png", 0.75, 1.55, 7.1, 4.55)
    add_math(s, "average", 7.9, 1.9, 4.55, 0.78, size=22)
    indep_16 = next(float(r["output_snr_db"]) for r in repeat_rows
                    if r["noise_structure"] == "independent" and r["M"] == "16" and
                    r["method"] == "correlation_aligned_average")
    shared_16 = next(float(r["output_snr_db"]) for r in repeat_rows
                     if r["noise_structure"] == "shared_tone" and r["M"] == "16" and
                     r["method"] == "correlation_aligned_average")
    add_metric(s, "独立噪声 · M=16", f"{indep_16:.2f} dB", 8.0, 3.18, w=2.1)
    add_metric(s, "共享窄带干扰 · M=16", f"{shared_16:.2f} dB", 10.35, 3.18, w=2.4, value_color=AMBER)
    indep_single = next(float(r["single_frame_input_snr_db"]) for r in repeat_rows
                        if r["noise_structure"] == "independent" and r["M"] == "16" and
                        r["method"] == "correlation_aligned_average")
    indep_gain = indep_16 - indep_single
    add_text(s, f"独立噪声单帧基准：{indep_single:.2f} dB。\nM=16 对齐平均增益：{indep_gain:.2f} dB。",
             8.0, 5.0, 4.55, 1.12, size=15, color=MUTED)

    # 12. Decoded message.
    s = _slide(prs, "恢复消息来自真实接收与 CRC 检查", "消息检测")
    add_image(s, "task1/screenshots/gui_zh_stft.png", 0.75, 1.55, 6.0, 3.4)
    add_image(s, "task2/figures/correlation_sync_compact_presentation.png", 6.85, 1.55, 5.7, 3.2)
    add_text(s, f"恢复内容前缀：{demo['decoded_ascii'][:18]}", 0.9, 5.12, 11.7, 0.65, size=17, color=TEAL, bold=True)
    add_tag(s, f"CRC {'PASS' if demo['crc_ok'] else 'FAIL'}", 0.95, 6.08, 1.75,
            TEAL if demo["crc_ok"] else CORAL)
    add_text(s, f"N1+N2   |   实测输入 SNR {demo['input_snr_db']:.2f} dB   |   BER {demo['ber']:.4f}   |   同步偏移 {demo['sync_start']} samples",
             2.95, 6.06, 9.55, 0.42, size=15, color=MUTED)

    # 13. Findings and limitations from measurements.
    s = _slide(prs, "实验发现", "实验结论")
    wavelet_row = next(r for r in evaluation if r["case"] == "N1_N2" and r["target_snr_db"] == 0 and r["method"] == "wavelet_soft")
    add_text(s, f"N1+N2、0 dB：自动估频陷波达到 {auto_result['output_snr_mean_db']:.2f} dB 输出 SNR，\n与已知频率参考一致；PSD-Wiener 为 {psd_result['output_snr_mean_db']:.2f} dB。",
             0.95, 1.65, 11.4, 1.15, size=24)
    add_text(s, f"小波软阈值输出 SNR 为 {wavelet_row['output_snr_mean_db']:.2f} dB，\n真实起点诊断 BER 仍为 {wavelet_row['ber_known_start_mean']:.3f}，载荷判决也受到影响。",
             0.95, 3.35, 11.4, 1.15, size=24)
    add_text(s, f"16 次相关对齐平均使独立噪声条件下的 SNR 提高 {indep_gain:.2f} dB；\n共同窄带干扰仍保留在平均信号中。",
             0.95, 5.05, 11.4, 1.15, size=24)

    # 14. References supporting the analysis and receiver methods.
    s = _slide(prs, "参考文献", "")
    references = [
        ("[1] Oppenheim, Willsky & Nawab. Signals and Systems, 2nd ed. Prentice Hall, 1997.", 0.58),
        ("[2] Donoho & Johnstone. Ideal spatial adaptation by wavelet shrinkage. Biometrika, 81(3): 425–455, 1994.", 0.74),
        ("[3] Boll. Suppression of acoustic noise in speech using spectral subtraction. IEEE TASSP, 27(2): 113–120, 1979.", 0.74),
        ("[4] Welch. The use of fast Fourier transform for the estimation of power spectra. IEEE Trans. Audio Electroacoust., 15(2): 70–73, 1967.", 0.88),
        ("[5] Ephraim & Malah. Speech enhancement using a minimum-mean square error short-time spectral amplitude estimator. IEEE TASSP, 32(6): 1109–1121, 1984.", 0.88),
        ("[6] Turin. An introduction to matched filters. IEEE Trans. Inf. Theory, 6(3): 311–329, 1960.", 0.74),
    ]
    y = 1.35
    for reference, height in references:
        add_text(s, reference, 0.85, y, 11.65, height, size=18, color=INK, valign=MSO_ANCHOR.TOP)
        y += height + 0.16

    OUT.parent.mkdir(parents=True, exist_ok=True)
    equation_count = sum(
        str(shape.name).startswith("Math_")
        for slide in prs.slides for shape in slide.shapes
    )
    requested = {shape.name for slide in prs.slides for shape in slide.shapes
                 if shape.name.startswith("Math_")}
    if requested <= native_equations.keys():
        for slide in prs.slides:
            for shape in list(slide.shapes):
                if shape.name not in requested:
                    continue
                equation = deepcopy(native_equations[shape.name])
                equation.find(".//{http://schemas.openxmlformats.org/presentationml/2006/main}cNvPr").set("id", str(shape.shape_id))
                transform = equation.find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}xfrm")
                transform.find("{http://schemas.openxmlformats.org/drawingml/2006/main}off").attrib.update({"x": str(shape.left), "y": str(shape.top)})
                transform.find("{http://schemas.openxmlformats.org/drawingml/2006/main}ext").attrib.update({"cx": str(shape.width), "cy": str(shape.height)})
                shape._element.getparent().replace(shape._element, equation)
                if shape.name == "Math_snr":
                    repair_snr_equation(equation)
        prs.save(OUT)
        print(f"preserved_native_equations={equation_count}")
    else:
        prs.save(OUT)
        _convert_equations_with_powerpoint(OUT, equation_count)
    refresh_analysis_figures(OUT)
    print(OUT)


def check_presentation(path: Path):
    with zipfile.ZipFile(path) as archive:
        slide_names = sorted((n for n in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                             key=lambda name: int(re.search(r"slide(\d+)", name).group(1)))
        equation_count = 0
        equation_text = []
        namespace = {"a14": A14_NS, "m": M_NS, "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
                     "p": "http://schemas.openxmlformats.org/presentationml/2006/main"}
        for name in slide_names:
            root = etree.fromstring(archive.read(name))
            eqs = root.xpath(".//a14:m[m:oMathPara/m:oMath]", namespaces=namespace)
            equation_count += len(eqs)
            for eq in eqs:
                assert eq.xpath("./m:oMathPara/m:oMath", namespaces=namespace)
            for shape in root.xpath('.//p:sp[p:nvSpPr/p:cNvPr[@name="Math_snr"]][.//m:oMath]', namespaces=namespace):
                function = shape.find(f".//{{{M_NS}}}func")
                assert function is not None
                assert "".join(function.find(f"{{{M_NS}}}fName").itertext()) == "log10"
                assert function.find(f"{{{M_NS}}}e/{{{M_NS}}}d/{{{M_NS}}}e/{{{M_NS}}}f") is not None
                assert not shape.xpath('.//m:e[not(*)]', namespaces=namespace)
        assert len(slide_names) == 14, len(slide_names)
        assert equation_count == 11, equation_count
        print(f"slides={len(slide_names)} native_omml_equations={equation_count}")


def _refresh_images(path: Path, image_names, *, repair_snr=False) -> None:
    """Refresh selected media and, when requested, repair the native SNR equation."""
    namespaces = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
                  "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
                  "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
                  "m": M_NS}
    screenshots = {Path(name).name: (ROOT / name).read_bytes() for name in image_names}
    temporary = path.with_name(path.stem + ".refresh.pptx")
    try:
        with zipfile.ZipFile(path) as source:
            replacements = {}
            for member in source.namelist():
                if not re.fullmatch(r"ppt/slides/slide\d+\.xml", member):
                    continue
                root = etree.fromstring(source.read(member))
                rel_file = posixpath.join(posixpath.dirname(member), "_rels",
                                         posixpath.basename(member) + ".rels")
                relationships = {element.get("Id"): element.get("Target")
                                 for element in etree.fromstring(source.read(rel_file))}
                for picture in root.findall(".//p:pic", namespaces):
                    properties = picture.find("p:nvPicPr/p:cNvPr", namespaces)
                    if properties is None or Path(properties.get("descr", "")).name not in screenshots:
                        continue
                    blip = picture.find(".//a:blip", namespaces)
                    target = relationships[blip.get("{" + namespaces["r"] + "}embed")]
                    media_path = posixpath.normpath(posixpath.join(posixpath.dirname(member), target))
                    replacements[media_path] = screenshots[Path(properties.get("descr")).name]
                if repair_snr:
                    for shape in root.xpath('.//p:sp[p:nvSpPr/p:cNvPr[@name="Math_snr"]][.//m:oMath]', namespaces=namespaces):
                        repair_snr_equation(shape)
                        replacements[member] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
            if not replacements:
                raise RuntimeError("No embedded GUI screenshots were found")
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as destination:
                for member in source.infolist():
                    destination.writestr(member, replacements.get(member.filename,
                                                                  source.read(member.filename)))
        check_presentation(temporary)
        temporary.replace(path)
        print(f"refreshed_presentation_parts={len(replacements)}")
    finally:
        temporary.unlink(missing_ok=True)


def refresh_gui_images(path: Path = OUT) -> None:
    _refresh_images(path, ("task1/screenshots/gui_zh_fft.png", "task1/screenshots/gui_zh_stft.png", "task1/screenshots/gui_zh_cwt.png", "task1/screenshots/gui_zh_corr.png"))


def refresh_analysis_figures(path: Path = OUT) -> None:
    _refresh_images(path, ("task2/figures/time_frequency_clean_noisy_presentation.png",
                          "task2/figures/correlation_sync_presentation.png",
                          "task2/figures/correlation_sync_compact_presentation.png"), repair_snr=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-gui", action="store_true")
    parser.add_argument("--refresh-analysis", action="store_true")
    args = parser.parse_args()
    if args.refresh_analysis:
        refresh_analysis_figures()
    elif args.refresh_gui:
        refresh_gui_images()
    else:
        build()
