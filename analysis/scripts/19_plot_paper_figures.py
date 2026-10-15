#!/usr/bin/env python3
"""원고 그림을 저장된 수치에서 그린다.

- 입력: verification/overview_notebook_20261015.json, topics_lda_20261015.json
- 출력: analysis/output/figures/, paper/figures/의 그림 두 개
- 원고: 4.1 연도별 규모, 4.2.1 대상 언어별 토픽 분포
"""
from pathlib import Path
import json
import sys
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def apply_style():
    shared = Path.home() / 'weekly-planner/scripts'
    if (shared / 'figure_style.py').exists():
        sys.path.insert(0, str(shared))
        import figure_style
        figure_style.apply_style()
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ['NanumGothic', 'Noto Sans CJK KR', 'AppleGothic', 'Malgun Gothic']:
        if name in available:
            plt.rcParams['font.family'] = name
            break
    else:
        raise RuntimeError('국문 그림에는 NanumGothic 또는 Noto Sans CJK KR 글꼴이 필요합니다.')
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})


def plot_language_topics(table, sizes, output):
    apply_style()
    order = table.sort_values('n_all', ascending=False).index
    sub = table.loc[order]
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.16 * len(table) + 1.2), sharey=True)
    names = {'ko': '한국어', 'zh': '중국어', 'ja': '일본어'}
    for ax, lang in zip(axes, names):
        y = np.arange(len(table))
        for offset, field, color, label in [
            (-0.27, f'obs_{lang}', '#46B07B', '관측 편수'),
            (0, f'exp_{lang}', '#BDC3C7', '기대 편수: 전체 집합 기준'),
            (0.27, f'exp_lm_{lang}', '#FFD080', '기대 편수: 언어명 언급 기준'),
        ]:
            ax.barh(y + offset, sub[field], 0.27, color=color, label=label, zorder=3)
        xmax = max(sub[f'obs_{lang}'].max(), sub[f'exp_{lang}'].max(), sub[f'exp_lm_{lang}'].max())
        for cut in (7.5, 16.5):
            ax.axhline(cut, color='#7F8C8D', lw=0.7, ls=':')
        ax.set_yticks(y)
        ax.set_yticklabels([f'{i+1}. {sub.loc[t, "label"]}' for i, t in enumerate(order)], fontsize=7.5)
        ax.set_title(f'{names[lang]} ({sizes[lang]}편)')
        ax.set_xlabel('논문 수(편)')
        ax.set_xlim(0, xmax * 1.08)
        ax.grid(axis='y', visible=False)
    axes[0].invert_yaxis()
    axes[1].legend(loc='lower right', fontsize=8)
    fig.suptitle('대상 언어별 토픽 분포', y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_growth(overview, output):
    apply_style()
    keys = ['2023', '2024', '2025', '2026 (Jan-Sep)']
    rows = [overview['table1'][k] for k in keys]
    counts = [r['LLM-agent papers'] for r in rows]
    shares = [100*r['LLM-agent papers']/r['cs papers (5 categories)'] for r in rows]
    labels = ['2023', '2024', '2025', '2026\n1~9월']
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7))
    for ax, values, color, title, unit in zip(axes, [counts, shares], ['#46B07B', '#FF9E02'],
                                            ['LLM 에이전트 논문 수', '전체 수집 논문 대비 비율'], ['편', '%']):
        bars = ax.bar(labels, values, width=0.65, color=color, zorder=3)
        bars[-1].set_hatch('///')
        bars[-1].set_edgecolor('#4A4A4A')
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(unit, fontsize=9)
        ax.set_ylim(0, max(values)*1.25)
        ax.tick_params(labelsize=9)
        ax.grid(axis='y', alpha=0.25)
        for bar, v in zip(bars, values):
            label = f'{v:,}' if unit == '편' else f'{v:.1f}'
            ax.text(bar.get_x()+bar.get_width()/2, v+max(values)*0.025, label, ha='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=250, bbox_inches='tight')
    plt.close(fig)


def main():
    output = ROOT / 'analysis/output/figures'
    paper = ROOT / 'paper/figures'
    output.mkdir(parents=True, exist_ok=True)
    paper.mkdir(parents=True, exist_ok=True)
    overview = json.loads((ROOT/'verification/overview_notebook_20261015.json').read_text())
    topics = json.loads((ROOT/'verification/topics_lda_20261015.json').read_text())
    table = pd.DataFrame(topics['project_lang']).T
    sizes = {lang: int(table[f'obs_{lang}'].sum()) for lang in ['ko', 'zh', 'ja']}
    plot_growth(overview, output/'fig1_corpus_growth.png')
    plot_language_topics(table, sizes, output/'fig2b_lda_obs_vs_exp_lang.png')
    for name in ['fig1_corpus_growth.png', 'fig2b_lda_obs_vs_exp_lang.png']:
        shutil.copy2(output/name, paper/name)
        print(name)


if __name__ == '__main__':
    main()
