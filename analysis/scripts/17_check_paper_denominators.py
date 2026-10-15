#!/usr/bin/env python3
"""현재 원고의 분모·ID·토픽 배정·기대 편수를 정본과 대조한다. 재학습 및 파일 쓰기는 하지 않는다."""
from pathlib import Path
import json,csv,collections,re,ast,hashlib,math
R=Path(__file__).resolve().parents[2]
def j(n):return json.loads((R/'verification'/f'{n}_20261015.json').read_text())
def rows(p):
 with (R/p).open() as f:return list(csv.DictReader(f))
def flag(v):return str(v).lower()=='true'
checks=[]
def check(ok,msg):
 assert ok,msg
 checks.append(msg)
c=j('corpus_counts');o=j('overview_notebook');t=j('topics_lda');h=j('hclt_notebook');a=j('agent_colloc');k=j('meta_keyness');m=j('meta_topic_covariates')
allc=rows('preprocessing/output/corpus_llm_agent_all_2023_2026q3.csv'); ids={x['id'] for x in allc}; index={x['id']:x for x in allc}; N=len(allc)
docs=rows('analysis/output/tables/lda_doc_topics.csv'); D={x['id']:x for x in docs}
check(len(ids)==N==20107==t['n_docs']==c['pipeline']['llm_agent_all'],'current corpus and LDA: 20107 unique documents')
check(len(docs)==len(D)==N and set(D)==ids,'LDA assignment IDs exactly equal current corpus IDs')
years=collections.Counter(int(x['year']) for x in allc)
check(dict(years)=={int(y):v for y,v in c['llm_agent_by_year'].items()},'year denominators match current corpus counts')
check(all(x['year']==x['year_id']==str(2000+int(x['id'][:2])) and x['month']==str(int(x['id'][2:4])) and (int(x['year'])<2026 or int(x['month'])<=9) for x in allc),'period and year basis use first-submission ID, through 2026-09')
check(all(D[x['id']]['year']==x['year'] and D[x['id']]['month']==x['month'] for x in allc),'LDA document years and months match current corpus')
lang={l:{x['id'] for x in allc if flag(x['lang_'+l])} for l in ['ko','zh','ja']}; sizes={l:len(v) for l,v in lang.items()}
check(sizes=={'ko':25,'zh':186,'ja':17}==c['cjk_lang_by_lang_total'],'current language denominators: 25 / 186 / 17')
check(len(set.union(*lang.values()))==226==c['pipeline']['llm_agent_cjk_lang'],'target-language union: 226; language memberships: 228')
check(sum(flag(x['cjk_signal']) for x in allc)==257,'language signals: 257, distinct from selected target-language union')
source=(R/'analysis/scripts/02_make_notebooks.py').read_text()
other=ast.literal_eval(re.search(r'OTHER_LANGS = (\[.*?\])',source).group(1))
lmre=re.compile(r'\b('+'|'.join(other)+r')\b|multi-?lingual|cross-?lingual')
lm={x['id'] for x in allc if not flag(x['cjk_signal']) and lmre.search((x['title']+' '+x['abstract']).lower())}
check(len(lm)==314==t['lang_mention_baseline']['n'],'second baseline independently selected from current titles/abstracts: 314')
topicids={str(z):{x['id'] for x in docs if int(x['lda_topic'])==z} for z in range(25)}
for z,sub in topicids.items():
 w=t['world_topic_table'][z];p=t['project_lang'][z]
 check(len(sub)==w['world_total']==p['n_all']==t['topics'][z]['n_docs'],f'topic {z}: world counts agree')
 check(len(sub&lm)==p['n_lang_mention'],f'topic {z}: second baseline topic count agrees')
 for y,ny in years.items():
  nytopic=sum(index[i]['year']==str(y) for i in sub)
  check(nytopic==w[f'world_n_{y}'],f'topic {z}: year {y} count agrees')
 for l,n in sizes.items():
  obs=len(sub&lang[l]); check(obs==w[f'{l}_obs']==p[f'obs_{l}']==m['year_adjusted'][z][f'{l}_obs'],f'topic {z}: {l} observations agree across latest analyses')
  check(round(len(sub)/N*n,1)==w[f'{l}_exp'] and round(len(sub&lm)/len(lm)*n,1)==w[f'{l}_exp_lm'],f'topic {z}: {l} both expected counts use current denominators')
  expadj=sum(sum(index[i]['year']==str(y) for i in sub)/ny*sum(index[i]['year']==str(y) for i in lang[l]) for y,ny in years.items())
  check(abs(round(expadj,2)-m['year_adjusted'][z][f'{l}_exp_adj'])<.011,f'topic {z}: {l} year-adjusted expected count agrees')
for row in m['logit']:
 check(row['obs']==len(topicids[str(row['topic'])]&lang[row['lang']]),'logistic regression observed counts use current language labels')
check(sum(math.isfinite(x['p']) for x in m['logit'])==47,'multiple-comparison count: 47 finite p values')
check(h['family_by_set']['n']=={'arXiv world':N,'Korean':25,'Chinese':186,'Japanese':17,'HCLT 2023-26':62},'keyword-family denominator metadata matches current corpus')
for w,rr in h['family_by_set']['rows'].items():
 for name,n in [('Korean',25),('Chinese',186),('Japanese',17),('HCLT 2023-26',62)]:
  check(abs(rr[name]-100*rr['n '+name]/n)<=.051,f'keyword family {w}: {name} percentage uses stated denominator')
check(a['set_sizes']=={'world':N,**sizes} and k['set_sizes']==sizes,'collocation/keyness language denominators agree')
check(k['n_2026']==years[2026] and k['n_prior']==sum(n for y,n in years.items() if y<2026),'keyness period denominators: 10887 and 9220')
for cache in ['lemmas','agent_colloc']:
 records=[json.loads(line) for line in (R/f'analysis/output/cache/{cache}_2023_2026q3.jsonl').open()]
 check(len(records)==N and {x['id'] for x in records}==ids,f'{cache} cache IDs equal current corpus IDs')
 if cache=='lemmas':
  for word in ['execution','memory','skill']:
   row=next(x for x in k['y2026_top'] if x['word']==word)
   for after,field,denom in [(True,'2026',k['n_2026']),(False,'prior',k['n_prior'])]:
    cnt=sum(word in x['t'] and ((index[x['id']]['year']=='2026')==after) for x in records)
    check(cnt==row['df_'+field] and abs(round(100*cnt/denom,2)-row['pct_docs_'+field])<.011,f'keyness {word}, {field}: document counts and percentages agree')
 else:
  for slot,word in [('before','coding'),('after','memory'),('after','workflow'),('after','benchmark')]:
   row=a['world'][slot][word]
   found={x['id'] for x in records if word in (set(x['prefix'])|set(x['left1']) if slot=='before' else set(x['right1']))}
   check(len(found)==row['n'],'collocation counts agree: '+word)
   for y in [2023,2026]:
    pct=100*sum(index[i]['year']==str(y) for i in found)/years[y]
    check(abs(round(pct,2)-row[f'pct_{y}'])<.011,f'collocation {word}: {y} denominator agrees')
# Check all table-2 rows against current JSON.
tex=(R/'paper/tables/tab_lda_topics.tex').read_text()
for line in tex.splitlines():
 line=re.sub(r'\\textbf\{([^{}]*)\}',r'\1',line)  # 표의 강조는 수치와 분리한다.
 if not re.match(r'^\d+ &',line):continue
 parts=line.split(' & ');assert len(parts)==7, 'table 2 must contain seven columns';rank=int(parts[0]);w=next(x for x in t['world_topic_table'].values() if x['world_rank']==rank)
 check(int(parts[2].split(' (')[0].replace(',',''))==w['world_total'],f'table 2 rank {rank}: world count agrees')
 for part,l in zip(parts[4:7],['ko','zh','ja']):
  vals=re.search(r'(\d+) \(([\d.]+); ([\d.]+)\)',part).groups()
  check(tuple(map(float,vals))==(w[f'{l}_obs'],w[f'{l}_exp'],w[f'{l}_exp_lm']),f'table 2 rank {rank}: {l} cell agrees')
for name in ['fig2b_lda_obs_vs_exp_lang.png', 'fig1_corpus_growth.png']:
 fig=Path('figures')/name
 check((R/'paper'/fig).read_bytes()==(R/'analysis/output'/fig).read_bytes(),'paper figure byte-identical to current analysis figure: '+name)
for entry in json.loads((R/'preprocessing/output/corpus_manifest.json').read_text())['files'][:2]:
 check(hashlib.sha256((R/entry['path']).read_bytes()).hexdigest()==entry['sha256'],'current corpus hash matches manifest: '+entry['path'])
print('DENOMINATORS',{'world':N,**sizes,'target_union':226,'signals':257,'language_mention':len(lm),'year':dict(years),'HCLT':62})
print('LIMIT: no model refit or permutation rerun; checked saved assignments, labels, denominators and current table cells.')
latest_stm_input=sorted((R/'verification').glob('stm_input_*.json'))[-1]
stm_sizes=json.loads(latest_stm_input.read_text())['set_sizes']
print('LATEST STM INPUT',latest_stm_input.name,stm_sizes)
check(stm_sizes=={'lang_'+l:n for l,n in sizes.items()},'latest STM input language denominators match current corpus')

print('PASS',len(checks),'checks')
