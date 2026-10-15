"""입력: 고정 말뭉치·표제어 캐시·기존 LDA 배정.
출력: 설정별 배정 캐시, verification/lda_sensitivity_20261015.json.
논문 위치: 4.1 토픽 변화, 4.2 한국어 연구 비교. 기존 정본은 보존한다.
"""
from pathlib import Path
import argparse, hashlib, json, time, sys, platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'analysis/output/cache/lda_sensitivity_20261015'
OUT = ROOT / 'verification/lda_sensitivity_20261015.json'
RUNS = ROOT / 'analysis/output/tables/lda_sensitivity_runs_20261015.json'
TARGETS = {'training': 6, 'software': 20, 'retrieval': 22}
YEARS = [2023, 2024, 2025, 2026]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def inputs():
    paths = {'corpus': ROOT/'preprocessing/output/corpus_llm_agent_all_2023_2026q3.csv',
             'lemmas': ROOT/'analysis/output/cache/lemmas_2023_2026q3.jsonl',
             'reference': ROOT/'verification/topics_lda_20261015.json',
             'assignments': ROOT/'analysis/output/tables/lda_doc_topics.csv',
             'topics': ROOT/'analysis/output/tables/lda_topics_K25.csv'}
    df = pd.read_csv(paths['corpus'], dtype={'id':str})
    old = pd.read_csv(paths['assignments'], dtype={'id':str}).set_index('id').loc[df.id]
    tab = pd.read_csv(paths['topics']).set_index('topic')
    ref = json.loads(paths['reference'].read_text())
    assert len(df) == 20107 and df.id.is_unique
    assert {l:int(df[f'lang_{l}'].sum()) for l in ['ko','zh','ja']} == {'ko':25,'zh':186,'ja':17}
    return paths, df, old.lda_topic.to_numpy(), tab, ref

def fit_all():
    paths, df, _, _, ref = inputs()
    lemmas = {d['id']:d['t'] for d in map(json.loads,paths['lemmas'].read_text().splitlines())}
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = {'input_sha256':{k:sha(p) for k,p in paths.items()}, 'ids':df.id.tolist()}
    for mode in ['current', 'keep_datum_cod']:
        stop = set(ENGLISH_STOP_WORDS)|set(ref['domain_stopwords'])
        if mode == 'keep_datum_cod': stop -= {'datum','cod'}
        cv = CountVectorizer(tokenizer=str.split,preprocessor=None,lowercase=False,token_pattern=None,min_df=10,max_df=.5)
        dtm = cv.fit_transform([' '.join(w for w in lemmas[i] if w not in stop) for i in df.id])
        vocab = cv.get_feature_names_out()
        for k in [25,10]:
            for seed in [42,43,44]:
                name=f'{mode}_k{k}_s{seed}'; file=CACHE/f'{name}.json'
                if file.exists():
                    saved=json.loads(file.read_text())
                    assert saved['input_sha256']==manifest['input_sha256']
                    print('reuse',name,flush=True); continue
                print('fit',name,'matrix',dtm.shape,flush=True); start=time.time()
                model=LatentDirichletAllocation(n_components=k,random_state=seed,learning_method='batch',max_iter=25,n_jobs=-1).fit(dtm)
                theta=model.transform(dtm)
                result={**manifest,'name':name,'mode':mode,'k':k,'seed':seed,'n_iter':int(model.n_iter_),
                    'vocab_size':len(vocab),'domain_stopwords':sorted(set(ref['domain_stopwords'])-({'datum','cod'} if mode!='current' else set())),
                    'elapsed_seconds':round(time.time()-start,2),'labels':theta.argmax(1).tolist(),
                    'top_words':[[str(vocab[j]) for j in c.argsort()[::-1][:12]] for c in model.components_],
                    'python':platform.python_version(),'sklearn':sklearn.__version__}
                file.write_text(json.dumps(result,ensure_ascii=False))
                print('done',name,result['elapsed_seconds'],flush=True)
    records = [json.loads(p.read_text()) for p in sorted(CACHE.glob('*.json'))]
    RUNS.write_text(json.dumps(records,ensure_ascii=False)+'\n')
    return summarize(records)

def jaccard(a,b):
    return len(a&b)/len(a|b) if a|b else 0.

def summarize(records=None):
    paths,df,old,tab,ref=inputs(); runs=[]
    masks={l:df[f'lang_{l}'].to_numpy(dtype=bool) for l in ['ko','zh','ja']}
    if records is None:
        records=json.loads(RUNS.read_text())
    for r in records:
        labels=np.array(r['labels']); assert r['ids']==df.id.tolist()
        assert r['input_sha256']=={k:sha(p) for k,p in paths.items()}
        targets={}
        for key,t in TARGETS.items():
            basewords=set(tab.loc[t,'top_words'].split(', ')[:10]); basemask=old==t
            word=[jaccard(basewords,set(w[:10])) for w in r['top_words']]
            docs=[jaccard(set(np.flatnonzero(basemask)),set(np.flatnonzero(labels==j))) for j in range(r['k'])]
            byword=int(np.argmax(word)); bydoc=int(np.argmax(docs)); candidates={}
            for method,j in [('top_words',byword),('documents',bydoc)]:
                m=labels==j; n=int(m.sum()); byyear={str(y):int((m&(df.year.to_numpy()==y)).sum()) for y in YEARS}
                share={str(y):100*byyear[str(y)]/int((df.year==y).sum()) for y in YEARS}
                nko=int((m&masks['ko']).sum()); nkobytopic=np.bincount(labels[masks['ko']],minlength=r['k'])
                candidates[method]={'topic':j,'top_words':r['top_words'][j],'word_jaccard':word[j],'document_jaccard':docs[j],
                    'overall_n':n,'language_n':{l:int((m&mask).sum()) for l,mask in masks.items()},
                    'ko_share_ratio_to_overall':nko/25/(n/20107) if n else None,
                    'ko_rank_min':1+int((nkobytopic>nko).sum()),'year_n':byyear,'year_share_pct':share,
                    'change_2026_minus_2023_pp':share['2026']-share['2023'],
                    'original_docs_retained':int((m&basemask).sum()),
                    'original_zh_docs_retained':int((m&basemask&masks['zh']).sum()),
                    'ko_ids':df.loc[m&masks['ko'],'id'].tolist(),'zh_ids':df.loc[m&masks['zh'],'id'].tolist()}
            targets[key]={'reference_topic':t,'matching_methods_agree':byword==bydoc,'matches':candidates,
                'reference_doc_destination_n':{str(j):int((basemask&(labels==j)).sum()) for j in range(r['k'])}}
        run={key:r[key] for key in ['name','mode','k','seed','elapsed_seconds','vocab_size','n_iter']}
        run.update({'targets':targets,'record_sha256':hashlib.sha256(json.dumps(r,ensure_ascii=False,sort_keys=True).encode()).hexdigest()})
        if r['name']=='current_k25_s42':
            run['baseline_reproduction']={'assignment_mismatches':int((old!=labels).sum()),
                'top_word_mismatches':sum(r['top_words'][i]!=tab.loc[i,'top_words'].split(', ') for i in range(25))}
        runs.append(run)
    return {'analysis':'fixed-corpus LDA sensitivity','n_docs':len(df),'language_n':{l:int(m.sum()) for l,m in masks.items()},
            'year_n':{str(y):int((df.year==y).sum()) for y in YEARS},'input_sha256':{k:sha(p) for k,p in paths.items()},
            'code_sha256':sha(Path(__file__)),'parameters':{'max_iter':25,'learning_method':'batch','seeds':[42,43,44],'k':[10,25]},
            'matching':'max top10 word Jaccard and max assigned-document Jaccard, reported separately',
            'runs':runs,'completed':len(runs)==12}

def main():
    p=argparse.ArgumentParser();p.add_argument('--summarize-only',action='store_true');p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    out=summarize() if a.summarize_only else fit_all()
    a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(a.out, 'runs',len(out['runs']),flush=True)
if __name__=='__main__': main()
