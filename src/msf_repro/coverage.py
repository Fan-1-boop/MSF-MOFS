"""Report actual end-to-end rerun coverage; do not count archived data as new reruns."""
import json
from pathlib import Path
import pandas as pd
from .statistics import CORE, BREADTH, CARRIERS, INITIALIZERS


def report_coverage(root:Path, output:Path|None=None) -> dict:
    report={}
    for name, datasets in [('public_core',CORE),('public_breadth',BREADTH)]:
        filename=root/'results/reference/table3'/('core' if name=='public_core' else 'breadth')/'raw.csv'
        expect={(ds,seed,car,init) for ds in datasets for seed in range(92,102)
                for car in CARRIERS for init in INITIALIZERS}
        if filename.is_file():
            raw=pd.read_csv(filename)
            if not {'dataset','seed','carrier','init'}.issubset(raw.columns):
                raise ValueError(f'Invalid schema: {filename}')
            got=[(str(r.dataset),int(r.seed),str(r.carrier),str(r.init)) for r in raw.itertuples(index=False)]
            valid=set(got).intersection(expect)
            extras=set(got)-expect
            repeats=len(got)-len(set(got))
        else:
            valid=set();extras=set();repeats=0
        missing=expect-valid
        report[name]={'expected_rows':len(expect),'valid_rows':len(valid),
                      'extra_rows':len(extras),'duplicate_rows':repeats,
                      'complete':not missing and not extras and not repeats,
                      'by_dataset':{ds:{'completed':sum(r[0]==ds for r in valid),'total':40} for ds in datasets},
                      'missing_first_ten':[list(r) for r in sorted(missing)[:10]],
                      'path':str(filename)}
    report['full_public_search_complete']=all(report[x]['complete'] for x in ('public_core','public_breadth'))
    if output is not None:
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return report
