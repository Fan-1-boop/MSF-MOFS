"""Paper-facing, explicitly rounded checks; separate from exact per-seed archive checks."""
from pathlib import Path
import pandas as pd

# The fixed reported values below are transcribed from the CURRENT submitted PDF.
TABLE2={
 'madelon':{'MSF-MOFS':(.7555,.0098),'HFMOEA-CP':(.7554,.0098),'MBLPE-CP':(.6297,.0441)},
 'hill_valley':{'MSF-MOFS':(.3616,.0188),'HFMOEA-CP':(.3205,.0213),'MBLPE-CP':(.3464,.0229)},
 'musk1':{'MSF-MOFS':(.7075,.0237),'HFMOEA-CP':(.6996,.0269),'MBLPE-CP':(.6902,.0285)},
 'arcene':{'MSF-MOFS':(.6728,.0499),'HFMOEA-CP':(.6057,.0603),'MBLPE-CP':(.5665,.0567)},
}
TABLE4={'power':(.8054,.7222,.0832),'thermal':(.8807,.8343,.0464),
        'attitude':(.9195,.8814,.0380),'data':(.9511,.8245,.1266)}
POP50={'madelon':.4165,'hill_valley':.0312,'Prostate-GE':.4217}


def _four_dp_same(a,b):
    # PDFs show 4 decimal places; differences smaller than half of one last digit
    # should be accepted without treating rounded values as high-precision ground truth.
    return abs(float(a)-float(b)) < .00005001


def check_paper_tables(folder:Path, scope:str)->pd.DataFrame:
    base=Path(folder)
    if scope=='table2':
        data=pd.read_csv(base/'table2_recomputed.csv')
        rows=[]
        for ds, methods in TABLE2.items():
            for name, (m,s) in methods.items():
                match=data[(data.dataset==ds)&(data.method==name)]
                if len(match)!=1:raise ValueError(f'missing table2: {ds}/{name}')
                rec=match.iloc[0]
                rows.extend([{'table':'Table 2','dataset':ds,'method':name,'field':field,
                     'paper_rounded':reported,'computed':actual,'matches_4dp':_four_dp_same(reported,actual)}
                     for field,reported,actual in [('mean_hv',m,rec.mean_hv),('std_hv',s,rec.std_hv)]])
    elif scope=='table4':
        data=pd.read_csv(base/'table4_recomputed.csv')
        rows=[]
        for ds,(msf,rnd,delta) in TABLE4.items():
            r=data[data.subsystem==ds]
            if len(r)!=1:raise ValueError(ds)
            rec=r.iloc[0]
            rows.extend([{'table':'Table 4','dataset':ds,'method':'MSF vs Random','field':key,
                'paper_rounded':reported,'computed':actual,'matches_4dp':_four_dp_same(reported,actual)}
                for key,reported,actual in [('msf_mean',msf,rec.msf_mean),('random_mean',rnd,rec.random_mean),('mean_delta_hv',delta,rec.mean_delta_hv)]])
    elif scope=='population50':
        data=pd.read_csv(base/'population50_paired_paper_holm.csv')
        rows=[]
        for ds,expected in POP50.items():
            r=data[(data.dataset==ds)&(data.stage=='gen1')]
            if len(r)!=1:raise ValueError(ds)
            rows.append({'table':'Section 5.3','dataset':ds,'method':'MSF vs Random','field':'gen1_mean_delta_hv',
                         'paper_rounded':expected,'computed':r.iloc[0].mean_delta_hv,
                         'matches_4dp':_four_dp_same(expected,r.iloc[0].mean_delta_hv)})
    else:raise ValueError(scope)
    df=pd.DataFrame(rows)
    if not df.matches_4dp.all():raise AssertionError(f'Paper rounded mismatch {df[~df.matches_4dp]}')
    return df
