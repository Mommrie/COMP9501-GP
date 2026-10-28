"""Fixed-plan CSA illustration, intentionally distinct from the 2008 datasets."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from csa import CSA,PCA,accuracy_1nn


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    plan=json.loads((root/'plan.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(args.data.read_bytes()).hexdigest()==plan['dataset_sha256']
    args.output.mkdir(parents=True,exist_ok=True)
    raw=np.load(args.data)
    def down(x):return x.astype(np.float64).reshape(-1,14,2,14,2).mean((2,4))/255.
    train,test=down(raw['x_train']),down(raw['x_test']);y,yt=raw['y_train'],raw['y_test']
    rng=np.random.default_rng(9501);test_ids=np.concatenate([rng.permutation(np.flatnonzero(yt==c))[:plan['test_per_class']] for c in range(10)])
    result={'state':'running','plan':plan,'test_native_ids':test_ids.tolist(),'runs':[],'skipped':[],'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'code').glob('*.py')}}
    for seed in plan['seeds']:
        rng=np.random.default_rng(seed);pools=[rng.permutation(np.flatnonzero(y==c)) for c in range(10)]
        for n in plan['train_per_class']:
            ids=np.concatenate([p[:n] for p in pools]);x=train[ids];z=test[test_ids]
            raw_acc=accuracy_1nn(x,y[ids],z,yt[test_ids])
            for rank in plan['ranks']:
                if rank*rank>len(x)-1:
                    result['skipped'].append({'seed':seed,'n':n,'rank':rank,'reason':'matched PCA dimension exceeds centered sample rank'});continue
                run={'seed':seed,'n_per_class':n,'rank':rank,'train_native_ids':ids.tolist(),'raw_1nn_accuracy':raw_acc,'methods':{}}
                for name,model,basis_count in [('CSA',CSA(rank,plan['max_sweeps'],plan['relative_tolerance']),2*14*rank),('PCA',PCA(rank*rank),196*rank*rank)]:
                    start=time.perf_counter();model.fit(x);seconds=time.perf_counter()-start
                    r={'train_mse':float(np.square(x-model.reconstruct(x)).mean()),'test_mse':float(np.square(z-model.reconstruct(z)).mean()),'accuracy_1nn':accuracy_1nn(model.transform(x),y[ids],model.transform(z),yt[test_ids]),'basis_coefficients':basis_count,'fit_seconds':seconds}
                    if name=='CSA':r.update(history=model.history,sweeps=model.sweeps,converged=model.converged)
                    run['methods'][name]=r
                    arrays={'mean':model.mean,**({'u':model.u,'v':model.v} if name=='CSA' else {'basis':model.basis})}
                    np.savez(args.output/f's{seed}-n{n}-r{rank}-{name}.npz',**arrays)
                assert run['methods']['PCA']['train_mse']<=run['methods']['CSA']['train_mse']+1e-10
                result['runs'].append(run)
                (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
                print(seed,n,rank,{k:round(v['accuracy_1nn'],3) for k,v in run['methods'].items()},flush=True)
    result['state']='passed';(args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('CSA_ALL_PASSED')


if __name__=='__main__':main()
